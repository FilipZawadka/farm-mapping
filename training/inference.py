"""Score all candidate patches with a trained model.

Outputs ``scored_candidates.parquet`` with prediction scores and confidence tiers.

Usage::

    python -m training.inference --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point
from torch.utils.data import DataLoader

from .config import PipelineConfig, load_config, resolve_paths
from .dataset import PatchDataset
from .model import build_model

log = logging.getLogger(__name__)


def _assign_confidence(
    scores: np.ndarray, cfg: PipelineConfig
) -> list[str]:
    """Map raw probability scores to confidence tier labels."""
    tiers = cfg.inference.confidence_tiers
    result = []
    for s in scores:
        if s >= tiers.high:
            result.append("high")
        elif s >= tiers.medium:
            result.append("medium")
        elif s >= tiers.low:
            result.append("low")
        else:
            result.append("very_low")
    return result


def _load_model(cfg: PipelineConfig, device: torch.device):
    model = build_model(cfg.model).to(device)
    ckpt_path = Path(cfg.inference.checkpoint)
    if not ckpt_path.exists():
        # Config-specific output directory
        ckpt_path = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem / "best_model.pt"
    if not ckpt_path.exists():
        # Legacy fallback
        ckpt_path = Path(cfg.patches.output_dir).parent / "output" / "best_model.pt"
    from .train import load_checkpoint
    model.load_state_dict(load_checkpoint(ckpt_path, device)["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint: %s", ckpt_path)
    return model


@torch.no_grad()
def _run_inference(model, loader, device, threshold, num_classes: int = 2):
    """Run inference. Returns (scores, preds, probs_matrix_or_None).

    Binary (num_classes=2):
        scores = P(class=1), preds = (scores >= threshold).
    Multi-class (num_classes>=3):
        scores = top-1 probability, preds = argmax over classes.
        Per-class probabilities are also returned for downstream coloring.
    """
    is_multi = num_classes >= 3
    all_scores, all_preds, all_probs = [], [], []
    for batch_x, _ in loader:
        probs = torch.softmax(model(batch_x.to(device)), dim=1).cpu().numpy()
        if is_multi:
            preds = probs.argmax(axis=1)
            top1 = probs.max(axis=1)
            all_preds.append(preds)
            all_scores.append(top1)
            all_probs.append(probs)
        else:
            pos = probs[:, 1]
            all_scores.append(pos)
            all_preds.append((pos >= threshold).astype(int))
    probs_matrix = np.vstack(all_probs) if is_multi else None
    return np.concatenate(all_scores), np.concatenate(all_preds), probs_matrix


def _attach_labels(result, candidates):
    cid_str = candidates["id"].astype(str)
    for col, src_col in [("true_label", "label"), ("source", "source"), ("country", "country")]:
        mapping = dict(zip(cid_str, candidates[src_col]))
        fill = 0 if col == "true_label" else "unknown"
        result[col] = result["candidate_id"].astype(str).map(mapping).fillna(fill)
    result["true_label"] = result["true_label"].astype(int)

    # Propagate diagnostic columns from the source parquet so reviewers can
    # audit "bad labels" without having to re-join. Each column is optional;
    # only attach the ones the candidate CSV actually carries.
    for diag_col in (
        "original_label", "standardized_label", "visual_label",
        "label_source", "notes", "eval_set", "random_sample", "viz_status",
    ):
        if diag_col not in candidates.columns:
            continue
        mapping = dict(zip(cid_str, candidates[diag_col]))
        fill = 0 if diag_col in ("eval_set", "random_sample") else ""
        result[diag_col] = result["candidate_id"].astype(str).map(mapping).fillna(fill)


def _find_patches_root(output_dir: Path) -> Path:
    """Walk up from *output_dir* to find the ``patches/`` ancestor."""
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == "patches":
            return parent
    return output_dir.parent


def _load_candidates_csv(candidates_dir: str | Path, countries: list[str]) -> pd.DataFrame:
    """Load candidate CSVs from ``{candidates_dir}/{country}.csv``.

    If *countries* is empty, loads all CSV files in the directory.
    """
    cdir = Path(candidates_dir)
    frames: list[pd.DataFrame] = []
    if countries:
        for country in countries:
            csv_path = cdir / f"{country}.csv"
            if csv_path.exists():
                frames.append(pd.read_csv(csv_path))
    else:
        for csv_path in sorted(cdir.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def score_candidates(cfg: PipelineConfig) -> gpd.GeoDataFrame:
    """Run inference on all extracted patches and return scored GeoDataFrame."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.patches.output_dir)
    patches_root = _find_patches_root(output_dir)

    meta_path = patches_root / "patch_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        meta = pd.read_parquet(patches_root / "patch_meta.parquet")

    if "imagery_config_hash" in meta.columns:
        from .config import imagery_config_hash
        current_hash = imagery_config_hash(cfg.patches)
        meta = meta[meta["imagery_config_hash"] == current_hash].reset_index(drop=True)
        if len(meta) == 0:
            raise FileNotFoundError(
                f"No patches with imagery_config_hash={current_hash}. Re-run patch extraction."
            )

    candidates = _load_candidates_csv(cfg.data.candidates_dir, cfg.data.countries)
    if len(candidates) == 0:
        cand_parquet = output_dir / "candidates.parquet"
        if cand_parquet.exists():
            candidates = pd.read_parquet(cand_parquet)

    # Optional: skip the ~124k unlabelled rest-of-world candidates so we only
    # score the ~17k labelled slices (train/val/test/inspected/eval/gen).
    # Roughly 8x faster; the world map won't have UP/UN points.
    labeled_only = getattr(cfg.inference, "labeled_only", False)
    if labeled_only and "label" in candidates.columns:
        before = len(candidates)
        candidates = candidates[candidates["label"].astype(int) != -1].copy()
        log.info(
            "labeled_only=True: filtered candidates from %d to %d (dropped %d unlabelled)",
            before, len(candidates), before - len(candidates),
        )

    # Filter meta to only patches matching this config's candidates
    config_ids = set(candidates["id"].astype(str))
    meta = meta[meta["candidate_id"].astype(str).isin(config_ids)].reset_index(drop=True)
    log.info("Filtered to %d patches matching config candidates", len(meta))

    valid_ids = set(meta["candidate_id"].astype(str))
    cands_filtered = candidates[candidates["id"].astype(str).isin(valid_ids)].copy()

    # Apply same channel_subset and crop as training
    channel_indices = None
    n_spectral = len(cfg.patches.bands)
    channel_subset = getattr(cfg.training, "channel_subset", None)
    if channel_subset:
        from .config import resolve_channel_indices
        channel_indices, n_spectral = resolve_channel_indices(
            channel_subset, cfg.patches.bands, cfg.patches.indices,
        )
    crop_size = getattr(cfg.training, "crop_center_px", None)

    ds = PatchDataset(meta, cands_filtered, patches_root, augment=False,
                      n_spectral_bands=n_spectral, channel_indices=channel_indices,
                      crop_size=crop_size)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    # Compute effective input_channels
    if channel_subset:
        cfg.model.input_channels = len(channel_subset)

    model = _load_model(cfg, device)
    num_classes = getattr(cfg.model, "num_classes", 2)
    scores_arr, preds_arr, probs_matrix = _run_inference(
        model, loader, device, cfg.inference.threshold, num_classes=num_classes,
    )

    result = meta[["candidate_id", "lat", "lng"]].copy()
    result["predicted_score"] = scores_arr
    result["predicted_label"] = preds_arr
    result["confidence_tier"] = _assign_confidence(scores_arr, cfg)
    if probs_matrix is not None:
        # Multi-class: store per-class probabilities for downstream coloring/sorting.
        for i in range(probs_matrix.shape[1]):
            result[f"prob_class{i}"] = probs_matrix[:, i]
    _attach_labels(result, candidates)

    # Attach split assignments if available (config-specific)
    splits_path = patches_root / "splits" / f"{cfg._config_stem}.csv"
    if not splits_path.exists():
        # Fallback to legacy shared path
        splits_path = patches_root / "split_assignments.csv"
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        split_map = dict(zip(splits["candidate_id"].astype(str), splits["split"]))
        result["split"] = result["candidate_id"].astype(str).map(split_map).fillna("unknown")
        log.info("Attached split assignments (train/val/test) from %s", splits_path)
    elif getattr(cfg.data, "inspected_only", False):
        # Inference-only on inspected clusters — no splits file expected.
        result["split"] = "inspected"
    else:
        result["split"] = "unknown"

    geometry = [Point(lng, lat) for lng, lat in zip(result["lng"], result["lat"])]
    scored_gdf = gpd.GeoDataFrame(result, geometry=geometry, crs="EPSG:4326")

    # Save to config-specific output directory.
    # Write a dated copy alongside the canonical name so previous runs are
    # preserved -- `ls` shows when each "latest" was made.
    import shutil
    from datetime import datetime

    scored_dir = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem
    scored_dir.mkdir(parents=True, exist_ok=True)
    output_path = scored_dir / "scored_candidates.parquet"
    scored_gdf.to_parquet(output_path, index=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dated_path = scored_dir / f"scored_candidates_{stamp}.parquet"
    shutil.copy2(output_path, dated_path)
    log.info("Saved %d scored candidates to %s (+ %s)", len(scored_gdf), output_path, dated_path.name)
    return scored_gdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with trained model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--labeled-only", action="store_true",
        help=(
            "Only score candidates with a label (drops the ~124k rest-of-world "
            "unlabelled rows). ~8x faster; the world map won't have UP/UN "
            "points but every metric slice is unaffected."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    if args.labeled_only:
        cfg.inference.labeled_only = True
    score_candidates(cfg)


if __name__ == "__main__":
    main()
