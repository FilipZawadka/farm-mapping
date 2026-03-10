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
        ckpt_path = Path(cfg.patches.output_dir).parent / "output" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    log.info("Loaded checkpoint: %s", ckpt_path)
    return model


@torch.no_grad()
def _run_inference(model, loader, device, threshold):
    all_scores, all_preds = [], []
    for batch_x, _ in loader:
        probs = torch.softmax(model(batch_x.to(device)), dim=1)
        pos = probs[:, 1].cpu().numpy()
        all_scores.append(pos)
        all_preds.append((pos >= threshold).astype(int))
    return np.concatenate(all_scores), np.concatenate(all_preds)


def _attach_labels(result, candidates):
    cid_str = candidates["id"].astype(str)
    for col, src_col in [("true_label", "label"), ("source", "source"), ("country", "country")]:
        mapping = dict(zip(cid_str, candidates[src_col]))
        fill = 0 if col == "true_label" else "unknown"
        result[col] = result["candidate_id"].astype(str).map(mapping).fillna(fill)
    result["true_label"] = result["true_label"].astype(int)


@torch.no_grad()
def score_candidates(cfg: PipelineConfig) -> gpd.GeoDataFrame:
    """Run inference on all extracted patches and return scored GeoDataFrame."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches_dir = Path(cfg.patches.output_dir)
    meta = pd.read_parquet(patches_dir / "patch_meta.parquet")
    candidates = pd.read_parquet(patches_dir / "candidates.parquet")

    valid_ids = set(meta["candidate_id"].astype(str))
    cands_filtered = candidates[candidates["id"].astype(str).isin(valid_ids)].copy()

    ds = PatchDataset(meta, cands_filtered, patches_dir, augment=False)
    loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = _load_model(cfg, device)
    scores_arr, preds_arr = _run_inference(model, loader, device, cfg.inference.threshold)

    result = meta[["candidate_id", "lat", "lng"]].copy()
    result["predicted_score"] = scores_arr
    result["predicted_label"] = preds_arr
    result["confidence_tier"] = _assign_confidence(scores_arr, cfg)
    _attach_labels(result, candidates)

    geometry = [Point(lng, lat) for lng, lat in zip(result["lng"], result["lat"])]
    scored_gdf = gpd.GeoDataFrame(result, geometry=geometry, crs="EPSG:4326")

    output_path = patches_dir / "scored_candidates.parquet"
    scored_gdf.to_parquet(output_path, index=False)
    log.info("Saved %d scored candidates to %s", len(scored_gdf), output_path)
    return scored_gdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with trained model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    score_candidates(cfg)


if __name__ == "__main__":
    main()
