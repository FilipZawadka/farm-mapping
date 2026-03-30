"""PyTorch Dataset for Sentinel-2 farm-detection patches.

Handles:
- Loading .npy patches + labels from patch_meta.csv
- Region-based **or** random stratified train / val / test splits
- Configurable augmentations (flip, rotation, brightness jitter)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PipelineConfig, imagery_config_hash, matches_any_region, build_country_key_map

log = logging.getLogger(__name__)


def _rotate_array(arr: np.ndarray, angle_deg: float, fill_mode: str) -> np.ndarray:
    """Rotate (C, H, W) array by *angle_deg* degrees around center."""
    from scipy.ndimage import rotate as ndimage_rotate
    mode = "reflect" if fill_mode == "reflect" else "constant"
    rotated = ndimage_rotate(arr, angle_deg, axes=(1, 2), reshape=False, mode=mode, order=1)
    return rotated.astype(arr.dtype)


def _random_resized_crop(
    arr: np.ndarray, scale_min: float, scale_max: float, rng: np.random.Generator,
) -> np.ndarray:
    """Crop a random sub-region and resize back to original HxW."""
    from scipy.ndimage import zoom as ndimage_zoom
    _, h, w = arr.shape
    scale = rng.uniform(scale_min, scale_max)
    crop_h, crop_w = int(h * scale), int(w * scale)
    top = rng.integers(0, h - crop_h + 1)
    left = rng.integers(0, w - crop_w + 1)
    cropped = arr[:, top:top + crop_h, left:left + crop_w]
    zoom_h, zoom_w = h / crop_h, w / crop_w
    resized = ndimage_zoom(cropped, (1, zoom_h, zoom_w), order=1)
    # Ensure exact shape (zoom can be off by 1 pixel)
    return resized[:, :h, :w].astype(arr.dtype)


def _recompute_indices(arr: np.ndarray, n_spectral: int) -> np.ndarray:
    """Recompute NDVI/NDBI/NDWI from augmented spectral bands.

    Assumes band order: B2(0), B3(1), B4(2), B8(3), B11(4), B12(5).
    Index order: NDVI(n), NDBI(n+1), NDWI(n+2).
    """
    eps = 1e-8
    b3, b4, b8, b11 = arr[1], arr[2], arr[3], arr[4]
    arr[n_spectral] = (b8 - b4) / (b8 + b4 + eps)       # NDVI
    arr[n_spectral + 1] = (b11 - b8) / (b11 + b8 + eps)  # NDBI
    arr[n_spectral + 2] = (b3 - b8) / (b3 + b8 + eps)    # NDWI
    return arr


def _cutout(
    arr: np.ndarray, n_holes: int, hole_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Zero-fill rectangular holes in all channels."""
    _, h, w = arr.shape
    for _ in range(n_holes):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        y1 = max(0, cy - hole_size // 2)
        y2 = min(h, cy + hole_size // 2)
        x1 = max(0, cx - hole_size // 2)
        x2 = min(w, cx + hole_size // 2)
        arr[:, y1:y2, x1:x2] = 0.0
    return arr


class PatchDataset(Dataset):
    """Loads pre-extracted .npy patches and binary labels.

    Each item returns ``(tensor[C, H, W], label_int)``.
    patch_path in meta may be relative (e.g. ``{candidate_id}.npy``) or absolute;
    relative paths are resolved against *patches_dir*.
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        candidates: pd.DataFrame,
        patches_dir: Path,
        augment: bool = False,
        aug_config: "AugmentationConfig | None" = None,
        rng_seed: int = 42,
        n_spectral_bands: int = 6,
    ):
        self.meta = meta.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.aug_config = aug_config
        self.augment = augment or (aug_config is not None and aug_config.enabled)
        self.rng = np.random.default_rng(rng_seed)
        self.n_spectral_bands = n_spectral_bands

        label_map = dict(zip(
            candidates["id"].astype(str),
            candidates["label"].astype(int),
        ))
        self.labels = np.array([
            label_map.get(str(cid), 0) for cid in self.meta["candidate_id"]
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.meta.iloc[idx]
        raw_path = row["patch_path"]
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.patches_dir / path).resolve()
        else:
            path = path.resolve()
        arr = np.load(str(path)).astype(np.float32)

        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)

        # S2 SR bands are stored as raw 0-10000; scale to ~0-1 for the model.
        # Index bands (NDVI, NDBI, NDWI) are already in [-1, 1].
        if self.n_spectral_bands > 0:
            arr[:self.n_spectral_bands] /= 10_000.0

        if self.augment:
            arr = self._augment(arr)

        tensor = torch.from_numpy(arr)
        label = int(self.labels[idx])
        return tensor, label

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Config-driven augmentation pipeline for multi-spectral satellite patches."""
        cfg = self.aug_config
        n_s = self.n_spectral_bands

        # Fallback: if no config, use legacy defaults
        if cfg is None:
            if self.rng.random() < 0.5:
                arr = np.flip(arr, axis=1).copy()
            if self.rng.random() < 0.5:
                arr = np.flip(arr, axis=2).copy()
            k = self.rng.integers(0, 4)
            if k > 0:
                arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
            arr = arr * self.rng.uniform(0.9, 1.1)
            return arr

        # --- Geometric (all channels) ---
        if cfg.horizontal_flip.enabled and self.rng.random() < cfg.horizontal_flip.probability:
            arr = np.flip(arr, axis=1).copy()

        if cfg.vertical_flip.enabled and self.rng.random() < cfg.vertical_flip.probability:
            arr = np.flip(arr, axis=2).copy()

        if cfg.random_rotation_90.enabled and self.rng.random() < cfg.random_rotation_90.probability:
            k = self.rng.integers(1, 4)  # 1, 2, or 3
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()

        if cfg.continuous_rotation.enabled and self.rng.random() < cfg.continuous_rotation.probability:
            angle = self.rng.uniform(
                -cfg.continuous_rotation.max_degrees,
                cfg.continuous_rotation.max_degrees,
            )
            arr = _rotate_array(arr, angle, cfg.continuous_rotation.fill_mode)

        if cfg.random_resized_crop.enabled and self.rng.random() < cfg.random_resized_crop.probability:
            arr = _random_resized_crop(
                arr, cfg.random_resized_crop.scale_min,
                cfg.random_resized_crop.scale_max, self.rng,
            )

        # --- Spectral (raw bands only, channels 0..n_s-1) ---
        if cfg.brightness_jitter.enabled and self.rng.random() < cfg.brightness_jitter.probability:
            factor = self.rng.uniform(cfg.brightness_jitter.range_min, cfg.brightness_jitter.range_max)
            arr[:n_s] *= factor

        if cfg.per_band_jitter.enabled and self.rng.random() < cfg.per_band_jitter.probability:
            factors = self.rng.uniform(
                cfg.per_band_jitter.range_min, cfg.per_band_jitter.range_max,
                size=(n_s, 1, 1),
            ).astype(np.float32)
            arr[:n_s] *= factors

        if cfg.gaussian_noise.enabled and self.rng.random() < cfg.gaussian_noise.probability:
            noise = self.rng.normal(0, cfg.gaussian_noise.sigma, size=arr[:n_s].shape).astype(np.float32)
            arr[:n_s] += noise

        if cfg.channel_dropout.enabled and self.rng.random() < cfg.channel_dropout.probability:
            n_drop = self.rng.integers(1, cfg.channel_dropout.max_channels + 1)
            drop_idx = self.rng.choice(n_s, size=n_drop, replace=False)
            arr[drop_idx] = 0.0

        # Optionally recompute indices from augmented spectral bands
        if cfg.recompute_indices and n_s > 0 and arr.shape[0] > n_s:
            arr = _recompute_indices(arr, n_s)

        # Cutout (all channels)
        if cfg.cutout.enabled and self.rng.random() < cfg.cutout.probability:
            arr = _cutout(arr, cfg.cutout.n_holes, cfg.cutout.hole_size, self.rng)

        # Clamp to valid ranges
        arr[:n_s] = np.clip(arr[:n_s], 0.0, 1.0)
        if arr.shape[0] > n_s:
            arr[n_s:] = np.clip(arr[n_s:], -1.0, 1.0)

        return arr


# ---------------------------------------------------------------------------
# Region-based splitting
# ---------------------------------------------------------------------------

def _join_region(meta: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    """Return a ``region`` Series aligned with *meta* rows."""
    region_map = dict(zip(
        candidates["id"].astype(str),
        candidates["region"].astype(str) if "region" in candidates.columns else "",
    ))
    return meta["candidate_id"].astype(str).map(region_map).fillna("")


def _join_country_key_and_state(
    meta: pd.DataFrame, candidates: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return (country_key, state) Series aligned with *meta*."""
    from .config import build_country_key_map
    name_to_key = build_country_key_map()

    cid_to_country = dict(zip(
        candidates["id"].astype(str),
        candidates["country"].astype(str) if "country" in candidates.columns else "",
    ))
    cid_to_state = dict(zip(
        candidates["id"].astype(str),
        candidates["state"].fillna("").astype(str) if "state" in candidates.columns else "",
    ))

    countries = meta["candidate_id"].astype(str).map(cid_to_country).fillna("")
    keys = countries.map(name_to_key).fillna("")
    states = meta["candidate_id"].astype(str).map(cid_to_state).fillna("")
    return keys, states


def _country_split_map(
    train_regions: list[str],
    val_regions: list[str],
    test_regions: list[str],
) -> dict[str, str | None]:
    """For each country key, determine which split it belongs to unambiguously.

    A whole-country region (e.g. ``chile``) assigns that country to exactly one
    split.  A country that only appears as sub-regions (e.g. ``united_states/OH``)
    spans multiple splits, so stateless candidates must be distributed
    proportionally — return ``None`` for those.
    """
    # Collect which splits each country's *whole-country* regions appear in
    country_splits: dict[str, set[str]] = {}
    for split_name, regions in [("train", train_regions), ("val", val_regions), ("test", test_regions)]:
        for r in regions:
            country, state = r.split("/", 1) if "/" in r else (r, None)
            if state is None:
                # Whole-country region → this country belongs to this split
                country_splits.setdefault(country, set()).add(split_name)
            else:
                # Sub-region → mark country as spanning splits
                country_splits.setdefault(country, set())

    result: dict[str, str | None] = {}
    for country, splits in country_splits.items():
        if len(splits) == 1:
            result[country] = next(iter(splits))
        else:
            result[country] = None  # ambiguous → distribute proportionally
    return result


def _assign_by_region(
    keys: pd.Series, states: pd.Series,
    train_regions: list[str],
    val_regions: list[str],
    test_regions: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically assign rows to splits by region membership.

    Whole-country regions (e.g. ``chile``) assign all candidates from that
    country to the configured split.  Country-wide candidates (no state) from
    countries that span multiple splits (e.g. US negatives) are collected
    separately and distributed proportionally.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    countrywide_idx: list[int] = []

    csm = _country_split_map(train_regions, val_regions, test_regions)
    split_lists = {"train": train_idx, "val": val_idx, "test": test_idx}

    for i in range(len(keys)):
        k, s = str(keys.iloc[i]), str(states.iloc[i])

        if not s:
            # No state — check if country has a single unambiguous split
            target = csm.get(k)
            if target is not None:
                split_lists[target].append(i)
                continue
            # Country spans multiple splits (e.g. US negatives) → defer
            if k in csm:
                countrywide_idx.append(i)
                continue

        if matches_any_region(k, s, test_regions):
            test_idx.append(i)
        elif matches_any_region(k, s, val_regions):
            val_idx.append(i)
        elif matches_any_region(k, s, train_regions):
            train_idx.append(i)

    # Distribute country-wide candidates proportionally across splits
    if countrywide_idx:
        n_tr, n_va, n_te = len(train_idx), len(val_idx), len(test_idx)
        total = n_tr + n_va + n_te
        if total > 0:
            countrywide_idx.sort()
            n = len(countrywide_idx)
            n_cw_te = max(1, round(n * n_te / total)) if n_te > 0 else 0
            n_cw_va = max(1, round(n * n_va / total)) if n_va > 0 else 0
            test_idx.extend(countrywide_idx[:n_cw_te])
            val_idx.extend(countrywide_idx[n_cw_te:n_cw_te + n_cw_va])
            train_idx.extend(countrywide_idx[n_cw_te + n_cw_va:])
        else:
            train_idx.extend(countrywide_idx)

    return train_idx, val_idx, test_idx


def _pool_and_random_split(
    keys: pd.Series, states: pd.Series,
    train_regions: list[str], cfg: PipelineConfig, rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Filter to train_regions pool then randomly sub-split for val/test."""
    pool_idx = [
        i for i in range(len(keys))
        if matches_any_region(str(keys.iloc[i]), str(states.iloc[i]), train_regions)
    ]
    rng.shuffle(pool_idx)
    n = len(pool_idx)
    n_test = max(1, int(n * cfg.training.test_split))
    n_val = max(1, int(n * cfg.training.val_split))
    return pool_idx[n_test + n_val:], pool_idx[n_test: n_test + n_val], pool_idx[:n_test]


def _region_split_indices(
    meta: pd.DataFrame,
    candidates: pd.DataFrame,
    cfg: PipelineConfig,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Assign meta rows to train/val/test by region membership."""
    keys, states = _join_country_key_and_state(meta, candidates)
    train_regions = cfg.data.train_regions or []
    val_regions = cfg.data.val_regions
    test_regions = cfg.data.test_regions

    if val_regions and test_regions:
        tr, va, te = _assign_by_region(keys, states, train_regions, val_regions, test_regions)
    else:
        tr, va, te = _pool_and_random_split(keys, states, train_regions, cfg, rng)

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


# ---------------------------------------------------------------------------
# Random stratified splitting (original behaviour)
# ---------------------------------------------------------------------------

def _random_split_indices(
    meta: pd.DataFrame,
    cfg: PipelineConfig,
    rng: np.random.Generator,
    label_col: str = "_label",
) -> tuple[list[int], list[int], list[int]]:
    pos_idx = meta.index[meta[label_col] == 1].tolist()
    neg_idx = meta.index[meta[label_col] == 0].tolist()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split(indices: list[int]) -> tuple[list, list, list]:
        n = len(indices)
        n_test = max(1, int(n * cfg.training.test_split))
        n_val = max(1, int(n * cfg.training.val_split))
        return (
            indices[n_test + n_val:],
            indices[n_test: n_test + n_val],
            indices[:n_test],
        )

    pt, pv, pe = _split(pos_idx)
    nt, nv, ne = _split(neg_idx)

    train_idx = pt + nt
    val_idx = pv + nv
    test_idx = pe + ne
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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


def _find_patches_root(output_dir: Path) -> Path:
    """Walk up from *output_dir* to find the ``patches/`` ancestor."""
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == "patches":
            return parent
    return output_dir.parent


def build_splits(
    cfg: PipelineConfig,
    patches_dir: Optional[Path] = None,
) -> tuple[PatchDataset, PatchDataset, PatchDataset]:
    """Load metadata + candidates and create train/val/test datasets.

    When ``train_regions`` is configured, rows are assigned to splits by
    geographic region. Otherwise the legacy random stratified split is used.

    If *patches_dir* is given, use it as the patch root; else derive
    the ``patches/`` ancestor from ``cfg.patches.output_dir``.

    Returns ``(train_ds, val_ds, test_ds)``.
    """
    output_dir = Path(cfg.patches.output_dir)
    if patches_dir is not None:
        patches_root = Path(patches_dir)
    else:
        patches_root = _find_patches_root(output_dir)

    meta_path = patches_root / "patch_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        meta_parquet = patches_root / "patch_meta.parquet"
        if meta_parquet.exists():
            meta = pd.read_parquet(meta_parquet)
        else:
            raise FileNotFoundError(
                f"No patch_meta.csv or patch_meta.parquet found in {patches_root}"
            )

    # Filter to patches matching current imagery config (bands, date_range, etc.)
    if "imagery_config_hash" in meta.columns:
        current_hash = imagery_config_hash(cfg.patches)
        meta = meta[meta["imagery_config_hash"] == current_hash].reset_index(drop=True)
        if len(meta) == 0:
            raise FileNotFoundError(
                f"No patches with imagery_config_hash={current_hash} in {patches_root}. "
                "Re-run patch extraction with the current config."
            )
        log.info("Filtered to %d patches matching imagery_config_hash=%s", len(meta), current_hash)

    candidates = _load_candidates_csv(cfg.data.candidates_dir, cfg.data.countries)
    if len(candidates) == 0:
        cand_parquet = output_dir / "candidates.parquet"
        if cand_parquet.exists():
            candidates = pd.read_parquet(cand_parquet)
        else:
            raise FileNotFoundError(
                f"No candidates found in {cfg.data.candidates_dir} or {cand_parquet}"
            )

    label_map = dict(zip(
        candidates["id"].astype(str),
        candidates["label"].astype(int),
    ))
    # Only keep patches that have a matching candidate — avoids leaking
    # data from other configs that share the same patch_meta.csv.
    valid_ids = set(candidates["id"].astype(str))
    meta = meta[meta["candidate_id"].astype(str).isin(valid_ids)].reset_index(drop=True)
    log.info("Filtered to %d patches matching config candidates", len(meta))

    meta["_label"] = meta["candidate_id"].astype(str).map(label_map).astype(int)

    rng = np.random.default_rng(cfg.training.seed)

    # If inspected_as_test, hold out inspected candidates as a separate eval set
    inspected_as_test = getattr(cfg.data, "inspected_as_test", False)
    inspected_idx: list[int] = []
    if inspected_as_test and "viz_status" in candidates.columns:
        viz_map = dict(zip(candidates["id"].astype(str), candidates["viz_status"].fillna("")))
        meta["_viz_status"] = meta["candidate_id"].astype(str).map(viz_map).fillna("")
        inspected_mask = meta["_viz_status"] == "inspected"
        inspected_idx = list(meta.index[inspected_mask])
        remaining_idx = list(meta.index[~inspected_mask])
        n_inspected = len(inspected_idx)
        log.info("Inspected hold-out: %d candidates (excluded from train/val/test)", n_inspected)

        # Split remaining into train/val/test normally
        remaining_meta = meta.iloc[remaining_idx].reset_index(drop=True)
        remaining_labels = remaining_meta["_label"].values
        pos_idx = [i for i, l in enumerate(remaining_labels) if l == 1]
        neg_idx = [i for i, l in enumerate(remaining_labels) if l == 0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_pte = int(len(pos_idx) * cfg.training.test_split)
        n_nte = int(len(neg_idx) * cfg.training.test_split)
        n_pv = int(len(pos_idx) * cfg.training.val_split)
        n_nv = int(len(neg_idx) * cfg.training.val_split)
        test_local = pos_idx[:n_pte] + neg_idx[:n_nte]
        val_local = pos_idx[n_pte:n_pte + n_pv] + neg_idx[n_nte:n_nte + n_nv]
        train_local = pos_idx[n_pte + n_pv:] + neg_idx[n_nte + n_nv:]
        rng.shuffle(test_local)
        rng.shuffle(val_local)
        rng.shuffle(train_local)
        # Map back to original meta indices
        train_idx = [remaining_idx[i] for i in train_local]
        val_idx = [remaining_idx[i] for i in val_local]
        test_idx = [remaining_idx[i] for i in test_local]
        meta.drop(columns=["_viz_status"], inplace=True)
    elif bool(cfg.data.train_regions):
        train_idx, val_idx, test_idx = _region_split_indices(
            meta, candidates, cfg, rng,
        )
    else:
        train_idx, val_idx, test_idx = _random_split_indices(
            meta, cfg, rng,
        )

    meta_clean = meta.drop(columns=["_label"])

    split_mode = "inspected_holdout" if inspected_idx else ("region" if bool(cfg.data.train_regions) else "random")
    log.info(
        "Splits (%s): train=%d  val=%d  test=%d  inspected=%d  (pos ratio: %.2f)",
        split_mode,
        len(train_idx),
        len(val_idx),
        len(test_idx),
        len(inspected_idx),
        meta["_label"].mean(),
    )

    # Persist split assignments so we can colour the map later
    split_col = pd.Series("unassigned", index=meta_clean.index)
    split_col.iloc[train_idx] = "train"
    split_col.iloc[val_idx] = "val"
    split_col.iloc[test_idx] = "test"
    if inspected_idx:
        split_col.iloc[inspected_idx] = "inspected"
    splits_df = meta_clean[["candidate_id"]].copy()
    splits_df["split"] = split_col
    splits_dir = patches_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / f"{cfg._config_stem}.csv"
    splits_df.to_csv(splits_path, index=False)
    log.info("Saved split assignments to %s", splits_path)

    n_spectral = len(cfg.patches.bands)

    aug_cfg = getattr(cfg.training, "augmentation", None)
    train_ds = PatchDataset(meta_clean.iloc[train_idx], candidates, patches_root, aug_config=aug_cfg, rng_seed=cfg.training.seed, n_spectral_bands=n_spectral)
    val_ds = PatchDataset(meta_clean.iloc[val_idx], candidates, patches_root, augment=False, rng_seed=cfg.training.seed, n_spectral_bands=n_spectral)
    test_ds = PatchDataset(meta_clean.iloc[test_idx], candidates, patches_root, augment=False, rng_seed=cfg.training.seed, n_spectral_bands=n_spectral)

    inspected_ds = None
    if inspected_idx:
        inspected_ds = PatchDataset(meta_clean.iloc[inspected_idx], candidates, patches_root, augment=False, rng_seed=cfg.training.seed, n_spectral_bands=n_spectral)

    # Compute per-sample weights for region upsampling
    if cfg.training.upsample_minority_regions:
        train_ds.sample_weights = _compute_region_weights(
            meta_clean.iloc[train_idx], candidates,
        )

    return train_ds, val_ds, test_ds, inspected_ds


def _compute_region_weights(
    meta: pd.DataFrame, candidates: pd.DataFrame,
) -> np.ndarray:
    """Compute per-sample weights so each country contributes equally per epoch.

    Samples from countries with fewer patches get higher weight, so that
    a ``WeightedRandomSampler`` draws roughly the same number of samples
    from each country.
    """
    name_to_key = build_country_key_map()
    cid_to_country = dict(zip(
        candidates["id"].astype(str),
        candidates["country"].astype(str) if "country" in candidates.columns else "",
    ))
    countries = meta["candidate_id"].astype(str).map(cid_to_country).fillna("unknown")
    country_keys = countries.map(name_to_key).fillna(countries)

    counts = country_keys.value_counts()
    n_countries = len(counts)
    total = len(meta)

    # Weight = total / (n_countries * count_for_this_country)
    # This makes each country's total weight equal to total / n_countries
    weight_map = {c: total / (n_countries * n) for c, n in counts.items()}
    weights = country_keys.map(weight_map).values.astype(np.float64)

    log.info(
        "Region upsampling weights: %s",
        ", ".join(f"{c}={w:.2f} (n={n})" for c, (n, w) in
                  zip(counts.index, zip(counts.values, [weight_map[c] for c in counts.index]))),
    )

    return weights
