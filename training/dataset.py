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
        channel_indices: list[int] | None = None,
        crop_size: int | None = None,
    ):
        self.meta = meta.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.aug_config = aug_config
        self.augment = augment or (aug_config is not None and aug_config.enabled)
        self.rng = np.random.default_rng(rng_seed)
        self.n_spectral_bands = n_spectral_bands
        self.channel_indices = channel_indices
        self.crop_size = crop_size

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

        # Center-crop to smaller spatial extent (after augmentation)
        if self.crop_size is not None:
            _, h, w = arr.shape
            y = (h - self.crop_size) // 2
            x = (w - self.crop_size) // 2
            arr = arr[:, y:y + self.crop_size, x:x + self.crop_size].copy()

        # Select channel subset (after scaling + augmentation)
        if self.channel_indices is not None:
            arr = arr[self.channel_indices].copy()

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


def _country_balanced_split_indices(
    meta: pd.DataFrame,
    candidates: pd.DataFrame,
    cfg: PipelineConfig,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Split so val and test have equal samples from each country.

    The number of samples per country in val/test is determined by the
    smallest country: ``n_per_country = int(min_country_size * split_frac)``.
    Every country contributes exactly that many samples to val and test.
    The rest goes to train (larger countries contribute more to train).
    """
    cid_to_country = dict(zip(
        candidates["id"].astype(str),
        candidates["country"].astype(str) if "country" in candidates.columns else "",
    ))
    meta_countries = meta["candidate_id"].astype(str).map(cid_to_country).fillna("unknown")

    # Group indices by country
    country_groups: dict[str, list[int]] = {}
    for country in sorted(meta_countries.unique()):
        indices = meta.index[meta_countries == country].tolist()
        rng.shuffle(indices)
        country_groups[country] = indices

    # Determine samples per country from the smallest country
    min_country_size = min(len(v) for v in country_groups.values())
    n_val_per = max(1, int(min_country_size * cfg.training.val_split))
    n_test_per = max(1, int(min_country_size * cfg.training.test_split))

    train_idx, val_idx, test_idx = [], [], []
    for country, indices in country_groups.items():
        test_idx.extend(indices[:n_test_per])
        val_idx.extend(indices[n_test_per:n_test_per + n_val_per])
        train_idx.extend(indices[n_test_per + n_val_per:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    n_countries = len(country_groups)
    log.info(
        "Country-balanced splits: %d countries, %d val/country, %d test/country "
        "(from smallest=%d, val_split=%.2f, test_split=%.2f)",
        n_countries, n_val_per, n_test_per,
        min_country_size, cfg.training.val_split, cfg.training.test_split,
    )

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

    # Rachel's representative-sample eval set: rows flagged eval_set=True must
    # NEVER enter train/val/test/inspected. They get their own "eval" split.
    eval_idx: list[int] = []
    if "eval_set" in candidates.columns:
        eval_map = dict(zip(
            candidates["id"].astype(str),
            candidates["eval_set"].fillna(0).astype(int),
        ))
        meta["_eval"] = meta["candidate_id"].astype(str).map(eval_map).fillna(0).astype(int)
        eval_idx = list(meta.index[meta["_eval"] == 1])
        if eval_idx:
            log.info("eval_set hold-out: %d candidates (excluded from train/val/test/inspected)", len(eval_idx))
    eval_set_filter = set(eval_idx)

    # Generalization-testing countries: any labelled candidate whose ADM0 is
    # in cfg.data.generalization_countries is forced into a "generalization"
    # split. Never enters train/val/test/eval/inspected. See
    # docs/EVAL_FRAMEWORK.md.
    gen_idx: list[int] = []
    gen_iso = {iso.upper() for iso in (getattr(cfg.data, "generalization_countries", []) or [])}
    if gen_iso:
        # Candidate IDs follow the `{ADM0}_cluster_{n}` convention from
        # rachel_to_candidates.py, so the prefix before the first underscore
        # is the ADM0 ISO code.
        meta_iso = (
            meta["candidate_id"].astype(str)
            .str.split("_", n=1, expand=True)[0].str.upper()
        )
        gen_mask = meta_iso.isin(gen_iso) & (meta["_label"] != -1)
        gen_idx = list(meta.index[gen_mask])
        if gen_idx:
            log.info(
                "generalization hold-out: %d candidates from %s "
                "(excluded from train/val/test/eval/inspected)",
                len(gen_idx), sorted(gen_iso),
            )
    gen_set_filter = set(gen_idx)

    # Generalization takes priority over eval_set: any candidate that is
    # both eval_set=True AND in a generalization country goes to the
    # generalization split, not eval. Otherwise eval_ds would be
    # contaminated with OOD rows (Rachel's representative sample is only
    # meaningful inside the training countries).
    if gen_set_filter and eval_idx:
        before = len(eval_idx)
        eval_idx = [i for i in eval_idx if i not in gen_set_filter]
        eval_set_filter = set(eval_idx)
        if before != len(eval_idx):
            log.info(
                "eval_set ∩ generalization: %d rows reassigned to generalization",
                before - len(eval_idx),
            )

    rng = np.random.default_rng(cfg.training.seed)

    # Pool of rows eligible for train/val/test: labeled (_label != -1) and not
    # in the eval_set hold-out. Restricting the splitter inputs to this pool
    # keeps unlabeled rows (carried by include_unlabeled=true) from claiming
    # val/test slots in country-balanced splits.
    labeled_pool_mask = (meta["_label"] != -1)
    if eval_set_filter:
        labeled_pool_mask = labeled_pool_mask & ~meta.index.isin(eval_set_filter)
    if gen_set_filter:
        labeled_pool_mask = labeled_pool_mask & ~meta.index.isin(gen_set_filter)

    # Strict whitelist: when cfg.data.training_countries is set, any labelled
    # row whose ADM0 is NOT in (training_countries ∪ generalization_countries)
    # is removed from the labeled pool -- it will end up in the "unlabeled"
    # split. This enforces Rachel's framework (see docs/EVAL_FRAMEWORK.md):
    # only the named training countries contribute to train/val/test/inspected.
    train_iso = {iso.upper() for iso in (getattr(cfg.data, "training_countries", []) or [])}
    if train_iso:
        allowed = train_iso | gen_iso  # gen rows are already routed elsewhere
        meta_iso_all = (
            meta["candidate_id"].astype(str)
            .str.split("_", n=1, expand=True)[0].str.upper()
        )
        non_allowed_mask = labeled_pool_mask & ~meta_iso_all.isin(allowed)
        n_demoted = int(non_allowed_mask.sum())
        if n_demoted:
            log.info(
                "training-country whitelist: demoting %d labelled rows outside %s "
                "to the unlabeled split",
                n_demoted, sorted(train_iso),
            )
            labeled_pool_mask = labeled_pool_mask & ~non_allowed_mask
            meta.loc[non_allowed_mask, "_label"] = -1

    # DMV protection: Rachel reserves rows whose label_source contains "DMV"
    # for IF fitting. To prevent val/test from being inflated by this easy
    # clean subset, force DMV rows into the train split. Their labels stay
    # available for training (CNN train ≡ IF fit) but they never enter the
    # holdout slices.
    dmv_idx: list[int] = []
    if getattr(cfg.data, "dmv_force_to_train_only", False) and "label_source" in candidates.columns:
        ls_map = dict(zip(candidates["id"].astype(str), candidates["label_source"].fillna("")))
        dmv_mask = (
            meta["candidate_id"].astype(str).map(ls_map).fillna("")
            .astype(str).str.contains("DMV", case=False, na=False)
        )
        dmv_idx = list(meta.index[dmv_mask & labeled_pool_mask])
        if dmv_idx:
            log.info("DMV protection: %d rows pinned to train", len(dmv_idx))
    dmv_set = set(dmv_idx)

    labeled_meta = meta[labeled_pool_mask]

    # If inspected_as_test, hold out inspected candidates as a separate eval set
    inspected_as_test = getattr(cfg.data, "inspected_as_test", False)
    inspected_idx: list[int] = []
    if inspected_as_test and "viz_status" in candidates.columns:
        viz_map = dict(zip(candidates["id"].astype(str), candidates["viz_status"].fillna("")))
        meta["_viz_status"] = meta["candidate_id"].astype(str).map(viz_map).fillna("")
        # Only count LABELED rows as "inspected" so the metric is meaningful;
        # unlabeled inspected rows fall through to the "unlabeled" split.
        inspected_mask = (meta["_viz_status"] == "inspected") & labeled_pool_mask
        inspected_idx = list(meta.index[inspected_mask])
        remaining_idx = list(meta.index[labeled_pool_mask & ~inspected_mask])
        n_inspected = len(inspected_idx)
        log.info("Inspected hold-out: %d candidates (excluded from train/val/test)", n_inspected)

        # Split remaining into train/val/test
        balanced = getattr(cfg.training, "balanced_country_splits", False)
        if balanced:
            # Build a temporary meta with remaining indices for country-balanced splitting
            remaining_meta = meta.iloc[remaining_idx].copy()
            remaining_meta.index = range(len(remaining_meta))
            tr_local, va_local, te_local = _country_balanced_split_indices(
                remaining_meta, candidates, cfg, rng,
            )
            train_idx = [remaining_idx[i] for i in tr_local]
            val_idx = [remaining_idx[i] for i in va_local]
            test_idx = [remaining_idx[i] for i in te_local]
        else:
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
            train_idx = [remaining_idx[i] for i in train_local]
            val_idx = [remaining_idx[i] for i in val_local]
            test_idx = [remaining_idx[i] for i in test_local]
        meta.drop(columns=["_viz_status"], inplace=True)
    elif bool(cfg.data.train_regions):
        train_idx, val_idx, test_idx = _region_split_indices(
            labeled_meta, candidates, cfg, rng,
        )
    elif getattr(cfg.training, "balanced_country_splits", False):
        # Re-index to 0..N-1 so the splitter's positional logic is correct;
        # then map back to original meta indices.
        lm = labeled_meta.copy(); lm.index = range(len(lm))
        labeled_orig = labeled_meta.index.tolist()
        tr_local, va_local, te_local = _country_balanced_split_indices(
            lm, candidates, cfg, rng,
        )
        train_idx = [labeled_orig[i] for i in tr_local]
        val_idx = [labeled_orig[i] for i in va_local]
        test_idx = [labeled_orig[i] for i in te_local]
    else:
        train_idx, val_idx, test_idx = _random_split_indices(
            labeled_meta, cfg, rng,
        )

    # DMV pin: any DMV row that landed in val/test/inspected is pulled into train.
    if dmv_set:
        moved_v = [i for i in val_idx if i in dmv_set]
        moved_t = [i for i in test_idx if i in dmv_set]
        moved_i = [i for i in inspected_idx if i in dmv_set]
        if moved_v or moved_t or moved_i:
            log.info(
                "DMV protection: moved %d rows from val/test/inspected to train",
                len(moved_v) + len(moved_t) + len(moved_i),
            )
            val_idx = [i for i in val_idx if i not in dmv_set]
            test_idx = [i for i in test_idx if i not in dmv_set]
            inspected_idx = [i for i in inspected_idx if i not in dmv_set]
            train_idx = list(dict.fromkeys(train_idx + moved_v + moved_t + moved_i))

    # Belt-and-braces: strip any eval_set / generalization / unlabeled rows
    # that somehow survived the input-filter (e.g. region-split path with mixed
    # inputs).
    strip_set = (
        set(eval_set_filter)
        | set(gen_set_filter)
        | set(meta.index[meta["_label"] == -1])
    )
    if strip_set:
        before = len(train_idx) + len(val_idx) + len(test_idx) + len(inspected_idx)
        train_idx = [i for i in train_idx if i not in strip_set]
        val_idx = [i for i in val_idx if i not in strip_set]
        test_idx = [i for i in test_idx if i not in strip_set]
        inspected_idx = [i for i in inspected_idx if i not in strip_set]
        after = len(train_idx) + len(val_idx) + len(test_idx) + len(inspected_idx)
        if before != after:
            log.info(
                "Belt-and-braces strip: %d unlabeled/eval rows removed from labelled splits (%d -> %d)",
                before - after, before, after,
            )

    meta_clean = meta.drop(columns=["_label"])
    if "_eval" in meta_clean.columns:
        meta_clean = meta_clean.drop(columns=["_eval"])

    split_mode = "inspected_holdout" if inspected_idx else ("region" if bool(cfg.data.train_regions) else "random")
    log.info(
        "Splits (%s): train=%d  val=%d  test=%d  inspected=%d  eval=%d  generalization=%d  (pos ratio: %.2f)",
        split_mode,
        len(train_idx),
        len(val_idx),
        len(test_idx),
        len(inspected_idx),
        len(eval_idx),
        len(gen_idx),
        meta["_label"].mean(),
    )

    # Persist split assignments so we can colour the map later
    split_col = pd.Series("unassigned", index=meta_clean.index)
    # Mark unlabeled rows up front so labelled splits (which come next) win.
    unlabeled_mask = meta["_label"] == -1
    if unlabeled_mask.any():
        split_col.loc[unlabeled_mask] = "unlabeled"
    split_col.iloc[train_idx] = "train"
    split_col.iloc[val_idx] = "val"
    split_col.iloc[test_idx] = "test"
    if inspected_idx:
        split_col.iloc[inspected_idx] = "inspected"
    if eval_idx:
        # eval overrides any prior assignment -- absolute hold-out.
        split_col.iloc[eval_idx] = "eval"
    if gen_idx:
        # generalization overrides any prior assignment -- absolute OOD hold-out.
        split_col.iloc[gen_idx] = "generalization"
    splits_df = meta_clean[["candidate_id"]].copy()
    splits_df["split"] = split_col
    splits_dir = patches_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / f"{cfg._config_stem}.csv"
    splits_df.to_csv(splits_path, index=False)
    log.info("Saved split assignments to %s", splits_path)

    n_spectral = len(cfg.patches.bands)

    # Resolve channel subset and crop size for ablation experiments
    channel_indices = None
    channel_subset = getattr(cfg.training, "channel_subset", None)
    if channel_subset:
        from .config import resolve_channel_indices
        channel_indices, n_spectral = resolve_channel_indices(
            channel_subset, cfg.patches.bands, cfg.patches.indices,
        )
        log.info("Channel subset %s -> indices %s (%d spectral)", channel_subset, channel_indices, n_spectral)
    crop_size = getattr(cfg.training, "crop_center_px", None)
    if crop_size:
        log.info("Center crop: %dx%d pixels", crop_size, crop_size)

    ds_kwargs = dict(rng_seed=cfg.training.seed, n_spectral_bands=n_spectral,
                     channel_indices=channel_indices, crop_size=crop_size)

    aug_cfg = getattr(cfg.training, "augmentation", None)
    train_ds = PatchDataset(meta_clean.iloc[train_idx], candidates, patches_root, aug_config=aug_cfg, **ds_kwargs)
    val_ds = PatchDataset(meta_clean.iloc[val_idx], candidates, patches_root, augment=False, **ds_kwargs)
    test_ds = PatchDataset(meta_clean.iloc[test_idx], candidates, patches_root, augment=False, **ds_kwargs)

    inspected_ds = None
    if inspected_idx:
        inspected_ds = PatchDataset(meta_clean.iloc[inspected_idx], candidates, patches_root, augment=False, **ds_kwargs)

    eval_ds = None
    if eval_idx:
        eval_ds = PatchDataset(meta_clean.iloc[eval_idx], candidates, patches_root, augment=False, **ds_kwargs)

    gen_ds = None
    if gen_idx:
        gen_ds = PatchDataset(meta_clean.iloc[gen_idx], candidates, patches_root, augment=False, **ds_kwargs)

    # Compute per-sample weights for region and/or class upsampling.
    # When both are enabled, weights multiply: each sample's draw probability
    # is proportional to (1/n_country) * (1/n_class), giving roughly equal
    # representation across both axes per epoch.
    region_w = None
    class_w = None
    if cfg.training.upsample_minority_regions:
        region_w = _compute_region_weights(
            meta_clean.iloc[train_idx], candidates,
        )
    if getattr(cfg.training, "balanced_class_sampling", False):
        class_w = _compute_class_weights(
            meta_clean.iloc[train_idx], candidates,
        )
    if region_w is not None and class_w is not None:
        train_ds.sample_weights = region_w * class_w
    elif region_w is not None:
        train_ds.sample_weights = region_w
    elif class_w is not None:
        train_ds.sample_weights = class_w

    return train_ds, val_ds, test_ds, inspected_ds, eval_ds, gen_ds


def _compute_class_weights(
    meta: pd.DataFrame, candidates: pd.DataFrame,
) -> np.ndarray:
    """Compute per-sample weights so each class contributes equally per epoch.

    Inversely proportional to class frequency in the training subset.
    """
    cid_to_label = dict(zip(
        candidates["id"].astype(str), candidates["label"].astype(int),
    ))
    labels = meta["candidate_id"].astype(str).map(cid_to_label).fillna(-1).astype(int)

    counts = labels.value_counts()
    # Exclude unlabeled (-1) from the balanced sampler reckoning, but keep
    # such rows with weight 0 so they never get sampled (defensive).
    counts = counts[counts.index >= 0]
    n_classes = len(counts)
    total = int(counts.sum())
    weight_map = {c: total / (n_classes * n) for c, n in counts.items()}
    weights = labels.map(weight_map).fillna(0.0).values.astype(np.float64)

    log.info(
        "Class upsampling weights: %s",
        ", ".join(f"class{c}={weight_map[c]:.2f} (n={counts[c]})" for c in sorted(weight_map)),
    )
    return weights


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
