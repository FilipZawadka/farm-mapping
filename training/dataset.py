"""PyTorch Dataset for Sentinel-2 farm-detection patches.

Handles:
- Loading .npy patches + labels from patch_meta.parquet
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

from .config import PipelineConfig, matches_any_region

log = logging.getLogger(__name__)


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
        rng_seed: int = 42,
    ):
        self.meta = meta.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

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

        if self.augment:
            arr = self._augment(arr)

        tensor = torch.from_numpy(arr)
        label = int(self.labels[idx])
        return tensor, label

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Random flip + 90-degree rotation + brightness jitter (in-place safe)."""
        if self.rng.random() < 0.5:
            arr = np.flip(arr, axis=1).copy()
        if self.rng.random() < 0.5:
            arr = np.flip(arr, axis=2).copy()

        k = self.rng.integers(0, 4)
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()

        jitter = self.rng.uniform(0.9, 1.1)
        arr = arr * jitter

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


def _assign_by_region(
    keys: pd.Series, states: pd.Series,
    train_regions: list[str],
    val_regions: list[str],
    test_regions: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically assign rows to splits by region membership."""
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i in range(len(keys)):
        k, s = str(keys.iloc[i]), str(states.iloc[i])
        if matches_any_region(k, s, test_regions):
            test_idx.append(i)
        elif matches_any_region(k, s, val_regions):
            val_idx.append(i)
        elif matches_any_region(k, s, train_regions):
            train_idx.append(i)
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

def build_splits(
    cfg: PipelineConfig,
    patches_dir: Optional[Path] = None,
) -> tuple[PatchDataset, PatchDataset, PatchDataset]:
    """Load metadata + candidates and create train/val/test datasets.

    When ``train_regions`` is configured, rows are assigned to splits by
    geographic region. Otherwise the legacy random stratified split is used.

    If *patches_dir* is given, use it as the patch root; else use
    ``cfg.patches.output_dir``.

    Returns ``(train_ds, val_ds, test_ds)``.
    """
    patches_dir = Path(patches_dir) if patches_dir is not None else Path(cfg.patches.output_dir)
    meta_path = patches_dir / "patch_meta.parquet"
    candidates_path = patches_dir / "candidates.parquet"

    meta = pd.read_parquet(meta_path)
    candidates = pd.read_parquet(candidates_path)

    label_map = dict(zip(
        candidates["id"].astype(str),
        candidates["label"].astype(int),
    ))
    meta["_label"] = meta["candidate_id"].astype(str).map(label_map).fillna(0).astype(int)

    rng = np.random.default_rng(cfg.training.seed)

    use_regions = bool(cfg.data.train_regions)
    if use_regions:
        train_idx, val_idx, test_idx = _region_split_indices(
            meta, candidates, cfg, rng,
        )
    else:
        train_idx, val_idx, test_idx = _random_split_indices(
            meta, cfg, rng,
        )

    meta_clean = meta.drop(columns=["_label"])

    log.info(
        "Splits (%s): train=%d  val=%d  test=%d  (pos ratio: %.2f)",
        "region" if use_regions else "random",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        meta["_label"].mean(),
    )

    return (
        PatchDataset(meta_clean.iloc[train_idx], candidates, patches_dir, augment=True, rng_seed=cfg.training.seed),
        PatchDataset(meta_clean.iloc[val_idx], candidates, patches_dir, augment=False, rng_seed=cfg.training.seed),
        PatchDataset(meta_clean.iloc[test_idx], candidates, patches_dir, augment=False, rng_seed=cfg.training.seed),
    )
