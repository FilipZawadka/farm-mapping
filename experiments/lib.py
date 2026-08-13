"""Shared harness for post-hoc experiments in paper/experiments_justification_plan.md.

Operates entirely on archived scored parquets (no GPU, no patch store, no training).
Every experiment script imports from here so metric definitions stay identical
across experiments.

Conventions
-----------
* 4-class label order: 0=NotFarm, 1=Poultry, 2=Pigs, 3=Cattle. true_label == -1
  means unlabeled/unscorable and is always excluded from metrics.
* Binary "farm" task: y = (true_label != 0), p_farm = 1 - prob_class0.
* All confidence intervals are percentile bootstrap over rows (Efron 1987).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "notebooks" / "results_cache" / "data" / "output"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

CLASS_NAMES = ["NotFarm", "Poultry", "Pigs", "Cattle"]
THREE_CLASS_NAMES = ["NotFarm", "Poultry", "OtherFarm"]

# Archived four-class runs that carry full-world scores with probabilities.
FOURCLASS = {
    "v6": "world_v10_fourclass_v6_scored",
    "v7": "world_v10_fourclass_v7_scored",
    "v8": "world_v10_fourclass_v8_scored",
    "v9": "world_v10_fourclass_v9_scored",
}
# Runs scored on the labeled subset only (labeled_only: true).
FOURCLASS_LABELED = {
    "v9_full": "world_v10_fourclass_v9",
    "v9_bal": "world_v10_fourclass_v9_bal",
    "v10": "world_v10_fourclass_v10",
}
THREECLASS = {
    "softcon": "world_v9_softcon",
    "ctx128": "world_v9_ctx128",
}

BOOT_N = 2000
SEED = 42


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(run_dir: str) -> pd.DataFrame:
    """Load a scored_candidates.parquet by output-directory name."""
    path = CACHE / run_dir / "scored_candidates.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    return df


def prob_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("prob_class")],
        key=lambda c: int(c.replace("prob_class", "")),
    )


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable supervised label."""
    return df[df["true_label"].notna() & (df["true_label"] >= 0)].copy()


def slice_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select a named split, preferring the explicit upstream column."""
    col = "cnn_split_assigned" if "cnn_split_assigned" in df.columns else "split"
    return labeled(df[df[col] == split])


def farm_binary(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, p_farm) for the binary farm-vs-not task."""
    y = (df["true_label"].to_numpy().astype(int) != 0).astype(int)
    p = 1.0 - df["prob_class0"].to_numpy(dtype=float)
    return y, p


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (equal-width bins), Guo et al. 2017."""
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def binary_suite(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Standard binary metric bundle at a given operating point."""
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "roc_auc": safe_auc(y, p),
        "pr_auc": safe_ap(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "balanced_acc": (recall + spec) / 2 if not np.isnan(recall) else float("nan"),
        "f1_farm": (2 * precision * recall / (precision + recall))
        if precision and recall and not np.isnan(precision) and not np.isnan(recall)
        else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def macro_f1_multiclass(df: pd.DataFrame, n_classes: int = 4) -> float:
    y = df["true_label"].to_numpy().astype(int)
    cols = prob_columns(df)[:n_classes]
    pred = df[cols].to_numpy(dtype=float).argmax(axis=1)
    return float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0))


def per_class_f1(df: pd.DataFrame, n_classes: int = 4) -> dict:
    y = df["true_label"].to_numpy().astype(int)
    cols = prob_columns(df)[:n_classes]
    probs = df[cols].to_numpy(dtype=float)
    pred = probs.argmax(axis=1)
    names = CLASS_NAMES if n_classes == 4 else THREE_CLASS_NAMES
    out = {}
    for c in range(n_classes):
        support = int((y == c).sum())
        f1 = float(f1_score((y == c).astype(int), (pred == c).astype(int), zero_division=0))
        auc = safe_auc((y == c).astype(int), probs[:, c]) if support else float("nan")
        out[names[c]] = {"support": support, "f1": f1, "ovr_auc": auc}
    return out


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(fn, *arrays, n: int = BOOT_N, alpha: float = 0.05, seed: int = SEED):
    """Percentile bootstrap CI for a statistic over paired row arrays."""
    arrays = [np.asarray(a) for a in arrays]
    size = len(arrays[0])
    if size == 0:
        return float("nan"), (float("nan"), float("nan"))
    point = fn(*arrays)
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, size, size)
        stats[i] = fn(*[a[idx] for a in arrays])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return point, (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), (float(lo), float(hi))


def paired_bootstrap_delta(fn, y, p_a, p_b, n: int = BOOT_N, alpha: float = 0.05, seed: int = SEED):
    """CI on the paired difference stat(B) - stat(A) using shared row resamples.

    Paired resampling is what makes a small delta testable: both models see the
    identical bootstrap rows, so shared row-difficulty variance cancels.
    """
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    size = len(y)
    if size == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    point = fn(y, p_b) - fn(y, p_a)
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, size, size)
        stats[i] = fn(y[idx], p_b[idx]) - fn(y[idx], p_a[idx])
    stats = stats[~np.isnan(stats)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value for H0: delta == 0.
    p_value = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return float(point), (float(lo), float(hi)), float(min(p_value, 1.0))


# --------------------------------------------------------------------------
# spatial
# --------------------------------------------------------------------------
EARTH_R_M = 6_371_000.0


def nearest_distance_m(query_latlng: np.ndarray, ref_latlng: np.ndarray) -> np.ndarray:
    """Great-circle distance (m) from each query point to the nearest ref point."""
    from sklearn.neighbors import BallTree

    if len(ref_latlng) == 0 or len(query_latlng) == 0:
        return np.full(len(query_latlng), np.inf)
    tree = BallTree(np.radians(ref_latlng), metric="haversine")
    dist, _ = tree.query(np.radians(query_latlng), k=1)
    return dist[:, 0] * EARTH_R_M


def latlng(df: pd.DataFrame) -> np.ndarray:
    return df[["lat", "lng"]].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def save(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{nd}f}"


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
