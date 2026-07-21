"""Build all_clusters_v6.parquet from ALL of Rachel's per-country
``{ISO}_selected_clusters_for_analysis.parquet`` files.

Why this exists
---------------
``merge_clusters_v5.py`` reads for_analysis files for only 7 countries
(USA/BRA/CHL/MEX/THA + BGD/NGA) and takes every other country from the older
``all_clusters_v4.parquet`` chain. Rachel now publishes for_analysis files for
**167** countries, so v5 silently omitted:

  * 58 countries entirely (~21k clusters), and
  * ~15.4k labels in the 102 countries it did include but sourced from v4 --
    e.g. Argentina has 1,366 labels on Drive and 0 in v5, despite both holding
    the same 3,289 clusters.

Nearly all of those labels are ``cnn_split_assigned == qual_eval`` -- Rachel's
qualitative-evaluation pool, deliberately separate from train/val/test. The
train/val/test/eval/generalization assignments exist ONLY in the 7 countries
and v5 already had all of them, so this rebuild adds evaluation and scoring
coverage, NOT training data.

cluster_id is NOT stable across sources
---------------------------------------
The v4 chain and the current for_analysis files number clusters differently:
for Argentina only 1,245 of 3,289 ids are shared, and 99.9% of those point to a
different place (median 278 km apart). Joining these two sources by id is
exactly the class of bug that corrupted the patch store (see
docs/EXPERIMENTS_LOG.md 2026-07-20). So:

  * the for_analysis files are the sole authority for cluster_id + geometry, and
  * the few v5-only columns are carried over by CENTROID PROXIMITY, never by id.

Usage::

    python scripts/merge_clusters_v6.py \
        --for-analysis-dir <dir containing {Country}/{ISO}_..._for_analysis.parquet> \
        --v5 data/rachel_geometry_candidates/all_countries/all_clusters_v5.parquet \
        --out data/rachel_geometry_candidates/all_countries/all_clusters_v6.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

# Carried over from v5 by geometry match; absent from the for_analysis schema.
# `dmv` is deliberately NOT here -- it is 100% null in v5, pure dead weight.
V5_CARRY_COLS = ["viz_status", "viz_label", "template_score_if"]

MATCH_TOLERANCE_M = 250.0


def _centroids_m(geom_wkb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lng) centroid arrays from a WKB column."""
    g = shapely.centroid(shapely.from_wkb(geom_wkb))
    return shapely.get_y(g), shapely.get_x(g)


def _to_local_xy(lat: np.ndarray, lng: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular projection to metres -- fine for a 250 m proximity test."""
    return np.c_[lat * 111_320.0, lng * 111_320.0 * np.cos(np.radians(lat0))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--for-analysis-dir", required=True,
                     help="dir holding the for_analysis parquets (searched recursively)")
    ap.add_argument("--v5", required=True, help="all_clusters_v5.parquet (source of the carry-over columns)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.for_analysis_dir, "**", "*_selected_clusters_for_analysis.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no for_analysis parquets under {args.for_analysis_dir}")
    print(f"reading {len(files)} for_analysis files ...")

    parts = []
    for f in files:
        iso = re.search(r"([A-Z]{3})_selected_clusters_for_analysis", os.path.basename(f)).group(1)
        d = pd.read_parquet(f)
        d["ADM0"] = iso
        parts.append(d)
    base = pd.concat(parts, ignore_index=True)
    print(f"  {len(base):,} clusters, {base.ADM0.nunique()} countries, "
          f"{int(base['final_label'].notna().sum()):,} labels")

    dup = int(base["cluster_id"].duplicated().sum())
    if dup:
        print(f"  WARNING: {dup:,} duplicate cluster_ids across files -- keeping last")
        base = base.drop_duplicates(subset="cluster_id", keep="last").reset_index(drop=True)

    # --- carry the v5-only columns over by centroid proximity (never by id) ---
    v5 = pd.read_parquet(args.v5)
    have = [c for c in V5_CARRY_COLS if c in v5.columns]
    print(f"\ncarrying {have} from v5 by centroid proximity (<= {MATCH_TOLERANCE_M:.0f} m) ...")

    b_lat, b_lng = _centroids_m(base["geometry"].values)
    v_lat, v_lng = _centroids_m(v5["geometry"].values)
    lat0 = float(np.nanmean(b_lat))
    tree = cKDTree(_to_local_xy(v_lat, v_lng, lat0))
    dist, idx = tree.query(_to_local_xy(b_lat, b_lng, lat0), k=1)
    ok = dist <= MATCH_TOLERANCE_M
    print(f"  {ok.sum():,}/{len(base):,} ({100*ok.mean():.1f}%) matched an existing v5 cluster")

    for c in have:
        src = v5[c].to_numpy()
        vals = np.where(ok, src[idx], None)
        base[c] = pd.Series(vals, index=base.index).where(pd.Series(ok, index=base.index))

    # normalise the flag columns the downstream loader expects
    for c in ("eval_set", "random_sample"):
        if c in base.columns:
            base[c] = base[c].fillna(False).astype(bool)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out, index=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dated = out.parent / f"{out.stem}_{stamp}{out.suffix}"
    base.to_parquet(dated, index=False)
    print(f"\nwrote {out} ({len(base):,} rows, {len(base.columns)} cols)")
    print(f"wrote {dated.name}")

    print("\n--- summary ---")
    print(f"  clusters        : {len(base):,}")
    print(f"  countries       : {base.ADM0.nunique()}")
    print(f"  labels          : {int(base['final_label'].notna().sum()):,}")
    print("\ncnn_split_assigned:")
    print(base["cnn_split_assigned"].fillna("<null>").value_counts().to_string())


if __name__ == "__main__":
    main()
