"""Re-point existing patches at renumbered cluster_ids, by location.

A parquet rebuild can renumber cluster_ids while the underlying clusters stay
put (v5 -> v6 does exactly this for ~160 countries: same sites, new ids). The
patch store is keyed by candidate_id, so after such a rebuild every existing
patch looks "missing" to the new ids -- even though the imagery on disk is
correct and already paid for.

This adds patch_meta rows mapping each NEW candidate_id to the .npy of the
existing patch at the same location. No Earth Engine calls, no new files: the
same array is simply reachable under the new id.

Safety: a row is only written when an existing patch sits within
--tolerance-m of the new cluster's centroid, and the row records the NEW
cluster's coordinates. So the coordinate guard in
``training/config.py validate_patch_locations`` re-validates every re-keyed
row at train/score time exactly as it would a freshly extracted one -- this
script cannot smuggle in a mislocated patch. That guard exists because
id-keyed reuse across a renumbering is what corrupted the store on 2026-06-23
(docs/EXPERIMENTS_LOG.md 2026-07-20).

Usage::

    python scripts/rekey_patches_by_location.py \
        --parquet .../all_clusters_v6.parquet \
        --patches-root data/patches --imagery-hash cc5a6ebb502a [--apply]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="rebuilt clusters parquet (new ids)")
    ap.add_argument("--patches-root", required=True)
    ap.add_argument("--imagery-hash", required=True)
    ap.add_argument("--tolerance-m", type=float, default=250.0)
    ap.add_argument("--apply", action="store_true",
                     help="write the rows; without this it is a dry run")
    args = ap.parse_args()

    patches_root = Path(args.patches_root)
    meta_path = patches_root / "patch_meta.csv"
    meta = pd.read_csv(meta_path, low_memory=False)
    cur = meta[meta["imagery_config_hash"].astype(str) == args.imagery_hash].copy()
    cur = cur.drop_duplicates(subset="candidate_id", keep="last")
    cur = cur.dropna(subset=["lat", "lng"])
    print(f"patch store: {len(cur):,} usable patches under hash {args.imagery_hash}")

    df = pd.read_parquet(args.parquet, columns=["cluster_id", "geometry"])
    g = shapely.centroid(shapely.from_wkb(df["geometry"].values))
    lat, lng = shapely.get_y(g), shapely.get_x(g)
    print(f"target parquet: {len(df):,} clusters")

    already = set(cur["candidate_id"].astype(str))
    need = ~df["cluster_id"].astype(str).isin(already)
    print(f"  already reachable by id: {(~need).sum():,}")
    print(f"  need a mapping         : {need.sum():,}")
    if not need.any():
        print("nothing to do")
        return

    lat0 = float(np.nanmean(lat))
    def xy(la, lo):
        return np.c_[la * 111_320.0, lo * 111_320.0 * np.cos(np.radians(lat0))]

    tree = cKDTree(xy(cur["lat"].to_numpy(), cur["lng"].to_numpy()))
    tgt_lat, tgt_lng = lat[need.to_numpy()], lng[need.to_numpy()]
    dist, idx = tree.query(xy(tgt_lat, tgt_lng), k=1)
    hit = dist <= args.tolerance_m
    print(f"  of those, {hit.sum():,} ({100*hit.mean():.1f}%) have an existing patch "
          f"within {args.tolerance_m:.0f} m -> re-key")
    print(f"  {(~hit).sum():,} genuinely need fresh extraction")
    if not hit.any():
        return

    src = cur.iloc[idx[hit]].reset_index(drop=True)
    new_rows = src.copy()
    new_rows["candidate_id"] = df.loc[need.to_numpy(), "cluster_id"].to_numpy()[hit]
    # Record the NEW cluster's coordinates so the coordinate guard validates
    # this row against the candidate it now serves.
    new_rows["lat"] = tgt_lat[hit]
    new_rows["lng"] = tgt_lng[hit]

    print(f"\nmedian offset of re-keyed pairs: {np.median(dist[hit]):.1f} m "
          f"(max {dist[hit].max():.1f} m)")

    if not args.apply:
        print("\nDRY RUN -- re-run with --apply to append these rows")
        return

    backup = meta_path.with_suffix(f".csv.bak_rekey")
    if not backup.exists():
        meta.to_csv(backup, index=False)
        print(f"backed up original patch_meta -> {backup.name}")
    combined = pd.concat([meta, new_rows[meta.columns]], ignore_index=True)
    tmp = meta_path.with_suffix(".csv.tmp_rekey")
    combined.to_csv(tmp, index=False)
    tmp.replace(meta_path)
    print(f"appended {len(new_rows):,} rows -> patch_meta.csv now {len(combined):,} rows")


if __name__ == "__main__":
    main()
