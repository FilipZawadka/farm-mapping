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

    lat0 = float(np.nanmean(lat))
    def xy(la, lo):
        return np.c_[la * 111_320.0, lo * 111_320.0 * np.cos(np.radians(lat0))]

    # An id needs a mapping when it is ABSENT from the store, and equally when
    # it is PRESENT but its stored patch sits somewhere else -- a renumbering
    # reassigns existing ids to new locations, so "the id is known" does not
    # mean "the patch is right". Missing that second case is what left ~15k
    # patches to be re-fetched from Earth Engine on the v5->v6 rebuild: the
    # coordinate guard correctly refused them, but a usable patch for the same
    # place was already on disk under a different id.
    stored = cur.set_index(cur["candidate_id"].astype(str))[["lat", "lng"]]
    cid = df["cluster_id"].astype(str)
    s_lat = cid.map(stored["lat"]).to_numpy(dtype=float)
    s_lng = cid.map(stored["lng"]).to_numpy(dtype=float)
    known = ~np.isnan(s_lat)
    off = np.full(len(df), np.inf)
    if known.any():
        d = xy(lat[known], lng[known]) - xy(s_lat[known], s_lng[known])
        off[known] = np.hypot(d[:, 0], d[:, 1])
    absent = ~known
    relocated = known & (off > args.tolerance_m)
    need = pd.Series(absent | relocated, index=df.index)
    print(f"  already reachable by id: {int((known & ~relocated).sum()):,}")
    print(f"  need a mapping         : {int(need.sum()):,} "
          f"({int(absent.sum()):,} id absent, {int(relocated.sum()):,} id present but relocated)")
    if not need.any():
        print("nothing to do")
        return

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
