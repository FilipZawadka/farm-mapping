"""Merge per-shard scored_candidates.parquet files into one scored set.

Each shard of a sharded scoring run (see scripts/make_inference_shards.py)
writes its own ``data/output/{stem}/scored_candidates.parquet``. This stitches
them back together and verifies the merge actually covers the work:

  * no candidate_id appears in two shards (shards partition by country, so an
    overlap means the configs were regenerated inconsistently),
  * no duplicate rows within a shard,
  * every shard contributed (an empty shard is a silent hole -- inference skips
    absent country CSVs, so a mis-specified shard "succeeds" with 0 rows),
  * optionally, coverage against the candidate universe.

Writes a GeoParquet (geometry rebuilt from lng/lat with an explicit CRS) --
plain pandas ``to_parquet`` silently drops the geo metadata that
``web/scripts/export_dataset.py`` needs to read it back.

Usage::

    python scripts/merge_inference_shards.py \
        --output-root data/output --stems world_v10_fourclass_scoreall_shard0 ... \
        --out data/output/world_v10_fourclass_scoreall/scored_candidates.parquet \
        --expect-candidates data/rachel_geometry_candidates/candidates_world_v10_scoreall
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("data/output"))
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-candidates", type=Path,
                     help="candidates dir; reports scored-vs-candidate coverage")
    args = ap.parse_args()

    frames = []
    for stem in args.stems:
        path = args.output_root / stem / "scored_candidates.parquet"
        if not path.exists():
            sys.exit(f"MISSING shard output: {path}")
        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            gdf = pd.read_parquet(path)
        n_dup = int(gdf["candidate_id"].duplicated().sum())
        print(f"  {stem:48s} {len(gdf):>8,} rows"
              + (f"  ({n_dup} intra-shard dupes)" if n_dup else ""))
        if len(gdf) == 0:
            sys.exit(f"EMPTY shard {stem} -- it scored nothing; check its data.countries "
                     "against the candidate CSVs before trusting this merge.")
        gdf["_shard"] = stem
        frames.append(gdf)

    merged = pd.concat(frames, ignore_index=True)

    overlap = merged["candidate_id"].duplicated().sum()
    if overlap:
        dupes = merged[merged["candidate_id"].duplicated(keep=False)]
        by = dupes.groupby("candidate_id")["_shard"].nunique()
        cross = int((by > 1).sum())
        print(f"\nWARNING: {overlap:,} duplicate candidate_id rows "
              f"({cross:,} appear in >1 shard) -- keeping first of each.")
        merged = merged.drop_duplicates(subset="candidate_id", keep="first")

    merged = merged.drop(columns=["_shard"])

    # Rebuild geometry so the output is a valid GeoParquet.
    if "geometry" in merged.columns:
        merged = merged.drop(columns=["geometry"])
    out_gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["lng"], merged["lat"]),
        crs="EPSG:4326",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_parquet(args.out)
    print(f"\nwrote {args.out} ({len(out_gdf):,} unique candidates, {len(out_gdf.columns)} cols)")

    if args.expect_candidates:
        cand_ids: set[str] = set()
        for csv in sorted(args.expect_candidates.glob("*.csv")):
            cand_ids |= set(pd.read_csv(csv, usecols=["id"])["id"].astype(str))
        scored = set(out_gdf["candidate_id"].astype(str))
        missing = cand_ids - scored
        print(f"coverage: {len(scored):,}/{len(cand_ids):,} candidates scored "
              f"({100*len(scored)/max(len(cand_ids),1):.2f}%); {len(missing):,} unscored "
              "(expected: candidates with no usable patch)")

    if "predicted_label" in out_gdf.columns:
        print("\npredicted_label distribution:")
        print(out_gdf["predicted_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
