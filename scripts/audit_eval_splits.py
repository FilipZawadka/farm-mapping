"""Audit Eval-2 (representative-sample) row counts across the pipeline.

Rachel flagged in `docs/Model eval mtg.odp` (slide 8) that the eval-set
row counts look wrong. Run this against a finished training+inference run
to verify nothing has been dropped or doubled at any stage.

Expected invariant for each country:

    source parquet `eval_set == True`
      == candidate CSV  `eval_set == 1`
      == splits CSV  `split == "eval"`
      == scored parquet  `split == "eval"`

Usage::

    python scripts/audit_eval_splits.py \\
        --master data/rachel_geometry_candidates/all_countries/all_clusters_v4.parquet \\
        --candidates-dir data/rachel_geometry_candidates/candidates_world_v4_three_class \\
        --splits-csv data/patches/splits/world_v4_three_class.csv \\
        --scored data/output/world_v4_three_class/scored_candidates.parquet
"""

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="Path to all_clusters_*.parquet")
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--splits-csv", required=True)
    ap.add_argument("--scored", required=True, help="scored_candidates.parquet")
    args = ap.parse_args()

    print("=== source parquet ===")
    src = pd.read_parquet(args.master)
    if "eval_set" not in src.columns:
        print(f"  {args.master} has no eval_set column"); return
    src["eval_set"] = src["eval_set"].fillna(False).astype(bool)
    src_evt = src[src["eval_set"]].copy()
    print(f"  total rows={len(src):,}  eval_set=True={len(src_evt):,}")
    src_counts = src_evt.groupby("ADM0").size().rename("source")

    print()
    print("=== candidate CSVs ===")
    csv_dir = Path(args.candidates_dir)
    csv_frames = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        csv_frames.append(pd.read_csv(csv_path))
    cand = pd.concat(csv_frames, ignore_index=True) if csv_frames else pd.DataFrame()
    print(f"  total rows={len(cand):,}")
    if "eval_set" not in cand.columns:
        print("  WARN: candidate CSVs have no eval_set column"); cand_counts = pd.Series(dtype=int)
    else:
        cand["eval_set"] = cand["eval_set"].fillna(0).astype(int)
        cand_evt = cand[cand["eval_set"] == 1].copy()
        # Extract ADM0 from candidate id prefix.
        cand_evt["ADM0"] = cand_evt["id"].astype(str).str.split("_", n=1, expand=True)[0]
        print(f"  eval_set=1 rows={len(cand_evt):,}")
        cand_counts = cand_evt.groupby("ADM0").size().rename("candidate_csv")

    print()
    print("=== splits CSV ===")
    sp = pd.read_csv(args.splits_csv)
    print(f"  total={len(sp):,}  by split:")
    for k, v in sp["split"].value_counts().to_dict().items():
        print(f"    {k:<16} {v:>7,}")
    sp_evt = sp[sp["split"] == "eval"].copy()
    sp_evt["ADM0"] = sp_evt["candidate_id"].astype(str).str.split("_", n=1, expand=True)[0]
    sp_counts = sp_evt.groupby("ADM0").size().rename("splits_csv")

    print()
    print("=== scored parquet ===")
    try:
        import geopandas as gpd
        sc = gpd.read_parquet(args.scored)
    except Exception:
        sc = pd.read_parquet(args.scored)
    if "split" in sc.columns:
        sc_evt = sc[sc["split"] == "eval"].copy()
        sc_evt["ADM0"] = sc_evt["candidate_id"].astype(str).str.split("_", n=1, expand=True)[0]
        sc_counts = sc_evt.groupby("ADM0").size().rename("scored")
        print(f"  total={len(sc):,}  eval rows={len(sc_evt):,}")
    else:
        print("  WARN: scored has no split column"); sc_counts = pd.Series(dtype=int)

    print()
    print("=== per-country comparison ===")
    out = pd.concat([src_counts, cand_counts, sp_counts, sc_counts], axis=1).fillna(0).astype(int)
    out["match"] = (out.nunique(axis=1) == 1)
    print(out.to_string())
    bad = out[~out["match"]]
    if len(bad):
        print()
        print(f"!!! {len(bad)} countries have mismatched eval counts across stages")
    else:
        print()
        print("OK: eval-set row counts match across all stages.")


if __name__ == "__main__":
    main()
