"""Build all_clusters_v5.parquet.

Same shape as v4 + carries through Rachel's new explicit split columns
(`cnn_split_assigned`, `if_split_assigned`) for the 5 training countries and
2 generalization-testing countries. These columns are now the single source
of truth for train/val/test/eval/generalization membership -- see
docs/EVAL_FRAMEWORK.md and training/dataset.py build_splits().

Source A: existing all_clusters_v4.parquet, filtered to ADM0 NOT in
    {USA, BRA, CHL, MEX, THA, BGD, NGA} -- everything outside Rachel's
    labeled-countries set (unchanged from v4).
Source B: each {Country}/{ISO}_selected_clusters_for_analysis.parquet for the
    5 training countries (USA, BRA, CHL, MEX, THA), refreshed from Drive.
Source C: each {Country}/{ISO}_selected_clusters_for_analysis.parquet for the
    generalization-testing countries (BGD, NGA), refreshed from Drive.

Usage::

    python scripts/merge_clusters_v5.py --v4 <path> --country-dir <dir> --out <path>

Where --country-dir contains the 7 refreshed {ISO}_selected_clusters_for_analysis.parquet
files directly (not nested under {CountryName}/ subfolders -- this script is meant to
run against a flat pull, e.g. from `rclone copyto drive:{Country}/{ISO}_selected_clusters_for_analysis.parquet`).
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import pandas as pd

TRAIN_COUNTRIES = ["USA", "BRA", "CHL", "MEX", "THA"]
GENERALIZATION_COUNTRIES = ["BGD", "NGA"]

UNION_COLS = [
    "ADM0", "cluster_id",
    "original_label", "standardized_label", "visual_label", "final_label",
    "label_source", "eval_set", "random_sample",
    "viz_status", "viz_label",
    "template_score_if", "dmv",
    "cnn_split_assigned", "if_split_assigned",
    "geometry",
]


def _normalize(df: pd.DataFrame, iso: str) -> pd.DataFrame:
    df = df.copy()
    if "ADM0" not in df.columns:
        df["ADM0"] = iso
    for c in UNION_COLS:
        if c not in df.columns:
            df[c] = None
    if "eval_set" in df.columns:
        df["eval_set"] = df["eval_set"].fillna(False).astype(bool)
    if "random_sample" in df.columns:
        df["random_sample"] = df["random_sample"].fillna(False).astype(bool)
    return df[UNION_COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", required=True, help="path to existing all_clusters_v4.parquet")
    ap.add_argument("--country-dir", required=True,
                     help="dir containing flat {ISO}_selected_clusters_for_analysis.parquet files")
    ap.add_argument("--out", required=True, help="output path for all_clusters_v5.parquet")
    args = ap.parse_args()

    v4_path = Path(args.v4)
    country_dir = Path(args.country_dir)
    out_path = Path(args.out)

    print(f"reading {v4_path} ...")
    v4 = pd.read_parquet(v4_path)
    print(f"  {len(v4):,} rows, {v4['ADM0'].nunique()} ADM0s")

    labeled_iso = set(TRAIN_COUNTRIES + GENERALIZATION_COUNTRIES)
    rest = v4[~v4["ADM0"].isin(labeled_iso)].copy()
    print(f"rest-of-world after dropping {sorted(labeled_iso)}: {len(rest):,} rows")
    rest = _normalize(rest, iso="")

    new_parts = []
    for iso in TRAIN_COUNTRIES + GENERALIZATION_COUNTRIES:
        path = country_dir / f"{iso}_selected_clusters_for_analysis.parquet"
        if not path.exists():
            print(f"  SKIP {iso}: {path} not found")
            continue
        df = pd.read_parquet(path)
        df["ADM0"] = iso
        df = _normalize(df, iso=iso)
        role = "train" if iso in TRAIN_COUNTRIES else "gen"
        eval_n = int(df["eval_set"].sum())
        labeled_n = int(df["final_label"].notna().sum())
        cnn_n = int(df["cnn_split_assigned"].notna().sum()) if "cnn_split_assigned" in df.columns else 0
        print(
            f"  + {iso} ({role}): {len(df):,} rows, "
            f"eval_set={eval_n}, labeled={labeled_n}, cnn_split_assigned={cnn_n}"
        )
        new_parts.append(df)

    merged = pd.concat([rest, *new_parts], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["cluster_id"], keep="last").reset_index(drop=True)
    print(f"deduped on cluster_id: {before:,} -> {len(merged):,}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dated = out_path.parent / f"{out_path.stem}_{stamp}{out_path.suffix}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    merged.to_parquet(dated, index=False)
    print(f"wrote {out_path} ({len(merged):,} rows)")
    print(f"wrote {dated.name} (dated sibling)")

    print()
    print("--- summary ---")
    print(f"  total rows:        {len(merged):,}")
    print(f"  distinct ADM0:     {merged['ADM0'].nunique()}")
    print(f"  labeled rows:      {int(merged['final_label'].notna().sum()):,}")
    print(f"  eval_set rows:     {int(merged['eval_set'].sum()):,}")
    print(f"  cnn_split_assigned rows: {int(merged['cnn_split_assigned'].notna().sum()):,}")
    print()
    print("cnn_split_assigned value counts (labeled countries only):")
    print(
        merged[merged["ADM0"].isin(labeled_iso)]["cnn_split_assigned"]
        .value_counts(dropna=False).to_string()
    )


if __name__ == "__main__":
    main()
