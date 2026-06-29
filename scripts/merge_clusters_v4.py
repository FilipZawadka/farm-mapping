"""Build all_clusters_v4.parquet.

Same shape as v3 + adds the generalization-testing countries
(BGD, NGA) using their _selected_clusters_for_analysis.parquet files.

Source A: existing all_clusters_v2.parquet, filtered to ADM0 NOT in
    {USA, BRA, CHL, MEX, THA, BGD, NGA} -- everything outside Rachel's
    Labeled-countries set.
Source B: each {Country}/{ISO}_selected_clusters_for_analysis.parquet for the
    5 training countries (USA, BRA, CHL, MEX, THA).
Source C: each {Country}/{ISO}_selected_clusters_for_analysis.parquet for the
    generalization-testing countries (BGD, NGA).

The result has Rachel's new columns preserved end-to-end
(`visual_label`, `label_source`, `final_label`, `eval_set`, `random_sample`).

Usage::

    python scripts/merge_clusters_v4.py
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path("/workspace/farm-mapping/data/rachel_geometry_candidates/all_countries")
V2 = ROOT / "all_clusters_v2.parquet"
OUT = ROOT / "all_clusters_v4.parquet"

TRAIN_COUNTRIES = [
    ("USA", "USA"),
    ("BRA", "Brazil"),
    ("CHL", "Chile"),
    ("MEX", "Mexico"),
    ("THA", "Thailand"),
]
GENERALIZATION_COUNTRIES = [
    ("BGD", "Bangladesh"),
    ("NGA", "Nigeria"),
]

UNION_COLS = [
    "ADM0", "cluster_id",
    "original_label", "standardized_label", "visual_label", "final_label",
    "label_source", "eval_set", "random_sample",
    "viz_status", "viz_label",
    "template_score_if", "dmv",
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
    print(f"reading {V2} ...")
    v2 = pd.read_parquet(V2)
    print(f"  {len(v2):,} rows, {v2['ADM0'].nunique()} ADM0s")

    labeled_iso = {iso for iso, _ in TRAIN_COUNTRIES + GENERALIZATION_COUNTRIES}
    rest = v2[~v2["ADM0"].isin(labeled_iso)].copy()
    print(f"rest-of-world after dropping {sorted(labeled_iso)}: {len(rest):,} rows")
    rest = _normalize(rest, iso="")

    new_parts = []
    for iso, country_dir in TRAIN_COUNTRIES + GENERALIZATION_COUNTRIES:
        path = ROOT / country_dir / f"{iso}_selected_clusters_for_analysis.parquet"
        if not path.exists():
            print(f"  SKIP {iso} ({country_dir}): {path} not found")
            continue
        df = pd.read_parquet(path)
        df["ADM0"] = iso
        df = _normalize(df, iso=iso)
        role = "train" if (iso, country_dir) in TRAIN_COUNTRIES else "gen"
        eval_n = int(df["eval_set"].sum())
        labeled_n = int(df["final_label"].notna().sum())
        print(
            f"  + {iso} ({country_dir}, {role}): {len(df):,} rows, "
            f"eval_set={eval_n}, labeled={labeled_n}"
        )
        new_parts.append(df)

    merged = pd.concat([rest, *new_parts], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["cluster_id"], keep="last").reset_index(drop=True)
    print(f"deduped on cluster_id: {before:,} -> {len(merged):,}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dated = OUT.parent / f"all_clusters_v4_{stamp}.parquet"

    merged.to_parquet(OUT, index=False)
    merged.to_parquet(dated, index=False)
    print(f"wrote {OUT} ({len(merged):,} rows)")
    print(f"wrote {dated.name} (dated sibling)")

    print()
    print("--- summary ---")
    print(f"  total rows:        {len(merged):,}")
    print(f"  distinct ADM0:     {merged['ADM0'].nunique()}")
    print(f"  labeled rows:      {int(merged['final_label'].notna().sum()):,}")
    print(f"  eval_set rows:     {int(merged['eval_set'].sum()):,}")
    print(f"  random_sample:     {int(merged['random_sample'].sum()):,}")
    print()
    print("eval_set per ADM0:")
    print(
        merged.groupby("ADM0")["eval_set"].sum().sort_values(ascending=False).head(10).to_string()
    )
    print()
    print("labelled rows per ADM0 (top 12):")
    print(
        merged[merged["final_label"].notna()]
        .groupby("ADM0").size().sort_values(ascending=False).head(12).to_string()
    )


if __name__ == "__main__":
    main()
