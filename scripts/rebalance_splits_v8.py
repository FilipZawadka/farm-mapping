"""Build all_clusters_v8.parquet: add farm POSITIVES from the inference countries.

Why
---
round_2 (v7) promoted up to 50 farm false-positives per country from qual_eval
into train/val -- 962 rows, **100% NotFarm**, across 106 countries where the
model previously had no training signal at all. The v7 result showed what that
teaches: at the default threshold farm false-positives collapsed (ALB+COD
66.2% -> 5.0%) but farm recall collapsed with them (92.5% -> 42.5%), and
threshold-free ROC-AUC did NOT improve (ALB+COD 0.777 -> 0.772; at a matched
false-positive rate the two models give identical recall). The model learned a
prior -- "things in unfamiliar countries are not farms" -- rather than a
discriminative feature.

This script supplies the missing half of the lesson: a country-stratified
sample of qual_eval rows that ARE farms, promoted to train/val on exactly the
convention Rachel used for the negatives (up to --per-country each, 80:20
train:val). Each inference country then teaches both "this is a farm here" and
"this is not", instead of negatives only.

Everything else is untouched: test / eval / generalization / predict keep their
membership, so the ALB+COD+BGD+NGA generalization slice stays a clean, directly
comparable benchmark across v7 and v8. qual_eval shrinks by the promoted rows;
compare models on the v8 qual_eval set, which neither model trained on.

Deterministic: selection is seeded, so re-running reproduces the same split.

Usage::

    python scripts/rebalance_splits_v8.py \
        --in  data/rachel_geometry_candidates/all_countries/all_clusters_v7.parquet \
        --out data/rachel_geometry_candidates/all_countries/all_clusters_v8.parquet
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 4-class mapping, mirrors training/rachel_to_candidates.py four_class mode.
MAP4 = {
    "NotFarm": 0,
    "Farm: Poultry: Meat Chickens": 1,
    "Farm: Poultry: Eggs": 1,
    "Farm: Poultry: Unspecified/Other": 1,
    "Farm: Pigs": 2,
    "Farm: Cattle": 3,
}
CLASS_NAMES = {0: "NotFarm", 1: "Poultry", 2: "Pigs", 3: "Cattle"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-country", type=int, default=50,
                     help="max farm positives promoted per country (Rachel used 50 for negatives)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="80:20 train:val, as Rachel used")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print(f"read {args.inp}: {len(df):,} rows")
    print("before:")
    print(df.cnn_split_assigned.value_counts(dropna=False).to_string())

    c4 = df.final_label.map(MAP4)
    # Eligible: in qual_eval, maps to a 4-class slot, and IS a farm.
    elig = (df.cnn_split_assigned == "qual_eval") & c4.notna() & (c4 >= 1)
    pool = df[elig]
    print(f"\npromotable farm positives in qual_eval: {len(pool):,} across {pool.ADM0.nunique()} countries")

    rng = np.random.default_rng(args.seed)
    picked: list[int] = []
    for iso, grp in pool.groupby("ADM0", sort=True):
        idx = grp.index.to_numpy()
        take = min(len(idx), args.per_country)
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    picked = np.array(sorted(picked))
    print(f"selected {len(picked):,} rows (<= {args.per_country}/country)")

    # 80:20 train:val, assigned per country so every country contributes to both
    to_val: list[int] = []
    for iso, grp in df.loc[picked].groupby("ADM0", sort=True):
        idx = grp.index.to_numpy()
        n_val = int(round(len(idx) * args.val_frac))
        if n_val:
            to_val.extend(rng.choice(idx, size=n_val, replace=False).tolist())
    to_val = set(to_val)
    to_train = [i for i in picked if i not in to_val]

    out = df.copy()
    out.loc[list(to_train), "cnn_split_assigned"] = "train"
    out.loc[sorted(to_val), "cnn_split_assigned"] = "val"
    print(f"  -> train +{len(to_train):,}, val +{len(to_val):,}")

    print("\nclass mix of the promoted rows:")
    print(c4.loc[picked].map(CLASS_NAMES).value_counts().to_string())

    print("\nafter:")
    print(out.cnn_split_assigned.value_counts(dropna=False).to_string())

    new_c4 = out.final_label.map(MAP4)
    tr = new_c4[out.cnn_split_assigned == "train"].dropna().astype(int).value_counts().sort_index()
    old_tr = c4[df.cnn_split_assigned == "train"].dropna().astype(int).value_counts().sort_index()
    comp = pd.DataFrame({"v7": old_tr, "v8": tr}).fillna(0).astype(int)
    comp.index = [CLASS_NAMES[i] for i in comp.index]
    comp["delta"] = comp.v8 - comp.v7
    comp["v8_%"] = (100 * comp.v8 / comp.v8.sum()).round(1)
    print("\ntrain class composition:")
    print(comp.to_string())

    n_tr_ctry = out[out.cnn_split_assigned == "train"].ADM0.nunique()
    both = (out[out.cnn_split_assigned == "train"]
            .assign(c=new_c4[out.cnn_split_assigned == "train"])
            .groupby("ADM0")["c"].agg(lambda s: (s >= 1).any() and (s == 0).any()))
    print(f"\ntrain countries: {n_tr_ctry} | with BOTH farm and not-farm examples: {int(both.sum())} "
          f"(v7 had only the 5 original)")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(o, index=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out.to_parquet(o.parent / f"{o.stem}_{stamp}{o.suffix}", index=False)
    print(f"\nwrote {o} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
