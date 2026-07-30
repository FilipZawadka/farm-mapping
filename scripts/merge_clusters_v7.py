"""Build all_clusters_v7.parquet from Rachel's per-country
``{ISO}_selected_clusters_round_2.parquet`` files.

What round_2 changes vs the for_analysis files behind v6
-------------------------------------------------------
Same 157,102 clusters, same 167 countries, and -- verified below -- **the same
cluster_ids on the same geometry**. What moved is labels and split membership:

  * training countries: a handful of bad labels corrected (28 relabels).
  * inference countries: up to 50 farm false-positives per country relabelled
    (all NotFarm -- largely solar arrays and greenhouse / plastic crop covers)
    and promoted qual_eval -> train(770)/val(192). Poultry/Pigs/Cattle train
    counts are untouched, so this adds hard negatives without diluting the
    farm-type signal.
  * two new fully-labelled generalization countries: ALB (140 clusters, a
    plastic-crop-cover test) and COD (47, industrial poultry in a low-income
    African country).

Why this can join by cluster_id when v6 could not
------------------------------------------------
v6 had to carry columns over by centroid proximity because the v4 chain and the
for_analysis files numbered clusters differently (Argentina: 1,245/3,289 ids
shared, 99.9% of those hundreds of km apart) -- joining by id there is the bug
class that corrupted the patch store (docs/EXPERIMENTS_LOG.md 2026-07-20).
round_2 is a re-export of the *same* generation, so ids are stable; this script
ASSERTS that (id overlap + zero geometry drift) and refuses to run otherwise,
rather than trusting it. Because the ids are stable, the existing patch store
also needs no re-keying or re-extraction.

Usage::

    python scripts/merge_clusters_v7.py \
        --round2-dir <dir with {Country}/{ISO}_selected_clusters_round_2.parquet> \
        --prev data/rachel_geometry_candidates/all_countries/all_clusters_v6.parquet \
        --out  data/rachel_geometry_candidates/all_countries/all_clusters_v7.parquet
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

# Present in our merged parquets but not in Rachel's per-country schema.
# `dmv` is deliberately excluded -- 100% null upstream, pure dead weight.
CARRY_COLS = ["viz_status", "viz_label", "template_score_if"]

# Max centroid drift tolerated on a shared cluster_id before we refuse the
# id-join. Anything above this means the ids were renumbered and a
# proximity-based merge (see merge_clusters_v6.py) is required instead.
MAX_DRIFT_DEG = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2-dir", required=True,
                     help="dir holding the round_2 parquets (searched recursively)")
    ap.add_argument("--prev", required=True,
                     help="all_clusters_v6.parquet -- source of the carry-over columns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.round2_dir, "**", "*_selected_clusters_round_2.parquet"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no round_2 parquets under {args.round2_dir}")
    print(f"reading {len(files)} round_2 files ...")

    parts = []
    for f in files:
        iso = re.search(r"([A-Z]{3})_selected_clusters_round_2", os.path.basename(f)).group(1)
        d = pd.read_parquet(f)
        # ADM0 is not in Rachel's per-country schema -- derived from the filename,
        # exactly as v3..v6 did. (Verified elsewhere to always equal the
        # cluster_id prefix.)
        d["ADM0"] = iso
        parts.append(d)
    base = pd.concat(parts, ignore_index=True)
    print(f"  {len(base):,} clusters, {base.ADM0.nunique()} countries, "
          f"{int(base['final_label'].notna().sum()):,} labels")

    dup = int(base["cluster_id"].duplicated().sum())
    if dup:
        print(f"  WARNING: {dup:,} duplicate cluster_ids across files -- keeping last")
        base = base.drop_duplicates(subset="cluster_id", keep="last").reset_index(drop=True)

    prev = pd.read_parquet(args.prev)

    # --- assert id stability before joining by id ---
    b_ids = base["cluster_id"].astype(str)
    p_ids = prev["cluster_id"].astype(str)
    shared = set(b_ids) & set(p_ids)
    frac = len(shared) / len(b_ids)
    print(f"\nid overlap with {Path(args.prev).name}: {len(shared):,}/{len(b_ids):,} ({frac:.4%})")
    if frac < 0.999:
        raise SystemExit(
            f"cluster_ids are NOT stable ({frac:.2%} overlap) -- ids were renumbered. "
            "Use a proximity-based merge (see merge_clusters_v6.py) instead of this script, "
            "and re-key the patch store (scripts/rekey_patches_by_location.py)."
        )

    bg = base.set_index(b_ids)["geometry"]
    pg = prev.set_index(p_ids)["geometry"]
    ids = sorted(shared)
    c1 = shapely.centroid(shapely.from_wkb(bg.loc[ids].values))
    c2 = shapely.centroid(shapely.from_wkb(pg.loc[ids].values))
    drift = np.hypot(shapely.get_x(c1) - shapely.get_x(c2), shapely.get_y(c1) - shapely.get_y(c2))
    n_moved = int((drift > MAX_DRIFT_DEG).sum())
    print(f"geometry drift on shared ids: max {drift.max():.2e} deg, "
          f"{n_moved:,} beyond {MAX_DRIFT_DEG:g} deg")
    if n_moved:
        raise SystemExit(
            f"{n_moved:,} shared cluster_ids sit on DIFFERENT geometry -- same-id/different-place "
            "is precisely the patch-store corruption mode. Refusing to id-join."
        )
    print("  -> ids verified stable; existing patches remain valid (no re-key needed)")

    # --- carry the columns Rachel's schema lacks, by id (safe: verified above) ---
    have = [c for c in CARRY_COLS if c in prev.columns]
    print(f"\ncarrying {have} from {Path(args.prev).name} by cluster_id ...")
    carry = prev.set_index(p_ids)[have]
    for c in have:
        base[c] = b_ids.map(carry[c]).values
        print(f"  {c}: {int(pd.notna(base[c]).sum()):,} non-null")

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
    print(f"  clusters : {len(base):,}")
    print(f"  countries: {base.ADM0.nunique()}")
    print(f"  labels   : {int(base['final_label'].notna().sum()):,}")
    print("\ncnn_split_assigned:")
    print(base["cnn_split_assigned"].fillna("<null>").value_counts().to_string())
    print("\ntrain rows by country (top 8):")
    print(base[base.cnn_split_assigned == "train"].ADM0.value_counts().head(8).to_string())
    print(f"  ... {base[base.cnn_split_assigned=='train'].ADM0.nunique()} countries have train rows")
    print("\ngeneralization countries:")
    print(base[base.cnn_split_assigned == "generalization"].ADM0.value_counts().to_string())


if __name__ == "__main__":
    main()
