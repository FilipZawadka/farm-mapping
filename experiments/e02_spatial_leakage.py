"""E0.2 -- Spatial leakage quantification and blocked-split construction.

Post-hoc portion of the plan entry: we cannot retrain on blocked splits without a
GPU, but we CAN (a) measure the leakage exposure exactly, (b) emit a blocked split
assignment for the next training run, and (c) bound the inflation by re-scoring the
existing production model on the leakage-free subset of each held-out slice.

(c) is a lower bound on the true inflation: the model still *trained* on the nearby
clusters, so this measures "how much of the reported score comes from evaluating on
near-duplicates", not "what would the model score if retrained without them".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lib

PATCH_M = 1280.0  # one patch width at 128 px x 10 m
BLOCK_RADII = [500.0, PATCH_M, 2560.0, 5000.0]
HELD_OUT = ["val", "test", "eval", "generalization", "qual_eval"]


def main() -> None:
    lib.header("E0.2  Spatial leakage and blocked splits")
    df = lib.load(lib.FOURCLASS["v9"])
    lab = lib.labeled(df)

    train = lab[lab.cnn_split_assigned == "train"]
    train_xy = lib.latlng(train)
    print(f"train reference clusters: {len(train):,}")

    # ---------------------------------------------------------------- exposure
    rows = []
    dist_by_split: dict[str, np.ndarray] = {}
    for split in HELD_OUT:
        s = lab[lab.cnn_split_assigned == split]
        if s.empty:
            continue
        d = lib.nearest_distance_m(lib.latlng(s), train_xy)
        dist_by_split[split] = d
        rec = {"split": split, "n": len(s), "median_km": float(np.median(d)) / 1000.0}
        for r in BLOCK_RADII:
            rec[f"within_{int(r)}m"] = float((d < r).mean())
        rows.append(rec)

    exposure = pd.DataFrame(rows)
    print("\nFraction of held-out clusters within X metres of a TRAIN cluster:")
    print(exposure.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------- inflation at 1.28 km
    print(f"\nProduction-model metrics, near vs far (block radius {PATCH_M:.0f} m):")
    inflation = []
    for split in HELD_OUT:
        if split not in dist_by_split:
            continue
        s = lab[lab.cnn_split_assigned == split].copy()
        s["dist_to_train_m"] = dist_by_split[split]
        near, far = s[s.dist_to_train_m < PATCH_M], s[s.dist_to_train_m >= PATCH_M]
        if len(far) < 30 or len(near) < 30:
            print(f"  {split}: skipped (near={len(near)}, far={len(far)})")
            continue

        y_n, p_n = lib.farm_binary(near)
        y_f, p_f = lib.farm_binary(far)
        auc_n, ci_n = lib.bootstrap_ci(lib.safe_auc, y_n, p_n)
        auc_f, ci_f = lib.bootstrap_ci(lib.safe_auc, y_f, p_f)
        mf_n, mf_f = lib.macro_f1_multiclass(near), lib.macro_f1_multiclass(far)

        inflation.append({
            "split": split,
            "n_near": len(near), "n_far": len(far),
            "auc_near": auc_n, "auc_near_ci": ci_n,
            "auc_far": auc_f, "auc_far_ci": ci_f,
            "auc_inflation": auc_n - auc_f,
            "macroF1_near": mf_n, "macroF1_far": mf_f,
            "macroF1_inflation": mf_n - mf_f,
        })
        print(
            f"  {split:<16} near n={len(near):<6} AUC={lib.fmt(auc_n)}  |  "
            f"far n={len(far):<6} AUC={lib.fmt(auc_f)}  |  "
            f"dAUC={lib.fmt(auc_n - auc_f):>7}  dMacroF1={lib.fmt(mf_n - mf_f):>7}"
        )

    # ------------------------------------------------------- blocked splits
    # Demote any held-out cluster within one patch width of a train cluster.
    # Emitted for the next training run; nothing is retrained here.
    blocked = lab[["candidate_id", "country", "cnn_split_assigned", "lat", "lng", "true_label"]].copy()
    blocked["dist_to_train_m"] = np.nan
    for split, d in dist_by_split.items():
        blocked.loc[blocked.cnn_split_assigned == split, "dist_to_train_m"] = d
    blocked["blocked_split"] = blocked["cnn_split_assigned"]
    demote = blocked.cnn_split_assigned.isin(["val", "test", "eval", "generalization"]) & (
        blocked.dist_to_train_m < PATCH_M
    )
    blocked.loc[demote, "blocked_split"] = "excluded_spatial_leak"

    out_csv = lib.RESULTS / "e02_blocked_splits.csv"
    blocked.to_csv(out_csv, index=False)
    print(f"\nBlocked-split artifact -> {out_csv}")
    print(blocked.blocked_split.value_counts().to_string())
    print(f"demoted for leakage: {int(demote.sum()):,} rows")

    lib.save("e02_spatial_leakage", {
        "block_radius_m": PATCH_M,
        "train_reference_n": len(train),
        "exposure": exposure.to_dict("records"),
        "inflation": inflation,
        "demoted_rows": int(demote.sum()),
    })


if __name__ == "__main__":
    main()
