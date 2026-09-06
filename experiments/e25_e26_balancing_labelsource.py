"""E2.5 (balanced sampling under spatial blocking) and E2.6 (label-source shift).

E2.5 hypothesis: the replicated eval-gain / gen-loss trade from class-balanced
sampling is partly a spatial-leakage artefact. If the eval gain is carried by
clusters near training clusters, it shrinks once those rows are removed.
Both models already exist (v9 and v9_bal), so this needs no GPU -- but note it
re-evaluates fixed models on a leakage-free subset rather than retraining on
blocked splits, so it tests "is the gain concentrated in leaked rows", which is
the diagnostic half of the plan entry.

E2.6 hypothesis: train/eval degradation is a measurement-instrument change
(registry vs visual labels), not overfitting. The training half needs a GPU; the
diagnostic half -- does performance differ by label source at fixed model -- runs
here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import lib

PATCH_M = 1280.0


def macro_f1(df: pd.DataFrame) -> float:
    return lib.macro_f1_multiclass(df, 4)


def e25() -> dict:
    lib.header("E2.5  Class-balanced sampling, near vs far from training clusters")
    v9 = lib.labeled(lib.load("world_v10_fourclass_v9"))
    vb = lib.labeled(lib.load("world_v10_fourclass_v9_bal"))
    train_xy = lib.latlng(v9[v9.cnn_split_assigned == "train"])

    out = []
    for split in ["eval", "generalization"]:
        a = v9[v9.cnn_split_assigned == split].drop_duplicates("candidate_id").set_index("candidate_id")
        b = vb[vb.cnn_split_assigned == split].drop_duplicates("candidate_id").set_index("candidate_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        d = lib.nearest_distance_m(a[["lat", "lng"]].to_numpy(dtype=float), train_xy)
        near_m, far_m = d < PATCH_M, d >= PATCH_M

        print(f"\n  {split}: n={len(a)}  near={int(near_m.sum())}  far={int(far_m.sum())}")
        rec = {"split": split, "n": len(a), "n_near": int(near_m.sum()), "n_far": int(far_m.sum())}
        for label, mask in [("all", np.ones(len(a), bool)), ("near", near_m), ("far", far_m)]:
            if mask.sum() < 30:
                print(f"    {label:<5} skipped (n={int(mask.sum())})")
                continue
            fa, fb = macro_f1(a[mask]), macro_f1(b[mask])
            ya, pa = lib.farm_binary(a[mask])
            _, pb = lib.farm_binary(b[mask])
            dauc, ci, pv = lib.paired_bootstrap_delta(lib.safe_auc, ya, pa, pb)
            rec[label] = {"n": int(mask.sum()), "macroF1_v9": fa, "macroF1_bal": fb,
                          "macroF1_delta": fb - fa, "auc_delta": dauc, "auc_ci": list(ci),
                          "auc_p": pv}
            print(f"    {label:<5} n={int(mask.sum()):<5} macroF1 v9={fa:.4f} bal={fb:.4f} "
                  f"delta={fb-fa:+.4f}   dAUC={dauc:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={pv:.3f}")
        out.append(rec)

    ev = next(r for r in out if r["split"] == "eval")
    if "near" in ev and "far" in ev:
        print(f"\n  Balanced-sampling eval gain: near rows {ev['near']['macroF1_delta']:+.4f}  "
              f"vs far rows {ev['far']['macroF1_delta']:+.4f}")
        leak_driven = ev["near"]["macroF1_delta"] > ev["far"]["macroF1_delta"]
        print(f"  => gain {'IS' if leak_driven else 'is NOT'} concentrated in leakage-exposed rows")
    return {"results": out}


def e26() -> dict:
    lib.header("E2.6  Label-source stratification (registry vs visual)")
    v9 = lib.labeled(lib.load(lib.FOURCLASS["v9"]))

    def source_group(s):
        if pd.isna(s) or str(s).strip() == "":
            return "none/pipeline"
        return "visual" if "visual" in str(s).lower() else "registry"

    out = []
    for split in ["train", "test", "eval", "generalization", "qual_eval"]:
        s = v9[v9.cnn_split_assigned == split].copy()
        if s.empty:
            continue
        s["src"] = s["label_source"].map(source_group)
        print(f"\n  {split} (n={len(s):,}):  " +
              "  ".join(f"{k}={v}" for k, v in s.src.value_counts().to_dict().items()))
        for src, g in s.groupby("src"):
            if len(g) < 30:
                continue
            y, p = lib.farm_binary(g)
            auc = lib.safe_auc(y, p)
            mf = macro_f1(g)
            farm_rate = float(y.mean())
            out.append({"split": split, "source": src, "n": len(g), "farm_rate": farm_rate,
                        "auc": auc, "macro_f1": mf})
            print(f"    {src:<15} n={len(g):<6} farm_rate={farm_rate:.3f}  "
                  f"AUC={lib.fmt(auc, 4)}  macroF1={mf:.4f}")

    # Direct test of the shift claim: compare sources within the SAME split so
    # domain is held constant and only the labelling instrument varies.
    print("\n  Within-split source contrast (domain held constant):")
    df = pd.DataFrame(out)
    contrasts = []
    for split, g in df.groupby("split"):
        if g.source.nunique() < 2:
            continue
        piv = g.set_index("source")
        if {"visual", "registry"}.issubset(piv.index):
            d_auc = piv.loc["visual", "auc"] - piv.loc["registry", "auc"]
            d_f1 = piv.loc["visual", "macro_f1"] - piv.loc["registry", "macro_f1"]
            contrasts.append({"split": split, "visual_minus_registry_auc": d_auc,
                              "visual_minus_registry_macroF1": d_f1})
            print(f"    {split:<16} visual-registry: dAUC={d_auc:+.4f}  dMacroF1={d_f1:+.4f}")
    return {"by_split_source": out, "contrasts": contrasts}


if __name__ == "__main__":
    lib.save("e25_e26_balancing_labelsource", {"e25": e25(), "e26": e26()})
