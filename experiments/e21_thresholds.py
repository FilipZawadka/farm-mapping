"""E2.1 -- Domain-conditional decision thresholds.

Plan decision rule: ship if OOD recall improves >=5 points at <=5 points FPR cost.

Honesty constraints:
  * The global threshold is fitted on `val` only, then applied to every other slice.
  * Per-country / per-domain thresholds are fitted leave-one-country-out, so a
    country's threshold is never fitted on that country's own rows. Without this,
    a "tuned threshold" is just an in-sample fit and the reported gain is fake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lib

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
OOD_SLICES = ["generalization"]


def best_threshold(y: np.ndarray, p: np.ndarray, objective: str = "f1") -> float:
    """Threshold maximising the chosen objective on the given rows."""
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best_v = 0.5, -np.inf
    for t in GRID:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        v = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        if objective == "balanced_acc":
            v = (rec + spec) / 2
        elif objective == "youden":
            v = rec + spec - 1
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t)


def main() -> None:
    lib.header("E2.1  Domain-conditional thresholds")
    df = lib.load(lib.FOURCLASS["v9"])
    lab = lib.labeled(df)

    # ------------------------------------------------ global threshold on val
    val = lab[lab.cnn_split_assigned == "val"]
    y_v, p_v = lib.farm_binary(val)
    t_global = best_threshold(y_v, p_v, "f1")
    t_global_ba = best_threshold(y_v, p_v, "balanced_acc")
    print(f"val-optimal global threshold: F1 {t_global:.2f} | balanced-acc {t_global_ba:.2f} "
          f"(default 0.50, n={len(val):,})")

    print("\nGlobal retune effect (val-fitted threshold applied to each slice):")
    global_rows = []
    for split in ["test", "eval", "generalization", "qual_eval"]:
        s = lab[lab.cnn_split_assigned == split]
        if s.empty:
            continue
        y, p = lib.farm_binary(s)
        d = lib.binary_suite(y, p, 0.5)
        t = lib.binary_suite(y, p, t_global)
        global_rows.append({
            "split": split, "n": len(s),
            "recall@0.5": d["recall"], "fpr@0.5": d["fpr"],
            f"recall@{t_global}": t["recall"], f"fpr@{t_global}": t["fpr"],
            "d_recall": t["recall"] - d["recall"], "d_fpr": t["fpr"] - d["fpr"],
        })
        print(f"  {split:<16} recall {d['recall']:.3f}->{t['recall']:.3f} "
              f"({t['recall']-d['recall']:+.3f})   FPR {d['fpr']:.3f}->{t['fpr']:.3f} "
              f"({t['fpr']-d['fpr']:+.3f})")

    # ------------------------------------- domain-conditional (leave-one-country-out)
    print("\nDomain-conditional threshold, leave-one-country-out on the OOD slice:")
    ood = lab[lab.cnn_split_assigned.isin(OOD_SLICES)].copy()
    countries = sorted(ood.country.dropna().unique())

    loco_pred, loco_y, loco_p, loco_thr, per_country = [], [], [], [], []
    for c in countries:
        held = ood[ood.country == c]
        rest = ood[ood.country != c]
        y_r, p_r = lib.farm_binary(rest)
        t_c = best_threshold(y_r, p_r, "f1")
        y_h, p_h = lib.farm_binary(held)
        loco_pred.append((p_h >= t_c).astype(int))
        loco_y.append(y_h)
        loco_p.append(p_h)  # keep aligned with loco_y for the baseline comparison
        loco_thr.append(t_c)
        base = lib.binary_suite(y_h, p_h, 0.5)
        tuned = lib.binary_suite(y_h, p_h, t_c)
        per_country.append({
            "country": c, "n": len(held), "threshold": t_c,
            "recall@0.5": base["recall"], "recall_tuned": tuned["recall"],
            "fpr@0.5": base["fpr"], "fpr_tuned": tuned["fpr"],
            "auc": base["roc_auc"],
        })
        print(f"  {c:<6} n={len(held):<5} thr={t_c:.2f}  recall {base['recall']:.3f}->{tuned['recall']:.3f}"
              f"   FPR {base['fpr']:.3f}->{tuned['fpr']:.3f}")

    y_all = np.concatenate(loco_y)
    pred_all = np.concatenate(loco_pred)
    p_all = np.concatenate(loco_p)  # same row order as y_all
    base_all = lib.binary_suite(y_all, p_all, 0.5)

    tp = int(((pred_all == 1) & (y_all == 1)).sum())
    fp = int(((pred_all == 1) & (y_all == 0)).sum())
    fn = int(((pred_all == 0) & (y_all == 1)).sum())
    tn = int(((pred_all == 0) & (y_all == 0)).sum())
    rec_loco = tp / (tp + fn)
    fpr_loco = fp / (fp + tn)

    print(f"\n  Pooled OOD @0.50 default : recall {base_all['recall']:.3f}  FPR {base_all['fpr']:.3f}")
    print(f"  Pooled OOD LOCO-tuned    : recall {rec_loco:.3f}  FPR {fpr_loco:.3f}")
    print(f"  Delta                    : recall {rec_loco-base_all['recall']:+.3f}  "
          f"FPR {fpr_loco-base_all['fpr']:+.3f}")

    # ---------------------------------- fixed OOD screening thresholds (sweep)
    print("\nFixed OOD screening thresholds (pooled generalization slice):")
    y_o, p_o = lib.farm_binary(ood)
    sweep = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        m = lib.binary_suite(y_o, p_o, t)
        sweep.append({"threshold": t, **{k: m[k] for k in ("recall", "fpr", "precision", "f1_farm")}})
        print(f"  thr={t:.1f}  recall={m['recall']:.3f}  FPR={m['fpr']:.3f}  "
              f"prec={m['precision']:.3f}  F1={m['f1_farm']:.3f}")

    # Decision rule: find every threshold meeting >=+5pt recall at <=+5pt FPR
    # versus the 0.5 default, and recommend the one with the most recall.
    at_05 = next(s for s in sweep if s["threshold"] == 0.5)
    print("\nDecision rule (>=+5pt recall for <=+5pt FPR vs the 0.50 default):")
    qualifying = []
    for s in sweep:
        if s["threshold"] >= 0.5:
            continue
        d_rec = s["recall"] - at_05["recall"]
        d_fpr = s["fpr"] - at_05["fpr"]
        ok = (d_rec >= 0.05) and (d_fpr <= 0.05)
        print(f"  thr={s['threshold']:.1f}  d_recall={d_rec:+.3f}  d_FPR={d_fpr:+.3f}  "
              f"{'PASS' if ok else 'fail'}")
        if ok:
            qualifying.append({"threshold": s["threshold"], "d_recall": d_rec, "d_fpr": d_fpr})

    if qualifying:
        rec_thr = max(qualifying, key=lambda q: q["d_recall"])
        print(f"\n==> SHIP threshold {rec_thr['threshold']:.1f} for OOD screening "
              f"(recall {rec_thr['d_recall']:+.3f} for FPR {rec_thr['d_fpr']:+.3f})")
    else:
        rec_thr = None
        print("\n==> DO NOT SHIP: no threshold satisfies the rule")

    lib.save("e21_thresholds", {
        "global_threshold_val_fitted": t_global,
        "global_threshold_balanced_acc": t_global_ba,
        "global_retune_effect": global_rows,
        "loco_per_country": per_country,
        "pooled_ood_default": {"recall": base_all["recall"], "fpr": base_all["fpr"]},
        "pooled_ood_loco": {"recall": rec_loco, "fpr": fpr_loco},
        "fixed_threshold_sweep": sweep,
        "decision": {
            "qualifying_thresholds": qualifying,
            "recommended": rec_thr,
            "ship": bool(qualifying),
        },
    })


if __name__ == "__main__":
    main()
