"""Full metrics suite over merged full-world scored parquets.

Computes, for each model, per evaluation slice:

  binary farm-vs-not:  ROC-AUC (+bootstrap 95% CI), PR-AUC, Brier, ECE,
                       precision / recall / FP-rate / F1 at threshold 0.5
  4-class (argmax):    accuracy, balanced accuracy, macro-F1, MCC,
                       per-class one-vs-rest ROC-AUC and PR-AUC,
                       confusion matrix
  comparisons:         matched-FP-rate recall vs a reference model,
                       threshold sweep on the OOD slices,
                       mean-probability ensemble of the last two models

Slice protocol
--------------
test / eval / generalization membership never changed across v6..v9 (the
round_2/round_3 promotions only moved qual_eval -> train/val), so those slices
compare directly. qual_eval DID shrink differently per version -- v8's promoted
rows were a seeded random pick, v9's were Rachel's 70-cap pick -- so the only
slice no model trained on is the INTERSECTION of qual_eval membership across
all supplied models. Reported as ``qual_eval_common``.

Usage::

    python scripts/full_metrics_report.py \
        --models v6=path.parquet v7=path.parquet v8=path.parquet v9=path.parquet \
        --focus v9 --ref v8 \
        --out-md docs/METRICS_v9_full_2026-08-06.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve,
)

MAP4 = {
    "NotFarm": 0,
    "Farm: Poultry: Meat Chickens": 1,
    "Farm: Poultry: Eggs": 1,
    "Farm: Poultry: Unspecified/Other": 1,
    "Farm: Pigs": 2,
    "Farm: Cattle": 3,
}
CLASS_NAMES = ["NotFarm", "Poultry", "Pigs", "Cattle"]
PROB_COLS = ["prob_class0", "prob_class1", "prob_class2", "prob_class3"]
SLICES = ["test", "eval", "generalization", "qual_eval_common"]
RNG = np.random.default_rng(0)


def load_model(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=[
        "candidate_id", "ADM0", "final_label", "cnn_split_assigned", *PROB_COLS,
    ])
    df["y4"] = df.final_label.map(MAP4)
    df["p_farm"] = df[PROB_COLS[1:]].sum(axis=1)
    return df.set_index("candidate_id")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    stats = []
    n = len(y)
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def binary_block(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    lo, hi = auc_ci(y, p)
    return {
        "n": int(len(y)), "n_farm": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)), "auc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "ece": float(ece(y, p)),
        "recall@0.5": tp / max(tp + fn, 1), "precision@0.5": tp / max(tp + fp, 1),
        "fp_rate@0.5": fp / max(fp + tn, 1),
        "f1@0.5": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def multiclass_block(y4: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float((pred == y4).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y4, pred)),
        "macro_f1": float(f1_score(y4, pred, average="macro", labels=range(4), zero_division=0)),
        "mcc": float(matthews_corrcoef(y4, pred)),
        "confusion": confusion_matrix(y4, pred, labels=range(4)).tolist(),
        "per_class": {},
    }
    for c in range(4):
        yc = (y4 == c).astype(int)
        if 0 < yc.sum() < len(yc):
            out["per_class"][CLASS_NAMES[c]] = {
                "n": int(yc.sum()),
                "ovr_auc": float(roc_auc_score(yc, probs[:, c])),
                "ap": float(average_precision_score(yc, probs[:, c])),
                "f1": float(f1_score(y4, pred, average=None, labels=range(4), zero_division=0)[c]),
            }
    return out


def matched_fp_recall(y, p_focus, p_ref, thr_ref: float = 0.5) -> dict:
    """Recall of focus at the threshold matching ref's FP-rate at thr_ref."""
    fpr_ref = ((p_ref >= thr_ref) & (y == 0)).sum() / max((y == 0).sum(), 1)
    fpr, tpr, thr = roc_curve(y, p_focus)
    i = int(np.searchsorted(fpr, fpr_ref, side="right")) - 1
    return {
        "ref_fp_rate": float(fpr_ref),
        "focus_recall_at_matched_fp": float(tpr[max(i, 0)]),
        "focus_threshold": float(thr[max(i, 0)]),
        "ref_recall@0.5": float(((p_ref >= thr_ref) & (y == 1)).sum() / max((y == 1).sum(), 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs, oldest first")
    ap.add_argument("--focus", required=True)
    ap.add_argument("--ref", required=True, help="reference model for matched-FP + ensemble")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--sweep", nargs="+", type=float,
                    default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    args = ap.parse_args()

    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = load_model(path)
        print(f"loaded {name}: {len(models[name]):,} rows")

    focus = models[args.focus]

    # qual_eval intersection across every supplied model (no model trained on it)
    qual_common = None
    for m in models.values():
        s = set(m.index[m.cnn_split_assigned == "qual_eval"])
        qual_common = s if qual_common is None else (qual_common & s)
    print(f"qual_eval_common: {len(qual_common):,} ids "
          f"(vs {int((focus.cnn_split_assigned == 'qual_eval').sum()):,} in {args.focus} alone)")

    def slice_mask(df: pd.DataFrame, sl: str) -> pd.Series:
        if sl == "qual_eval_common":
            return df.index.isin(qual_common) & df.y4.notna()
        return (df.cnn_split_assigned == sl) & df.y4.notna()

    report: dict = {"slice_sizes": {}, "models": {}, "comparisons": {}}
    for sl in SLICES:
        report["slice_sizes"][sl] = int(slice_mask(focus, sl).sum())

    for name, df in models.items():
        entry = {}
        for sl in SLICES:
            d = df[slice_mask(df, sl)]
            if d.empty:
                continue
            y4 = d.y4.astype(int).to_numpy()
            probs = d[PROB_COLS].to_numpy()
            y = (y4 >= 1).astype(int)
            p = d.p_farm.to_numpy()
            entry[sl] = {"binary": binary_block(y, p), "fourclass": multiclass_block(y4, probs)}
        report["models"][name] = entry

    # --- comparisons on the OOD slices ---
    ref = models[args.ref]
    for sl in ("generalization", "qual_eval_common"):
        d = focus[slice_mask(focus, sl)]
        common = d.index.intersection(ref.index)
        d = d.loc[common]
        y = (d.y4.astype(int) >= 1).astype(int).to_numpy()
        p_f = d.p_farm.to_numpy()
        p_r = ref.p_farm.reindex(common).to_numpy()
        comp = {
            "matched_fp_vs_" + args.ref: matched_fp_recall(y, p_f, p_r),
            "ensemble_mean_auc": float(roc_auc_score(y, (p_f + p_r) / 2)),
            "focus_auc": float(roc_auc_score(y, p_f)),
            "ref_auc": float(roc_auc_score(y, p_r)),
            "threshold_sweep": {
                str(t): {
                    "recall": float(((p_f >= t) & (y == 1)).sum() / max(y.sum(), 1)),
                    "fp_rate": float(((p_f >= t) & (y == 0)).sum() / max((y == 0).sum(), 1)),
                    "precision": float(((p_f >= t) & (y == 1)).sum() / max((p_f >= t).sum(), 1)),
                } for t in args.sweep
            },
        }
        report["comparisons"][sl] = comp

    # --- per-country farm AUC for the focus model (OOD pool) ---
    ood = focus[slice_mask(focus, "generalization") | slice_mask(focus, "qual_eval_common")]
    rows = []
    for iso, g in ood.groupby("ADM0"):
        y = (g.y4.astype(int) >= 1).astype(int).to_numpy()
        if len(g) >= 20 and 0 < y.sum() < len(y):
            rows.append({"ADM0": iso, "n": len(g), "n_farm": int(y.sum()),
                         "auc": float(roc_auc_score(y, g.p_farm))})
    report["per_country_ood"] = sorted(rows, key=lambda r: r["auc"])

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1))
        print("wrote", args.out_json)
    if args.out_md:
        Path(args.out_md).write_text(render_md(report, args))
        print("wrote", args.out_md)


def render_md(rep: dict, args) -> str:
    L = [f"# Full metrics — focus `{args.focus}` (ref `{args.ref}`)", ""]
    L.append("Slice sizes: " + ", ".join(f"{k}={v:,}" for k, v in rep["slice_sizes"].items()))
    L.append("")
    for sl in SLICES:
        L.append(f"## {sl}")
        L.append("")
        L.append("| model | ROC-AUC | 95% CI | PR-AUC | Brier | ECE | rec@.5 | prec@.5 | FPR@.5 | bal-acc | macro-F1 | MCC |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, entry in rep["models"].items():
            if sl not in entry:
                continue
            b, m = entry[sl]["binary"], entry[sl]["fourclass"]
            L.append(f"| {name} | {b['roc_auc']:.3f} | [{b['auc_ci95'][0]:.3f}, {b['auc_ci95'][1]:.3f}] "
                     f"| {b['pr_auc']:.3f} | {b['brier']:.3f} | {b['ece']:.3f} "
                     f"| {b['recall@0.5']:.3f} | {b['precision@0.5']:.3f} | {b['fp_rate@0.5']:.3f} "
                     f"| {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {m['mcc']:.3f} |")
        L.append("")
        foc = rep["models"][args.focus].get(sl)
        if foc:
            L.append(f"Per-class ({args.focus}):")
            L.append("")
            L.append("| class | n | OvR-AUC | AP | F1 |")
            L.append("|---|---|---|---|---|")
            for cname, c in foc["fourclass"]["per_class"].items():
                L.append(f"| {cname} | {c['n']} | {c['ovr_auc']:.3f} | {c['ap']:.3f} | {c['f1']:.3f} |")
            L.append("")
            cm = foc["fourclass"]["confusion"]
            L.append(f"Confusion ({args.focus}, rows=true): " +
                     "; ".join(f"{CLASS_NAMES[i]}: {row}" for i, row in enumerate(cm)))
            L.append("")
    for sl, comp in rep["comparisons"].items():
        L.append(f"## Comparisons — {sl}")
        L.append("")
        L.append(f"- focus AUC {comp['focus_auc']:.3f} vs ref {comp['ref_auc']:.3f}; "
                 f"mean-prob ensemble {comp['ensemble_mean_auc']:.3f}")
        mf = comp["matched_fp_vs_" + args.ref]
        L.append(f"- at ref's FP-rate ({mf['ref_fp_rate']:.3f}): focus recall "
                 f"{mf['focus_recall_at_matched_fp']:.3f} (thr {mf['focus_threshold']:.3f}) "
                 f"vs ref recall@0.5 {mf['ref_recall@0.5']:.3f}")
        L.append("")
        L.append("| thr | recall | FP-rate | precision |")
        L.append("|---|---|---|---|")
        for t, v in comp["threshold_sweep"].items():
            L.append(f"| {t} | {v['recall']:.3f} | {v['fp_rate']:.3f} | {v['precision']:.3f} |")
        L.append("")
    L.append("## Per-country farm AUC (focus, OOD pool, n>=20, both classes present)")
    L.append("")
    L.append("| country | n | n_farm | AUC |")
    L.append("|---|---|---|---|")
    for r in rep["per_country_ood"]:
        L.append(f"| {r['ADM0']} | {r['n']} | {r['n_farm']} | {r['auc']:.3f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
