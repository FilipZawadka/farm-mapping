"""Post-hoc logit adjustment sweep on saved inference outputs.

Zero-retraining fix for minority-class collapse (Menon et al. 2021,
https://arxiv.org/abs/2007.07314): re-decide predictions as

    argmax_y [ log p_y(x) - tau * log prior_y ]

where prior_y is the class frequency in the TRAIN split and tau is swept on
the VAL split (macro-F1). Reports the chosen tau's metrics on every labelled
slice. If OtherFarm F1 jumps here, the backbone features were fine and only
the decision rule was miscalibrated.

Usage::

    python -m scripts.logit_adjust_sweep --scored data/output/<stem>/scored_candidates.parquet
    # or resolve from a config:
    python -m scripts.logit_adjust_sweep --config configs/rachel_clusters/world_v8_three_class.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    class_ids = list(range(n_classes))
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=class_ids, zero_division=0)), 4),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=class_ids, zero_division=0)
    for i, v in zip(class_ids, per_class):
        out[f"f1_class{i}"] = round(float(v), 4)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=class_ids).tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc logit adjustment sweep")
    parser.add_argument("--scored", default=None, help="Path to scored_candidates.parquet")
    parser.add_argument("--config", default=None, help="Resolve scored parquet from config")
    parser.add_argument("--taus", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--select-split", default="val",
                        help="Split used to pick tau (default: val)")
    args = parser.parse_args()

    if args.scored:
        scored_path = Path(args.scored)
    elif args.config:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from training.config import load_config, resolve_paths
        cfg = resolve_paths(load_config(args.config))
        scored_path = (Path(cfg.patches.output_dir).parent / "output"
                       / cfg._config_stem / "scored_candidates.parquet")
    else:
        raise SystemExit("Pass --scored or --config")

    df = pd.read_parquet(scored_path)
    prob_cols = sorted(c for c in df.columns if c.startswith("prob_class"))
    if not prob_cols:
        raise SystemExit("No prob_class* columns — run multi-class inference first")
    n_classes = len(prob_cols)

    labelled = df[df["true_label"] >= 0].copy()
    train = labelled[labelled["split"] == "train"]
    if len(train) == 0:
        raise SystemExit("No train rows in scored parquet — cannot compute priors "
                         "(run inference without labeled_only filtering train away)")
    priors = np.bincount(train["true_label"].astype(int), minlength=n_classes).astype(float)
    priors /= priors.sum()
    print(f"Train priors: {np.round(priors, 4).tolist()}")

    log_priors = np.log(np.maximum(priors, 1e-12))

    def predict(rows: pd.DataFrame, tau: float) -> np.ndarray:
        logp = np.log(np.maximum(rows[prob_cols].to_numpy(dtype=float), 1e-12))
        return (logp - tau * log_priors).argmax(axis=1)

    taus = [float(t) for t in args.taus.split(",")]
    sel = labelled[labelled["split"] == args.select_split]
    if len(sel) == 0:
        raise SystemExit(f"No rows in selection split {args.select_split!r}")

    best_tau, best_f1 = 0.0, -1.0
    print(f"\ntau sweep on split={args.select_split!r} (n={len(sel)}):")
    for tau in taus:
        m = _metrics(sel["true_label"].to_numpy(int), predict(sel, tau), n_classes)
        marker = ""
        if m["f1_macro"] > best_f1:
            best_tau, best_f1 = tau, m["f1_macro"]
            marker = "  <-- best"
        per_cls = " ".join(f"c{i}={m[f'f1_class{i}']:.3f}" for i in range(n_classes))
        print(f"  tau={tau:4.2f}  macroF1={m['f1_macro']:.4f}  {per_cls}{marker}")

    print(f"\nSelected tau={best_tau}")
    report = {"tau": best_tau, "priors": priors.tolist(), "select_split": args.select_split}
    for split in ("test", "inspected", "eval", "generalization"):
        rows = labelled[labelled["split"] == split]
        if len(rows) == 0:
            continue
        base = _metrics(rows["true_label"].to_numpy(int), predict(rows, 0.0), n_classes)
        adj = _metrics(rows["true_label"].to_numpy(int), predict(rows, best_tau), n_classes)
        report[split] = {"n": len(rows), "baseline": base, "adjusted": adj}
        print(f"\n== {split} (n={len(rows)}) ==")
        print(f"  baseline: macroF1={base['f1_macro']:.4f}  "
              + " ".join(f"c{i}={base[f'f1_class{i}']:.3f}" for i in range(n_classes)))
        print(f"  adjusted: macroF1={adj['f1_macro']:.4f}  "
              + " ".join(f"c{i}={adj[f'f1_class{i}']:.3f}" for i in range(n_classes)))

    out_path = scored_path.parent / "logit_adjust_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
