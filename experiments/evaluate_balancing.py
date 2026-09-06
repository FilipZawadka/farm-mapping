"""Evaluate the region-balancing campaign (arms G/H/I + ablation J) against round_4 arm A.

Reuses the round_4 machinery (experiments/evaluate_r4.py: slices, score loading,
per-arm seed replicates, SE_total with the mandatory seed term, Holm) with its
own pre-registered family, and adds the diagnostics this campaign is about:
does the model still score a patch by *where* it is rather than what it shows?

Primary (confirmatory, Holm m=3, docs/COUNTRY_BALANCING_PLAN.md):
    g > a, h > a, i > a on generalization farm ROC-AUC.
    A win needs point estimate > 0.005 AUC AND Holm p < 0.05 (EVAL_METHODS.md).
Secondary (exploratory, unadjusted):
    * j vs g (what class conditioning adds) and j vs a; test / eval slices
    * within-country AUC on generalization -- ranking skill with the
      cross-country calibration drift removed (R4_RUN_NOTES section 10)
    * per-bucket (us / europe / rest) behaviour on val / test / generalization:
      AUC, mean P(farm) on true negatives and positives, FPR and recall at the
      shipped 0.4 threshold. The shortcut signature is a wide spread of the
      negatives' mean score across buckets (US negatives scored high, European
      negatives scored low); a balanced arm should narrow it.
    * per-country FPR@0.4 on val for the NotFarm-dominated countries (RUS, UKR,
      BLR, DEU, ...). val chose the checkpoints, so it is optimistic for every
      arm alike -- but it is the only held-out slice with European / Russian
      rows now that round_4 folded qual_eval into train.
    * sampling_report.json per run: proves an arm really sampled balanced.

Run:  python3 experiments/evaluate_balancing.py
      (--v10 / --gpu-dir override the data locations; --boot lowers the
      bootstrap count for a quick pass)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402
import evaluate_r4 as r4  # noqa: E402
from training.balancing import DEFAULT_BUCKETS, UNKNOWN_ISO3, assign_buckets  # noqa: E402

ARMS = {
    "a": "baseline (round_4 arm A)",
    "g": "grouped countries, uniform, class-cond.",
    "h": "3 buckets us/europe/rest, class-cond.",
    "i": "per-country cap 20%, class-cond.",
    "j": "ABLATION grouped, no class-cond.",
}
BALANCED = ("g", "h", "i", "j")
CONFIRMATORY = [("g", "a"), ("h", "a"), ("i", "a")]
EXPLORATORY = [("j", "g"), ("j", "a")]
THRESHOLD = 0.4          # shipped OOD operating point (EVAL_METHODS E2.1)
DECISION_SLICES = ("generalization", "test", "eval")
DIAG_SLICES = ("generalization", "test", "val")
BUCKET_ORDER = ("us", "europe", "rest")


# ------------------------------------------------------------- helpers
def common_rows(arms: dict) -> list[str]:
    ids = None
    for v in arms.values():
        ids = set(v["ids"]) if ids is None else ids & set(v["ids"])
    return sorted(ids or [])


def arm_matrix(arms: dict, ids: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """y over *ids* and, per arm, the seed-averaged P(farm) over the same ids."""
    first = next(iter(arms.values()))
    pos = {c: i for i, c in enumerate(first["ids"])}
    y = first["y"][[pos[c] for c in ids]]
    P = {}
    for a, v in arms.items():
        idx = {c: i for i, c in enumerate(v["ids"])}
        take = np.array([idx[c] for c in ids])
        P[a] = np.mean([v["P"][s][take] for s in v["P"]], axis=0)
    return y, P


def bucket_table(arms: dict, cmap: pd.Series, threshold: float = THRESHOLD) -> dict:
    """Per-bucket score behaviour, seed-averaged. The shortcut signature is a
    large spread of mean P(farm | NotFarm) across buckets."""
    ids = common_rows(arms)
    if len(ids) < 50:
        return {}
    y, P = arm_matrix(arms, ids)
    iso = cmap.reindex(ids).fillna(UNKNOWN_ISO3).to_numpy().astype(str)
    buckets = assign_buckets(iso, DEFAULT_BUCKETS)
    out: dict = {}
    print(f"\n{'bucket':<8}{'n':>6}{'n_neg':>7}{'n_pos':>6} | {'arm':<4}{'AUC':>8}{'P|neg':>8}{'P|pos':>8}"
          f"{'FPR@.4':>8}{'rec@.4':>8}")
    for b in BUCKET_ORDER:
        m = buckets == b
        if m.sum() < 20:
            continue
        yb = y[m]
        n_neg, n_pos = int((yb == 0).sum()), int((yb == 1).sum())
        out[b] = {"n": int(m.sum()), "n_neg": n_neg, "n_pos": n_pos, "arms": {}}
        for k, a in enumerate(arms):
            p = P[a][m]
            auc = lib.safe_auc(yb, p) if n_neg and n_pos else float("nan")
            p_neg = float(p[yb == 0].mean()) if n_neg else float("nan")
            p_pos = float(p[yb == 1].mean()) if n_pos else float("nan")
            fpr = float((p[yb == 0] >= threshold).mean()) if n_neg else float("nan")
            rec = float((p[yb == 1] >= threshold).mean()) if n_pos else float("nan")
            out[b]["arms"][a] = {"auc": auc, "p_neg": p_neg, "p_pos": p_pos, "fpr": fpr, "recall": rec}
            lead = f"{b:<8}{int(m.sum()):>6}{n_neg:>7}{n_pos:>6}" if k == 0 else " " * 27
            print(f"{lead} | {a:<4}{lib.fmt(auc, 4):>8}{lib.fmt(p_neg, 3):>8}{lib.fmt(p_pos, 3):>8}"
                  f"{lib.fmt(fpr, 3):>8}{lib.fmt(rec, 3):>8}")
    # spread of the negatives' mean score across buckets, per arm
    spread = {}
    for a in arms:
        vals = [out[b]["arms"][a]["p_neg"] for b in out if not np.isnan(out[b]["arms"][a]["p_neg"])]
        spread[a] = float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")
    print("  spread of mean P(farm | NotFarm) across buckets (smaller = less region-driven): "
          + "  ".join(f"{a}={lib.fmt(v, 3)}" for a, v in spread.items()))
    out["p_neg_spread"] = spread
    return out


def country_fpr_table(arms: dict, cmap: pd.Series, min_neg: int = 20,
                      threshold: float = THRESHOLD, max_rows: int = 30) -> dict:
    """Per-country FPR@threshold (and AUC where both classes exist), seed-averaged.
    Includes single-class countries, which per_country_table must skip."""
    ids = common_rows(arms)
    if len(ids) < 50:
        return {}
    y, P = arm_matrix(arms, ids)
    iso = cmap.reindex(ids).fillna(UNKNOWN_ISO3).to_numpy().astype(str)
    counts = pd.Series(iso[y == 0]).value_counts()
    names = [c for c, n in counts.items() if n >= min_neg and c != UNKNOWN_ISO3][:max_rows]
    if not names:
        return {}
    out = {}
    print(f"\n{'country':<8}{'n_neg':>6}{'n_pos':>6} | " + " ".join(f"{'FPR ' + a:>8}" for a in arms)
          + " | " + " ".join(f"{'AUC ' + a:>8}" for a in arms))
    for c in names:
        m = iso == c
        yc = y[m]
        n_neg, n_pos = int((yc == 0).sum()), int((yc == 1).sum())
        row = {"n_neg": n_neg, "n_pos": n_pos, "fpr": {}, "auc": {}}
        for a in arms:
            p = P[a][m]
            row["fpr"][a] = float((p[yc == 0] >= threshold).mean())
            row["auc"][a] = lib.safe_auc(yc, p) if (n_pos >= 5 and n_neg >= 5) else float("nan")
        out[c] = row
        print(f"{c:<8}{n_neg:>6}{n_pos:>6} | " + " ".join(f"{row['fpr'][a]:>8.3f}" for a in arms)
              + " | " + " ".join(f"{lib.fmt(row['auc'][a], 4):>8}" for a in arms))
    return out


def within_country_auc(arms: dict, cmap: pd.Series, min_n: int = 20) -> dict:
    """Mean per-country AUC per seed -> arm mean +/- sd. Removes the between-country
    score offsets that dominate pooled OOD AUC's seed variance."""
    out = {}
    print(f"\n{'arm':<4}{'per-seed within-country AUC':<34}{'mean':>8}{'sd':>8}   countries")
    for a, v in arms.items():
        iso = cmap.reindex(v["ids"]).fillna(UNKNOWN_ISO3).to_numpy().astype(str)
        per_seed = {}
        used = []
        for s, p in v["P"].items():
            vals = []
            for c in sorted(set(iso)):
                m = iso == c
                if m.sum() < min_n or len(np.unique(v["y"][m])) < 2:
                    continue
                vals.append(lib.safe_auc(v["y"][m], p[m]))
                if c not in used:
                    used.append(c)
            per_seed[int(s)] = float(np.mean(vals)) if vals else float("nan")
        arr = np.array(list(per_seed.values()))
        out[a] = {"per_seed": per_seed, "mean": float(np.nanmean(arr)),
                  "sd": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                  "countries": used}
        print(f"{a:<4}{' '.join(f'{x:.4f}' for x in arr):<34}{out[a]['mean']:>8.4f}"
              f"{lib.fmt(out[a]['sd'], 4):>8}   {','.join(used)}")
    return out


def sampling_reports(gpu: Path) -> dict:
    """What each balanced run's sampler actually did (written by train.py)."""
    out = {}
    print(f"\n{'run':<34}{'scheme':<17}{'groups':>7}{'NMI nat->ach':>15}{'ESS':>7}{'w_max':>7}{'clipped':>9}")
    for a in BALANCED:
        for s in r4.SEEDS:
            run = f"world_v10_fourclass_r4_{a}_s{s}"
            f = gpu / run / "sampling_report.json"
            if not f.exists():
                if (gpu / run / "scored_candidates.parquet").exists():
                    print(f"{run:<34}! no sampling_report.json -- cannot verify this run sampled balanced")
                continue
            rep = json.loads(f.read_text())
            dep = rep["label_region_dependence"]
            out[run] = {"scheme": rep["scheme"], "class_conditional": rep["class_conditional"],
                        "n_groups": rep["n_groups"], "nmi_natural": dep["nmi_natural"],
                        "nmi_achieved": dep["nmi_achieved"], "ess_ratio": rep["ess_ratio"],
                        "weight_max": rep["weight_max"], "clipped_frac": rep["clipped_frac"]}
            cc = "" if rep["class_conditional"] else " (no cc)"
            print(f"{run:<34}{rep['scheme'] + cc:<17}{rep['n_groups']:>7}"
                  f"{dep['nmi_natural']:>7.3f}->{dep['nmi_achieved']:<6.3f}{rep['ess_ratio']:>7.3f}"
                  f"{rep['weight_max']:>7.2f}{100 * rep['clipped_frac']:>8.1f}%")
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10", help="master parquet (default: evaluate_r4.V10)")
    ap.add_argument("--gpu-dir", help="collected runs dir (default: experiments/gpu_results)")
    ap.add_argument("--boot", type=int, default=r4.BOOT, help="bootstrap resamples per contrast")
    ap.add_argument("--out", default="balancing_evaluation", help="results/<out>.json")
    args = ap.parse_args()
    if args.v10:
        r4.V10 = Path(args.v10)
    if args.gpu_dir:
        r4.GPU = Path(args.gpu_dir)
    r4.BOOT = args.boot

    lib.header("Region-balancing campaign evaluation (docs/COUNTRY_BALANCING_PLAN.md)")
    SL = r4.slices(("generalization", "test", "eval", "val"))
    print("slices:", {k: len(v) for k, v in SL.items()})

    names = [f"world_v10_fourclass_r4_{a}_s{s}" for a in ARMS for s in r4.SEEDS]
    scores = {n: v for n in names if (v := r4.load_scores(n)) is not None}
    have = [a for a in ARMS if any(f"world_v10_fourclass_r4_{a}_s{s}" in scores for s in r4.SEEDS)]
    print(f"runs loaded: {len(scores)} | arms with data: {have or 'none yet'}")
    if not have:
        print("\nNo campaign runs collected yet -- rerun when training finishes.")
        return
    if "a" not in have:
        print("! round_4 arm A (the control) is not collected; contrasts will be missing")

    cmap = r4._country_map()
    report: dict = {"arms": ARMS, "confirmatory": [f"{x}>{y}" for x, y in CONFIRMATORY]}

    lib.header("sampler verification (sampling_report.json per balanced run)")
    report["sampling_reports"] = sampling_reports(r4.GPU)

    for sname, sl in SL.items():
        lib.header(f"slice: {sname}  (n={len(sl)}, farm rate {sl.y.mean():.2f})"
                   + ("  [checkpoint-selection slice: optimistic for every arm alike]" if sname == "val" else ""))
        arms = {a: r4.arm_auc(scores, sl, a) for a in ARMS}
        arms = {a: v for a, v in arms.items() if v}
        if not arms:
            print("  no arm has scores on this slice")
            continue
        entry: dict = {}

        print(f"{'arm':<4} {'description':<40} {'per-seed AUC':<28} {'mean':>8} {'sd':>8}")
        sigmas = []
        for a, v in arms.items():
            vals = np.array(list(v["per_seed"].values()))
            if len(vals) > 1:
                sigmas.append(vals.std(ddof=1))
            print(f"{a:<4} {ARMS[a]:<40} {' '.join(f'{x:.4f}' for x in vals):<28} {vals.mean():>8.4f} "
                  f"{(vals.std(ddof=1) if len(vals) > 1 else float('nan')):>8.4f}")
        sigma_seed = float(np.mean(sigmas)) if sigmas else 0.0078
        print(f"\npooled sigma_seed on this slice: {sigma_seed:.4f}"
              f"   (single-run decision band 2*sqrt2*sigma = +/-{2 * np.sqrt(2) * sigma_seed:.4f})")
        entry["sigma_seed"] = sigma_seed
        entry["per_arm"] = {a: v["per_seed"] for a, v in arms.items()}

        if sname in DECISION_SLICES and len(arms) > 1:
            print("\ndelta = second arm minus first (positive => second arm better)")
            print(f"\n{'contrast':<12} {'n':>6} {'dAUC_rec':>9} {'dAUC_ens':>9} {'SE_tot':>9} "
                  f"{'z':>7} {'p_raw':>8} {'MDE80':>8}")
            raw_p, rows = {}, {}
            for a1, a2 in itertools.combinations(arms, 2):
                r = r4.compare(arms[a1], arms[a2], sigma_seed)
                key = f"{a1}_vs_{a2}"
                rows[key] = r
                raw_p[key] = r["p_total"]
                print(f"{key:<12} {r['n']:>6} {r['delta']:>+9.4f} {r['delta_ens']:>+9.4f} "
                      f"{r['se_total']:>9.4f} {r['z']:>+7.2f} {r['p_total']:>8.3f} {r['mde80']:>8.4f}")
            entry["contrasts"] = rows

            def _resolve(x, y_):
                if f"{y_}_vs_{x}" in raw_p:
                    return f"{y_}_vs_{x}", +1.0
                if f"{x}_vs_{y_}" in raw_p:
                    return f"{x}_vs_{y_}", -1.0
                return None, 0.0

            conf, orient = {}, {}
            for x, y_ in CONFIRMATORY:
                src, sign = _resolve(x, y_)
                if src:
                    conf[f"{x}>{y_}"] = raw_p[src]
                    orient[f"{x}>{y_}"] = (src, sign)
            if conf:
                adj = r4.holm(conf)
                tag = "confirmatory family (Holm-adjusted)" if sname == "generalization" \
                    else "confirmatory contrasts on a supporting slice (Holm-adjusted, exploratory)"
                print(f"\n{tag}:")
                for k, p in sorted(adj.items(), key=lambda kv: kv[1]):
                    src, sign = orient[k]
                    d = sign * rows[src]["delta"]
                    better, base = k.split(">")
                    verdict = (f"{better} BETTER than {base}" if d > 0 else f"{better} WORSE than {base}") \
                        if (p < 0.05 and abs(d) >= 0.005) else (
                        "significant but below practical floor" if p < 0.05 else "not distinguishable")
                    print(f"  {k:<10} d={d:+.4f}  p_holm={p:.3f}  {verdict}")
                    rows[src]["p_holm"] = p
                    rows[src]["verdict"] = verdict
            elif rows:
                print("\n  ! confirmatory family empty -- no comparable arm pair collected yet")
            expl = []
            for x, y_ in EXPLORATORY:
                src, sign = _resolve(x, y_)
                if src:
                    expl.append(f"{x}>{y_}: d={sign * rows[src]['delta']:+.4f} p_raw={raw_p[src]:.3f}")
            if expl:
                print("exploratory (unadjusted): " + "; ".join(expl))

        lib.header(f"calibration (ECE) -- slice {sname}")
        r4.calibration_table(arms, sl)

        lib.header(f"per-country farm AUC (n>=20, both classes) -- slice {sname}")
        entry["per_country"] = r4.per_country_table(arms, sl, cmap)

        if sname == "generalization":
            lib.header("within-country AUC (mean over countries, per seed) -- generalization")
            entry["within_country"] = within_country_auc(arms, cmap)

        if sname in DIAG_SLICES:
            lib.header(f"per-bucket score behaviour (us / europe / rest) -- slice {sname}")
            entry["buckets"] = bucket_table(arms, cmap)
        if sname == "val":
            lib.header("per-country FPR@0.4 on val (>=20 NotFarm rows; single-class countries included)")
            entry["country_fpr"] = country_fpr_table(arms, cmap)

        report[sname] = entry

    report["per_class"] = r4.class_table([n for n in scores if n.startswith("world_")])
    lib.save(args.out, report)
    print(f"\nsaved -> experiments/results/{args.out}.json")


if __name__ == "__main__":
    main()
