"""Round_4 evaluation, implementing experiments/EVAL_METHODS.md.

Design: 5 single-factor arms x 3 seeds. Comparisons are made ARM vs ARM using
per-arm seed replicates, so a recipe effect is separated from seed noise rather
than confounded with it.

Key methodological point (verified on our own data, see EVAL_METHODS.md): the row
bootstrap alone declares 3 of 10 IDENTICAL-recipe seed pairs "significant" on the
generalization slice, because it conditions on fitted scores and is blind to seed
variance. Every contrast here therefore carries

    SE_total = sqrt(SE_boot^2 + 2 * sigma_seed^2)

and Holm correction is applied across the pre-registered confirmatory family.

Run: python3 experiments/evaluate_r4.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

V10 = lib.REPO / "data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet"
GPU = Path(__file__).resolve().parent / "gpu_results"
ARMS = {"a": "baseline (v9/v6 recipe)", "b": "freeze0 only", "c": "6 bands only",
        "d": "freeze0 + 6 bands", "e": "DenseNet-121 (+freeze0+6bands)"}
SEEDS = (42, 43, 44)
CLASSES = ["NotFarm", "Poultry", "Pigs", "Cattle"]
BOOT = 10_000   # EVAL_METHODS: raised from 2,000 so p-values are not floored at 2/2000

# Pre-registered confirmatory family (EVAL_METHODS section 4). Everything else
# printed by this script is exploratory and labelled as such.
CONFIRMATORY = [("b", "a"), ("d", "a"), ("e", "d")]


# ---------------------------------------------------------------- slices
def slices() -> dict[str, pd.DataFrame]:
    v10 = pd.read_parquet(V10)
    v10["cid"] = v10.cluster_id.astype(str)
    out = {}
    for name in ("generalization", "test", "eval"):
        d = v10[v10.cnn_split_assigned == name][["cid", "final_label"]].copy()
        d["y"] = d.final_label.map(lambda l: np.nan if pd.isna(l) else int(l != "NotFarm"))
        d = d.dropna(subset=["y"])
        if d.y.nunique() > 1:          # binary AUC needs both classes present
            out[name] = d.set_index("cid")
        else:
            print(f"  ! slice {name} dropped: only one class present")
    return out


def load_scores(run: str) -> pd.Series | None:
    """P(farm) by candidate id for a run name, or None if not collected."""
    p = GPU / run / "scored_candidates.parquet"
    if not p.exists():
        arch = {"v6": "v6", "v7": "v7", "v8": "v8", "v9": "v9"}.get(run)
        if arch is None:
            return None
        d = lib.load(lib.FOURCLASS[arch])
    else:
        d = pd.read_parquet(p)
    d["cid"] = d.candidate_id.astype(str)
    d = d.drop_duplicates("cid").set_index("cid")
    return (1.0 - d["prob_class0"]).dropna()


# ---------------------------------------------------------------- stats
def holm(pvals: dict) -> dict:
    """Holm-Bonferroni; valid under arbitrary dependence (models share rows)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def arm_auc(scores: dict, sl: pd.DataFrame, arm: str) -> dict:
    """Per-seed AUCs for one arm on one slice, over rows common to all its seeds."""
    runs = {s: scores.get(f"world_v10_fourclass_r4_{arm}_s{s}") for s in SEEDS}
    runs = {s: v for s, v in runs.items() if v is not None}
    if not runs:
        return {}
    common = set(sl.index)
    for v in runs.values():
        common &= set(v.index)
    common = sorted(common)
    if len(common) < 50:
        return {}
    y = sl.loc[common, "y"].to_numpy().astype(int)
    return {"ids": common, "y": y,
            "per_seed": {s: lib.safe_auc(y, v.loc[common].to_numpy()) for s, v in runs.items()},
            "P": {s: v.loc[common].to_numpy() for s, v in runs.items()}}


def compare(a1: dict, a2: dict, sigma_seed: float) -> dict:
    """Arm-vs-arm on shared rows: bootstrap SE + mandatory seed term."""
    common = sorted(set(a1["ids"]) & set(a2["ids"]))
    idx1 = {c: i for i, c in enumerate(a1["ids"])}
    idx2 = {c: i for i, c in enumerate(a2["ids"])}
    take1 = np.array([idx1[c] for c in common])
    take2 = np.array([idx2[c] for c in common])
    y = a1["y"][take1]
    # arm score = mean P(farm) across that arm's seeds (a seed-averaged model)
    p1 = np.mean([a1["P"][s][take1] for s in a1["P"]], axis=0)
    p2 = np.mean([a2["P"][s][take2] for s in a2["P"]], axis=0)
    delta, ci, pv = lib.paired_bootstrap_delta(lib.safe_auc, y, p1, p2, n=BOOT)
    se_boot = (ci[1] - ci[0]) / (2 * 1.96)
    n1, n2 = len(a1["per_seed"]), len(a2["per_seed"])
    se_total = float(np.sqrt(se_boot ** 2 + sigma_seed ** 2 * (1 / n1 + 1 / n2)))
    z = delta / se_total if se_total else np.nan
    from math import erfc
    p_total = float(erfc(abs(z) / np.sqrt(2))) if se_total else np.nan
    return {"n": len(common), "delta": delta, "ci_boot": list(ci), "p_boot": pv,
            "se_boot": se_boot, "se_total": se_total, "z": z, "p_total": p_total,
            "mde80": 2.80 * se_total}


def main() -> None:
    lib.header("Round_4 evaluation (see experiments/EVAL_METHODS.md)")
    SL = slices()
    print("slices:", {k: len(v) for k, v in SL.items()})

    names = [f"world_v10_fourclass_r4_{a}_s{s}" for a in ARMS for s in SEEDS] + ["v6","v7","v8","v9"]
    scores = {}
    for n in names:
        v = load_scores(n)
        if v is not None:
            scores[n] = v
    have = [a for a in ARMS if any(f"world_v10_fourclass_r4_{a}_s{s}" in scores for s in SEEDS)]
    print(f"runs loaded: {len(scores)} | arms with data: {have or 'none yet'}")
    if not have:
        print("\nNo round_4 runs collected yet -- rerun when training finishes.")
        return

    report = {}
    for sname, sl in SL.items():
        lib.header(f"slice: {sname}  (n={len(sl)}, farm rate {sl.y.mean():.2f})")

        arms = {a: arm_auc(scores, sl, a) for a in ARMS}
        arms = {a: v for a, v in arms.items() if v}

        # per-arm seed spread -> the sigma that matters for THIS slice
        print(f"{'arm':<4} {'description':<32} {'per-seed AUC':<28} {'mean':>8} {'sd':>8}")
        sigmas = []
        for a, v in arms.items():
            vals = np.array(list(v["per_seed"].values()))
            if len(vals) > 1:
                sigmas.append(vals.std(ddof=1))
            per = " ".join(f"{x:.4f}" for x in vals)
            print(f"{a:<4} {ARMS[a]:<32} {per:<28} {vals.mean():>8.4f} "
                  f"{(vals.std(ddof=1) if len(vals)>1 else float('nan')):>8.4f}")
        sigma_seed = float(np.mean(sigmas)) if sigmas else 0.0078
        print(f"\npooled sigma_seed on this slice: {sigma_seed:.4f}"
              f"   (single-run decision band 2*sqrt2*sigma = +/-{2*np.sqrt(2)*sigma_seed:.4f})")

        # archived reference models
        for arch in ("v6","v7","v8","v9"):
            if arch in scores:
                common = sorted(set(sl.index) & set(scores[arch].index))
                if len(common) > 50:
                    y = sl.loc[common,"y"].to_numpy().astype(int)
                    print(f"  archived {arch}: AUC {lib.safe_auc(y, scores[arch].loc[common].to_numpy()):.4f} (n={len(common)})")

        # contrasts
        print(f"\n{'contrast':<12} {'n':>6} {'dAUC':>9} {'SE_boot':>9} {'SE_tot':>9} "
              f"{'z':>7} {'p_raw':>8} {'MDE80':>8}")
        raw_p, rows = {}, {}
        for a1, a2 in itertools.combinations(arms, 2):
            r = compare(arms[a1], arms[a2], sigma_seed)
            key = f"{a1}_vs_{a2}"
            rows[key] = r
            raw_p[key] = r["p_total"]
            print(f"{key:<12} {r['n']:>6} {r['delta']:>+9.4f} {r['se_boot']:>9.4f} "
                  f"{r['se_total']:>9.4f} {r['z']:>+7.2f} {r['p_total']:>8.3f} {r['mde80']:>8.4f}")

        conf = {f"{x}_vs_{y_}": raw_p[f"{x}_vs_{y_}"] for x, y_ in CONFIRMATORY
                if f"{x}_vs_{y_}" in raw_p}
        if conf:
            adj = holm(conf)
            print("\nconfirmatory family (Holm-adjusted):")
            for k, p in sorted(adj.items(), key=lambda kv: kv[1]):
                d = rows[k]["delta"]
                practical = abs(d) >= 0.005
                verdict = ("BETTER" if d > 0 else "WORSE") if (p < 0.05 and practical) else (
                    "below practical floor" if p < 0.05 else "not distinguishable")
                print(f"  {k:<12} d={d:+.4f}  p_holm={p:.3f}  {verdict}")
                rows[k]["p_holm"] = p
                rows[k]["verdict"] = verdict
        report[sname] = {"sigma_seed": sigma_seed,
                         "per_arm": {a: v["per_seed"] for a, v in arms.items()},
                         "contrasts": rows}

    lib.save("r4_evaluation", report)
    print("\nsaved -> experiments/results/r4_evaluation.json")


if __name__ == "__main__":
    main()
