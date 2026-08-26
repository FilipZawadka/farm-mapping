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
        "d": "freeze0 + 6 bands", "e": "DenseNet-121 (+freeze0+6bands)", "f": "freeze5 + full-LR unfreeze"}
SEEDS = (42, 43, 44)
CLASSES = ["NotFarm", "Poultry", "Pigs", "Cattle"]
BOOT = 10_000   # EVAL_METHODS: raised from 2,000 so p-values are not floored at 2/2000

# Pre-registered confirmatory family (EVAL_METHODS section 4). Everything else
# printed by this script is exploratory and labelled as such.
CONFIRMATORY = [("b", "a"), ("d", "a"), ("e", "d"),
                # Amendment 2026-08-21 (see EVAL_METHODS "amendments"):
                ("f", "a"),   # pure LR effect  (warm-up held fixed)
                ("b", "f")]   # pure warm-up effect (backbone LR held high)


# ---------------------------------------------------------------- slices
def slices() -> dict[str, pd.DataFrame]:
    v10 = pd.read_parquet(V10)
    v10["cid"] = v10.cluster_id.astype(str)
    out = {}
    for name in ("generalization", "test", "eval"):
        d = v10[v10.cnn_split_assigned == name][["cid", "final_label"]].copy()
        # Binary farm target. "Ambiguous" means the annotator could NOT tell
        # whether it is a farm, so it belongs in neither class -- excluded, like
        # NaN. Farm: Unknown/Mixed/Other/PigsOrPoultry ARE farms (type unknown)
        # and count as positives. This mattered silently before: rows with these
        # labels had no scores (the candidates dir dropped them), so the
        # label!=NotFarm mapping never met an Ambiguous row; with full-world
        # scoring it would have counted "can't tell" as "farm".
        def _y(l):
            if pd.isna(l) or l == "Ambiguous":
                return np.nan
            return int(l != "NotFarm")
        d["y"] = d.final_label.map(_y)
        d = d.dropna(subset=["y"])
        if d.y.nunique() > 1:          # binary AUC needs both classes present
            out[name] = d.set_index("cid")
        else:
            print(f"  ! slice {name} dropped: only one class present")
    return out


def load_scores(run: str) -> pd.Series | None:
    """P(farm) by candidate id for a run name, or None if not collected."""
    # Prefer the full-world scoring pass: it covers rows the training run's
    # candidates dir drops (unscorable labels), which the slices need.
    p = GPU / f"{run}_score" / "scored_candidates.parquet"
    if not p.exists():
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
    delta_ens, ci, pv = lib.paired_bootstrap_delta(lib.safe_auc, y, p1, p2, n=BOOT)
    se_boot = (ci[1] - ci[0]) / (2 * 1.96)
    n1, n2 = len(a1["per_seed"]), len(a2["per_seed"])

    # Two DIFFERENT estimands; do not conflate them (see EVAL_METHODS "estimands"):
    #  delta_ens = AUC(seed-averaged probability) -- compares two trained ARTIFACTS
    #              (a 3-seed ensemble each). It saturates near the ceiling, so it
    #              systematically COMPRESSES recipe differences.
    #  delta_rec = difference in mean per-seed AUC -- compares two RECIPES, which is
    #              the question every arm here was designed to answer. PRIMARY.
    m1 = float(np.mean(list(a1["per_seed"].values())))
    m2 = float(np.mean(list(a2["per_seed"].values())))
    delta_rec = m2 - m1

    # se_boot (row-sampling noise) is taken from the ensemble bootstrap as a proxy for
    # the mean-per-seed statistic. Immaterial in practice: the seed term dominates it
    # by ~5x, and both arms share rows so row noise largely cancels.
    se_total = float(np.sqrt(se_boot ** 2 + sigma_seed ** 2 * (1 / n1 + 1 / n2)))
    z = delta_rec / se_total if se_total else np.nan
    from math import erfc
    p_total = float(erfc(abs(z) / np.sqrt(2))) if se_total else np.nan
    return {"n": len(common), "delta": delta_rec, "delta_ens": delta_ens,
            "ci_boot": list(ci), "p_boot": pv,
            "se_boot": se_boot, "se_total": se_total, "z": z, "p_total": p_total,
            "mde80": 2.80 * se_total}


# ------------------------------------------------- secondary reporting
def _country_map() -> "pd.Series":
    """cid -> country. v10 carries ADM0 (ISO3), not a `country` column."""
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(V10).schema.names)
    key = "ADM0" if "ADM0" in cols else ("country" if "country" in cols else None)
    if key is None:
        return pd.Series(dtype=object)
    v10 = pd.read_parquet(V10, columns=["cluster_id", key])
    v10["cid"] = v10.cluster_id.astype(str)
    return v10.drop_duplicates("cid").set_index("cid")[key]


def calibration_table(arms: dict, sl: pd.DataFrame) -> None:
    """ECE of the farm probability per arm (Guo et al. 2017).

    Reported separately from AUC because they answer different questions: AUC is
    ranking quality (threshold-free), ECE is whether the probability can be read as
    a probability. A recipe can win on AUC and still be the worse deployment choice
    if its scores are miscalibrated at the operating threshold.
    """
    print(f"\n{'arm':<4} {'ECE (mean over seeds)':<24} {'per-seed ECE'}")
    for a, v in arms.items():
        y = v["y"]
        eces = [lib.ece(y, v["P"][sd]) for sd in v["P"]]
        per = " ".join(f"{x:.4f}" for x in eces)
        print(f"{a:<4} {np.mean(eces):<24.4f} {per}")


def per_country_table(arms: dict, sl: pd.DataFrame, cmap, min_n: int = 20) -> dict:
    """Seed-averaged farm AUC by country -- catches an arm that wins overall while
    regressing somewhere specific, which a pooled metric hides."""
    out = {}
    countries = cmap.reindex([c for c in next(iter(arms.values()))["ids"]]).fillna("?")
    names = sorted({c for c in countries.unique()
                    if (countries == c).sum() >= min_n and c != "?"})
    if not names:
        return out
    print(f"\n{'country':<10} {'n':>5} " + " ".join(f"{a:>8}" for a in arms))
    for cn in names:
        mask = (countries == cn).to_numpy()
        row = {}
        cells = []
        for a, v in arms.items():
            y = v["y"][mask]
            if len(np.unique(y)) < 2:
                cells.append(f"{'--':>8}"); continue
            p = np.mean([v["P"][sd][mask] for sd in v["P"]], axis=0)
            auc = lib.safe_auc(y, p)
            row[a] = auc
            cells.append(f"{auc:>8.4f}")
        if row:
            out[cn] = row
            print(f"{cn:<10} {int(mask.sum()):>5} " + " ".join(cells))
    return out


def class_table(runs_present: list) -> dict:
    """4-class macro-F1 and Cattle-excluded mean F1 from the collected metrics files.

    Cattle is ~6-7 rows in these slices and was shown to drive ~80% of seed variance,
    so the Cattle-excluded figure is the stable read; both are reported.
    """
    import json
    out = {}
    for slice_name, fname in (("eval", "eval_metrics.json"),
                              ("generalization", "generalization_metrics.json")):
        rows = {}
        for run in runs_present:
            f = GPU / run / fname
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            per_class = [d.get(f"f1_class{i}") for i in range(4)]
            have = [x for x in per_class if isinstance(x, (int, float))]
            no_cattle = [x for i, x in enumerate(per_class[:3])
                         if isinstance(x, (int, float))]
            rows[run] = {"macro_f1": float(np.mean(have)) if have else float("nan"),
                         "f1_excl_cattle": float(np.mean(no_cattle)) if no_cattle else float("nan"),
                         "per_class": per_class}
        if not rows:
            continue
        out[slice_name] = rows
        lib.header(f"per-class F1 -- slice {slice_name}")
        print(f"{'run':<34} {'macro-F1':>9} {'excl-Cattle':>12}   per-class [NotFarm,Poultry,Pigs,Cattle]")
        for run, r in sorted(rows.items()):
            pc = ", ".join("--" if x is None else f"{x:.3f}" for x in r["per_class"])
            print(f"{run:<34} {r['macro_f1']:>9.4f} {r['f1_excl_cattle']:>12.4f}   [{pc}]")
    return out


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

    CMAP = _country_map()
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

        # Archived reference models, scored on the SAME rows as the arms.
        # Previously these used slice-intersect-archived (n=662) while the arms used
        # rows common to all their seeds (n=617), so the series table compared
        # different denominators. Align to the arms' row set whenever arms exist.
        arm_common = None
        for v in arms.values():
            arm_common = set(v["ids"]) if arm_common is None else (arm_common & set(v["ids"]))
        for arch in ("v6","v7","v8","v9"):
            if arch in scores:
                base = set(sl.index) if arm_common is None else arm_common
                common = sorted(base & set(scores[arch].index))
                if len(common) > 50:
                    y = sl.loc[common,"y"].to_numpy().astype(int)
                    tag = "" if arm_common is None else " [arm rows]"
                    print(f"  archived {arch}: AUC "
                          f"{lib.safe_auc(y, scores[arch].loc[common].to_numpy()):.4f} "
                          f"(n={len(common)}){tag}")

        # contrasts
        print("\ndelta = second arm minus first (positive => second arm better)")
        print(f"\n{'contrast':<12} {'n':>6} {'dAUC_rec':>9} {'dAUC_ens':>9} {'SE_tot':>9} "
              f"{'z':>7} {'p_raw':>8} {'MDE80':>8}")
        raw_p, rows = {}, {}
        for a1, a2 in itertools.combinations(arms, 2):
            r = compare(arms[a1], arms[a2], sigma_seed)
            key = f"{a1}_vs_{a2}"
            rows[key] = r
            raw_p[key] = r["p_total"]
            print(f"{key:<12} {r['n']:>6} {r['delta']:>+9.4f} {r['delta_ens']:>+9.4f} "
                  f"{r['se_total']:>9.4f} {r['z']:>+7.2f} {r['p_total']:>8.3f} {r['mde80']:>8.4f}")

        # rows are keyed by itertools.combinations order ("a_vs_b"), where delta is
        # oriented second-minus-first. CONFIRMATORY is written as (better?, baseline),
        # so resolve the key in whichever order it exists and flip the sign to match.
        conf, orient = {}, {}
        for x, y_ in CONFIRMATORY:
            if f"{y_}_vs_{x}" in raw_p:          # delta already = x - y_
                conf[f"{x}>{y_}"] = raw_p[f"{y_}_vs_{x}"]; orient[f"{x}>{y_}"] = (f"{y_}_vs_{x}", +1.0)
            elif f"{x}_vs_{y_}" in raw_p:        # delta = y_ - x, needs flipping
                conf[f"{x}>{y_}"] = raw_p[f"{x}_vs_{y_}"]; orient[f"{x}>{y_}"] = (f"{x}_vs_{y_}", -1.0)
        if not conf and rows:
            print("\n  ! confirmatory family empty -- CONFIRMATORY names no comparable arm pair")
        if conf:
            adj = holm(conf)
            print("\nconfirmatory family (Holm-adjusted):")
            for k, p in sorted(adj.items(), key=lambda kv: kv[1]):
                src, sign = orient[k]
                d = sign * rows[src]["delta"]
                practical = abs(d) >= 0.005
                better, base = k.split(">")
                verdict = (f"{better} BETTER than {base}" if d > 0 else
                           f"{better} WORSE than {base}") if (p < 0.05 and practical) else (
                    "significant but below practical floor" if p < 0.05 else "not distinguishable")
                print(f"  {k:<10} d={d:+.4f}  p_holm={p:.3f}  {verdict}")
                rows[src]["p_holm"] = p
                rows[src]["verdict"] = verdict
        lib.header(f"calibration (ECE) -- slice {sname}")
        calibration_table(arms, sl)

        lib.header(f"per-country farm AUC (n>=20) -- slice {sname}")
        pc = per_country_table(arms, sl, CMAP)

        report[sname] = {"sigma_seed": sigma_seed,
                         "per_arm": {a: v["per_seed"] for a, v in arms.items()},
                         "contrasts": rows,
                         "per_country": pc}

    report["per_class"] = class_table([n for n in scores if n.startswith("world_")])

    lib.save("r4_evaluation", report)
    print("\nsaved -> experiments/results/r4_evaluation.json")


if __name__ == "__main__":
    main()
