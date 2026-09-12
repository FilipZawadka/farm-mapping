"""Rachel-style per-country threshold metrics for every collected run.

For each run this computes the report in ``training/country_metrics.py`` (a
port of Rachel Mason's CAFO-AI_v2 evaluation: precision/recall vs threshold
per country, one global threshold maximising the unweighted mean per-country
F1, P/R/F1 per country at that threshold, FP/FN rates, poultry-gate confusion,
headline counts) and then lines the runs up side by side, arm by arm.

Labels and splits
-----------------
Rachel re-delivers the label files between rounds, so a run's own embedded
labels are not the latest ones and different rounds' held-out slices are not
the same rows. By default every run is scored against ONE reference file
(``--labels v11``, the round_5 delivery, which carries the 2026-09 label
fixes and the new IDN/MOZ/PER generalization labels) and any row that ANY
evaluated run trained on is blanked (its label is dropped, so it still counts
as a candidate but never as a labelled one). Every model is therefore judged
on identical rows none of them saw -- the same idea as her
``hack_gen_country_info`` overlay, made leakage-proof. ``--own-splits`` turns
this off and scores each run on its own embedded labels instead.

Outputs
-------
``experiments/results/country_metrics/<run>/country_threshold_metrics.json``
and her figures per run; ``summary.json`` / ``summary.md`` and two cross-run
figures at the top level.

Run:  python3 experiments/evaluate_country_metrics.py [--no-plots] [--labels v10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lib  # noqa: E402
import evaluate_r4  # noqa: E402
from training import country_metrics as cm  # noqa: E402

GPU = HERE / "gpu_results"
OUT = lib.RESULTS / "country_metrics"
LABEL_FILES = {
    "v10": lib.REPO / "data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet",  # round_4
    "v11": lib.REPO / "data/rachel_geometry_candidates/all_countries/all_clusters_v11.parquet",  # round_5
}
ARM_DESC = {
    **{f"r4_{k}": v for k, v in evaluate_r4.ARMS.items()},
    # round_5 (docs/COUNTRY_BALANCING_PLAN.md); kept in step with evaluate_balancing.ARMS
    "r5_a": "r5 baseline: v11 labels, no sampler",
    "r5_g": "grouped countries, uniform, class-cond.",
    "r5_h": "3 buckets us/europe/rest, class-cond.",
    "r5_i": "per-country cap 20%, class-cond.",
    "r5_j": "ABLATION grouped, no class-cond.",
    "archived_v6": "archived v6 (round_1 labels)",
    "archived_v9": "archived v9 (production; round_3 labels)",
}
RUN_RE = re.compile(r"_(r\d)_([a-z])_s(\d+)$")


# ----------------------------------------------------------------- discovery
def discover_runs(include_archived=("v6", "v9")) -> dict[str, Path]:
    runs = {}
    for k in include_archived:
        p = lib.CACHE / lib.FOURCLASS[k] / "scored_candidates.parquet"
        if p.exists():
            runs[k] = p
    for d in sorted(GPU.glob("world_v10_fourclass_r[0-9]_*")):
        if d.name.endswith("_score") or not (d / "scored_candidates.parquet").exists():
            continue
        full = GPU / f"{d.name}_score" / "scored_candidates.parquet"
        runs[d.name] = full if full.exists() else d / "scored_candidates.parquet"
    return runs


def run_key(name: str) -> tuple[str, str, int | None]:
    m = RUN_RE.search(name)
    if m:
        return m.group(1), f"{m.group(1)}_{m.group(2)}", int(m.group(3))
    return "archived", f"archived_{name}", None


def class_names_for(name: str) -> list[str]:
    cfg = GPU / name / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            names = (yaml.safe_load(cfg.read_text()).get("model") or {}).get("class_names")
            if names:
                return list(names)
        except Exception:
            pass
    return list(cm.DEFAULT_CLASS_NAMES)


def load_scores(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    if "geometry" in d.columns:
        d = d.drop(columns=["geometry"])
    d["cluster_id"] = d["cluster_id"].astype(str)
    d = d.drop_duplicates("cluster_id")
    own = d["cnn_split_assigned"].astype(str).str.strip() if "cnn_split_assigned" in d else pd.Series("", index=d.index)
    fallback = d["split"].astype(str).str.strip() if "split" in d else pd.Series("unknown", index=d.index)
    d["own_split"] = own.where(~own.isin(("", "nan", "None")), fallback)
    keep = ["cluster_id", "own_split"] + [c for c in d.columns if c.startswith("prob_class")]
    return d[keep]


# ----------------------------------------------------------------- helpers
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def mean_sd(vals, nd=3):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} ± {np.std(vals, ddof=1):.{nd}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="v11", choices=sorted(LABEL_FILES))
    ap.add_argument("--own-splits", action="store_true", help="score each run on its own embedded labels")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--runs", nargs="*", help="substrings; default = every collected round run + archived v6/v9")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    if args.runs:
        runs = {k: v for k, v in runs.items() if any(s in k for s in args.runs)}
    if not runs:
        sys.exit("no runs found")
    lib.header(f"Per-country threshold metrics (Rachel-style) -- {len(runs)} runs, labels={args.labels}"
               f"{' (own splits)' if args.own_splits else ' (common clean rows)'}")

    ref_cols = ["cluster_id", "ADM0", "final_label", "cnn_split_assigned"]
    ref = pd.read_parquet(LABEL_FILES[args.labels], columns=ref_cols)
    ref["cluster_id"] = ref["cluster_id"].astype(str)
    ref = ref.drop_duplicates("cluster_id")

    scores = {name: load_scores(p) for name, p in runs.items()}
    trained = set()
    if not args.own_splits:
        for s in scores.values():
            trained |= set(s.loc[s["own_split"].isin(cm.TRAIN_SPLITS), "cluster_id"])
        lab = ref[ref["final_label"].notna()]
        blanked = lab[lab["cluster_id"].isin(trained)]["cnn_split_assigned"].value_counts()
        kept = lab[~lab["cluster_id"].isin(trained)]["cnn_split_assigned"].value_counts()
        print("\nreference labelled rows blanked because some evaluated run trained on them "
              "(remaining in brackets):")
        for sp in ("eval", "generalization", "test", "qual_eval", "train", "val"):
            print(f"  {sp:<15} {int(blanked.get(sp, 0)):>6}  [{int(kept.get(sp, 0))} kept]")

    reports, flats, prepared = {}, {}, {}
    for name, sc in scores.items():
        campaign, arm, seed = run_key(name)
        names = class_names_for(name)
        if args.own_splits:
            df = pd.read_parquet(runs[name])
            if "geometry" in df.columns:
                df = df.drop(columns=["geometry"])
        else:
            df = ref.merge(sc, on="cluster_id", how="inner")
            df.loc[df["cluster_id"].isin(trained), "final_label"] = None
        try:
            report = cm.full_report(df, class_names=names)
        except Exception as exc:  # a broken run must not stop the sweep
            print(f"  ! {name}: {exc}")
            continue
        report["meta"].update({"run": name, "campaign": campaign, "arm": arm, "seed": seed,
                               "arm_description": ARM_DESC.get(arm, ""), "labels": args.labels,
                               "own_splits": args.own_splits, "scores_from": str(runs[name])})
        rdir = out / name
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "country_threshold_metrics.json").write_text(json.dumps(report, indent=1, default=cm._jsonable))
        reports[name] = report
        flat = cm.flat_summary(report)
        flat.update({"run": name, "campaign": campaign, "arm": arm, "seed": seed})
        flats[name] = flat
        farm, poul = report["tasks"].get("farm"), report["tasks"].get("poultry")
        print(f"  {name:<34} farm t={fmt(farm['threshold']['mean_f1'])} F1={fmt(farm['summary']['mean_f1'])} "
              f"loco={fmt(farm['summary_loco'].get('mean_f1'))}"
              + (f" | poultry t={fmt(poul['threshold']['mean_f1'])} F1={fmt(poul['summary']['mean_f1'])}" if poul else ""))
        d = cm.prepare(df, names)
        prepared[name] = d[["cluster_id", "truth", "split", "prob_Farm"]].assign(heldout=cm.heldout_mask(d))
        if not args.no_plots:
            import country_metric_plots as cmp
            cmp.render_run(report, d, rdir, name, heldout_mask=cm.heldout_mask(d))

    if not reports:
        sys.exit("nothing evaluated")

    # ------------------------------------------------------------ cross-run
    order = sorted(flats, key=lambda n: (flats[n]["campaign"] != "archived", flats[n]["arm"], flats[n]["seed"] or 0))
    arms = []
    for n in order:
        if flats[n]["arm"] not in arms:
            arms.append(flats[n]["arm"])
    tasks = [t for t in cm.TASKS if any(f"{t}_mean_f1" in flats[n] for n in order)]
    lines = [f"# Per-country threshold metrics (Rachel-style) -- {len(order)} runs",
             "",
             f"Reference labels: `{LABEL_FILES[args.labels].name}`"
             + (" (each run on its own embedded labels)" if args.own_splits else
                f"; {len(trained):,} candidate ids trained on by at least one run are blanked for everyone."),
             "Threshold per run = argmax over a 200-point grid of the unweighted mean per-country F1 "
             "(focal countries on `eval`, generalization countries on `generalization`; IDN/MOZ/PER excluded). "
             "`loco` = leave-one-country-out threshold (each country scored at the threshold tuned on the others). "
             "`@0.4` = mean per-country F1 at the shipped OOD operating point.",
             ""]

    for task in tasks:
        lines += [f"## {task}: per run", ""]
        hdr = ["run", "arm", "seed", "t (mean-F1)", "plateau ±0.01", "t (pooled)", "mean F1", "mean P", "mean R",
               "loco F1", "@0.4", "@0.75", "min-F1 country", "held-out FP rate", "held-out FN rate"]
        rows = []
        for n in order:
            f = flats[n]
            if f"{task}_mean_f1" not in f:
                continue
            span = f.get(f"{task}_plateau_0.01")
            rows.append([n, f["arm"], f["seed"] if f["seed"] is not None else "--",
                         fmt(f[f"{task}_t_mean"]), f"{span[0]:.2f}–{span[1]:.2f}" if span else "--",
                         fmt(f[f"{task}_t_pooled"]), fmt(f[f"{task}_mean_f1"]),
                         fmt(f[f"{task}_mean_precision"]), fmt(f[f"{task}_mean_recall"]),
                         fmt(f.get(f"{task}_mean_f1_loco")), fmt(f.get(f"{task}_mean_f1_at_0.4")),
                         fmt(f.get(f"{task}_mean_f1_at_0.75")),
                         f"{f[f'{task}_min_f1_country']} ({fmt(f[f'{task}_min_f1'], 2)})",
                         fmt(f.get(f"{task}_heldout_FP_rate")), fmt(f.get(f"{task}_heldout_FN_rate"))])
        lines += [md_table(hdr, rows), ""]

        lines += [f"## {task}: per arm (mean ± sd over seeds)", ""]
        hdr = ["arm", "description", "seeds", "t (mean-F1)", "mean F1", "loco F1", "@0.4", "mean P", "mean R",
               "t (P≥0.8)", "F1 (P≥0.8)", "R (P≥0.8)"]
        rows = []
        for arm in arms:
            members = [flats[n] for n in order if flats[n]["arm"] == arm and f"{task}_mean_f1" in flats[n]]
            if not members:
                continue
            rows.append([arm, ARM_DESC.get(arm, ""), len(members),
                         mean_sd([m[f"{task}_t_mean"] for m in members]),
                         mean_sd([m[f"{task}_mean_f1"] for m in members]),
                         mean_sd([m.get(f"{task}_mean_f1_loco") for m in members]),
                         mean_sd([m.get(f"{task}_mean_f1_at_0.4") for m in members]),
                         mean_sd([m[f"{task}_mean_precision"] for m in members]),
                         mean_sd([m[f"{task}_mean_recall"] for m in members]),
                         mean_sd([m.get(f"{task}_t_pfloor") for m in members])
                         + ("" if all(m.get(f"{task}_pfloor_met", True) for m in members) else " (floor not met)"),
                         mean_sd([m.get(f"{task}_mean_f1_pfloor") for m in members]),
                         mean_sd([m.get(f"{task}_mean_recall_pfloor") for m in members])])
        lines += [md_table(hdr, rows), ""]

        countries = []
        for n in order:
            for c in flats[n].get(f"{task}_f1_by_country", {}):
                if c not in countries:
                    countries.append(c)
        for metric in ("f1", "precision", "recall"):
            lines += [f"## {task}: per-country {metric} by arm (at each run's own threshold; mean ± sd over seeds)", ""]
            hdr = ["arm"] + countries
            rows = []
            for arm in arms:
                members = [flats[n] for n in order if flats[n]["arm"] == arm and f"{task}_{metric}_by_country" in flats[n]]
                if not members:
                    continue
                rows.append([arm] + [mean_sd([m[f"{task}_{metric}_by_country"].get(c) for m in members], 2)
                                     for c in countries])
            lines += [md_table(hdr, rows), ""]

    # Her 06_compare_models.ipynb: what changes on the map when one model
    # replaces another -- farms caught by both, lost, rescued -- on labelled
    # held-out rows and on the unlabelled rest of the world.
    ref_name = next((n for n in order if flats[n]["arm"] == "archived_v9"), None)
    if ref_name and len(order) > 1:
        for mode in ("own", "0.4"):
            lines += [f"## Farm flags vs archived v9 ({'each model at its own mean-F1 threshold' if mode == 'own' else 'both models at t = 0.4'})",
                      "",
                      "Per run: on held-out labelled farms, how many the reference caught, how many this run catches, "
                      "and the exchange (lost = caught by v9 only, rescued = caught by this run only); "
                      "on held-out NotFarm, how many each flags; on the unlabelled rest of the world, the flag "
                      "counts, the exchange and the Jaccard overlap of the two flagged sets.",
                      ""]
            hdr = ["run", "t v9 / t run", "farms n", "caught v9", "caught run", "lost", "rescued",
                   "NotFarm n", "flagged v9", "flagged run", "unlab. flagged v9", "unlab. flagged run",
                   "unlab. lost", "unlab. rescued", "unlab. Jaccard"]
            rows = []
            a_all = prepared[ref_name].set_index("cluster_id")
            t_ref = flats[ref_name]["farm_t_mean"] if mode == "own" else 0.4
            for n in order:
                if n == ref_name:
                    continue
                b_all = prepared[n].set_index("cluster_id")
                common = a_all.index.intersection(b_all.index)
                a, b = a_all.loc[common], b_all.loc[common]
                t_run = flats[n]["farm_t_mean"] if mode == "own" else 0.4
                fa, fb = a["prob_Farm"] >= t_ref, b["prob_Farm"] >= t_run
                farms = a["heldout"] & a["truth"].isin(cm.FARM_TRUTH)
                notf = a["heldout"] & (a["truth"] == "NotFarm")
                unl = a["truth"].isna()
                inter, union = int((fa & fb & unl).sum()), int(((fa | fb) & unl).sum())
                rows.append([n, f"{t_ref:.3f} / {t_run:.3f}", int(farms.sum()), int((fa & farms).sum()),
                             int((fb & farms).sum()), int((fa & ~fb & farms).sum()), int((~fa & fb & farms).sum()),
                             int(notf.sum()), int((fa & notf).sum()), int((fb & notf).sum()),
                             int((fa & unl).sum()), int((fb & unl).sum()), int((fa & ~fb & unl).sum()),
                             int((~fa & fb & unl).sum()), f"{inter / union:.2f}" if union else "--"])
                flats[n].setdefault("vs_v9", {})[mode] = dict(zip(hdr[1:], rows[-1][1:]))
            lines += [md_table(hdr, rows), ""]

    lines += ["## Headline counts (full world, at each run's own thresholds)", ""]
    hdr = ["run", "farm t", "n farm ≥ t", "unlabelled farm ≥ t", "poultry t", "n poultry ≥ t", "unlabelled poultry ≥ t"]
    rows = []
    for n in order:
        f = flats[n]
        rows.append([n, fmt(f.get("farm_threshold")), f.get("n_farm_above", "--"), f.get("n_unlabelled_farm_above", "--"),
                     fmt(f.get("poultry_threshold")), f.get("n_poultry_above", "--"), f.get("n_unlabelled_poultry_above", "--")])
    lines += [md_table(hdr, rows), ""]

    # eval-set composition once (same reference rows for every run)
    any_rep = reports[order[0]]
    lines += ["## Evaluation-set composition (reference rows after blanking)", ""]
    hdr = ["country", "split", "labelled", "ambiguous", "farm-task rows", "farm positives", "generic farm",
           "poultry-task rows", "poultry positives"]
    rows = [[r["country"], r["split"], r["n_labelled"], r["n_ambiguous"], r["n_farm_task"], r["n_farm_pos"],
             r["n_generic_farm"], r["n_poultry_task"], r["n_poultry_pos"]] for r in any_rep["eval_sets"]["composition"]]
    lines += [md_table(hdr, rows), ""]

    summary_md = "\n".join(lines)
    (out / "summary.md").write_text(summary_md)
    (out / "summary.json").write_text(json.dumps(
        {"labels": args.labels, "own_splits": args.own_splits, "n_trained_blanked": len(trained),
         "runs": {n: flats[n] for n in order}}, indent=1, default=cm._jsonable))
    print("\n" + summary_md)

    if not args.no_plots:
        import country_metric_plots as cmp
        for task in tasks:
            curves = {}
            for arm in arms:
                members = [reports[n] for n in order if flats[n]["arm"] == arm and task in reports[n]["tasks"]]
                if not members:
                    continue
                grid = members[0]["tasks"][task]["mean_f1_curve"]["grid"]
                ys = np.mean([m["tasks"][task]["mean_f1_curve"]["mean_f1"] for m in members], axis=0)
                curves[f"{arm} (n={len(members)})"] = {"grid": grid, "mean_f1": ys,
                                                       "lw": 2.2 if arm.startswith("archived") else 1.5,
                                                       "alpha": 1.0}
            cmp.plot_mean_f1_curves(curves, out / f"mean_f1_vs_threshold_{task}.png",
                                    title=f"{task}: mean per-country F1 vs threshold, seed-averaged per arm")
            points = [{"arm": flats[n]["arm"], "seed": flats[n]["seed"], "country": c, "f1": v}
                      for n in order for c, v in flats[n].get(f"{task}_f1_by_country", {}).items()
                      if not flats[n]["arm"].startswith("archived")]
            ref_arm = next((n for n in order if flats[n]["arm"] == "archived_v9"), None)
            ref_f1 = flats[ref_arm].get(f"{task}_f1_by_country") if ref_arm else None
            cmp.plot_country_f1_by_arm(points, out / f"country_f1_by_arm_{task}.png",
                                       title=f"{task}: per-country F1 by arm (points = seeds, dashed = archived v9)",
                                       arms=[a for a in arms if not a.startswith("archived")], ref=ref_f1)
    print(f"\nsaved -> {out}/summary.md, summary.json, per-run folders")


if __name__ == "__main__":
    main()
