"""Judge every GPU experiment against seed noise, on the frozen blind benchmark.

The point of E0.3 is that a delta is only meaningful relative to run-to-run
variance. This script:
  1. computes seed sigma from the E0.3 runs (seeds 42-46, 42 = production),
  2. scores every other run on the frozen 11,365-row blind benchmark,
  3. reports each lever's delta as a multiple of sigma, plus a paired bootstrap CI.

Primary metric is binary farm ROC-AUC: it is threshold-free and comparable across
runs with different taxonomies (the 3-class run included). Four-class macro-F1 is
reported alongside where the taxonomy allows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import lib

GPU = Path(__file__).resolve().parent / "gpu_results"
FROZEN = lib.RESULTS / "e01_blind_benchmark_frozen.csv"

SEED_RUNS = {42: None, 43: "e03_seed43", 44: "e03_seed44", 45: "e03_seed45", 46: "e03_seed46"}
EXPERIMENT_OF = {
    "e12_crop128": "E1.2", "e12_crop48": "E1.2",
    "e11_rgb": "E1.1", "e11_rgb_nir": "E1.1", "e11_rgb_ndwi": "E1.1",
    "e11_6bands": "E1.1", "e11_recompute_idx": "E1.1",
    "e16_ssl4eo": "E1.6", "e17_val_loss": "E1.7", "e19_three_class": "E1.9",
    "e15_cutout_only": "E1.5", "e15_no_photometric": "E1.5", "e15_geometric_only": "E1.5",
    "e14_lr3e-5": "E1.4", "e14_lr3e-4": "E1.4", "e14_freeze0": "E1.4",
}
DESC = {
    "e12_crop128": "context 128 px", "e12_crop48": "context 48 px",
    "e11_rgb": "RGB only", "e11_rgb_nir": "RGB+NIR", "e11_rgb_ndwi": "RGB+NDWI",
    "e11_6bands": "6 bands, no indices", "e11_recompute_idx": "recompute indices",
    "e16_ssl4eo": "SSL4EO backbone", "e17_val_loss": "checkpoint on val_loss",
    "e19_three_class": "3-class taxonomy", "e15_cutout_only": "cutout only",
    "e15_no_photometric": "no photometric augs", "e15_geometric_only": "geometric augs only",
    "e14_lr3e-5": "lr 3e-5", "e14_lr3e-4": "lr 3e-4", "e14_freeze0": "no freeze phase",
}


def load_run(name: str | None) -> pd.DataFrame | None:
    """Scored rows for a run; None means the archived production baseline."""
    if name is None:
        return lib.labeled(lib.load(lib.FOURCLASS["v9"]))
    p = GPU / name / "scored_candidates.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    return lib.labeled(df)


def on_frozen(df: pd.DataFrame, ids: pd.Index) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    d = df.drop_duplicates("candidate_id").set_index("candidate_id").reindex(ids)
    if d["prob_class0"].isna().mean() > 0.02:
        return None
    d = d[d["prob_class0"].notna()]
    y = (d["true_label"].to_numpy().astype(int) != 0).astype(int)
    p = 1.0 - d["prob_class0"].to_numpy(dtype=float)
    return y, p, d


def main() -> None:
    lib.header("GPU experiment results on the frozen blind benchmark")
    frozen = pd.read_csv(FROZEN)
    ids = pd.Index(frozen["candidate_id"])
    print(f"frozen benchmark: {len(ids):,} rows, {frozen.country.nunique()} countries")

    # ---------------------------------------------------------------- E0.3 sigma
    lib.header("E0.3  Seed variance")
    seed_scores, seed_rows = {}, []
    for seed, run in SEED_RUNS.items():
        df = load_run(run)
        if df is None:
            print(f"  seed {seed}: MISSING ({run})")
            continue
        got = on_frozen(df, ids)
        if got is None:
            print(f"  seed {seed}: incomplete coverage on frozen slice")
            continue
        y, p, d = got
        auc = lib.safe_auc(y, p)
        mf = float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0))
        seed_scores[seed] = {"auc": auc, "macro_f1": mf, "y": y, "p": p, "n": len(y)}
        seed_rows.append({"seed": seed, "run": run or "production(v9)", "n": len(y),
                          "auc": auc, "macro_f1": mf})
        print(f"  seed {seed:<3} {(run or 'production(v9)'):<16} n={len(y):<6} "
              f"AUC={auc:.4f}  macroF1={mf:.4f}")

    sigma_auc = sigma_f1 = None
    if len(seed_scores) >= 3:
        aucs = np.array([v["auc"] for v in seed_scores.values()])
        f1s = np.array([v["macro_f1"] for v in seed_scores.values()])
        sigma_auc, sigma_f1 = float(aucs.std(ddof=1)), float(f1s.std(ddof=1))
        print(f"\n  n_seeds={len(aucs)}   AUC: mean={aucs.mean():.4f} sigma={sigma_auc:.4f} "
              f"range=[{aucs.min():.4f}, {aucs.max():.4f}]")
        print(f"  {'':>13}macro-F1: mean={f1s.mean():.4f} sigma={sigma_f1:.4f} "
              f"range=[{f1s.min():.4f}, {f1s.max():.4f}]")
        print(f"\n  => 2-sigma decision band: AUC +/-{2*sigma_auc:.4f}, macro-F1 +/-{2*sigma_f1:.4f}")
    else:
        print("\n  not enough seed runs yet to estimate sigma")

    # ------------------------------------------------------------- lever deltas
    lib.header("Levers vs production baseline (seed 42)")
    base = seed_scores.get(42)
    results = []
    if base is None:
        print("  baseline missing -- cannot compute deltas")
    else:
        baseline_idx = (load_run(None).drop_duplicates("candidate_id")
                        .set_index("candidate_id"))
        print(f"{'run':<22} {'exp':<6} {'AUC':>8} {'dAUC':>9} {'d/sigma':>9} "
              f"{'95% CI':>20} {'verdict':>12}")
        print("-" * 92)
        for name in EXPERIMENT_OF:
            df = load_run(name)
            if df is None:
                print(f"{name:<22} {EXPERIMENT_OF[name]:<6} {'--- not finished ---':>50}")
                continue
            got = on_frozen(df, ids)
            if got is None:
                print(f"{name:<22} {EXPERIMENT_OF[name]:<6} {'--- incomplete coverage ---':>50}")
                continue
            y, p, d = got
            # Compare on exactly the rows this run scored, so a run with slightly
            # different coverage is not penalised for the rows it is missing.
            b = baseline_idx.reindex(d.index)
            yb = (b["true_label"].to_numpy().astype(int) != 0).astype(int)
            pb = 1.0 - b["prob_class0"].to_numpy(dtype=float)
            auc = lib.safe_auc(y, p)
            delta, ci, pval = lib.paired_bootstrap_delta(lib.safe_auc, yb, pb, p)
            nsig = delta / sigma_auc if sigma_auc else float("nan")
            if sigma_auc and abs(delta) < 2 * sigma_auc:
                verdict = "within noise"
            elif delta > 0:
                verdict = "BETTER"
            else:
                verdict = "WORSE"
            print(f"{name:<22} {EXPERIMENT_OF[name]:<6} {auc:8.4f} {delta:+9.4f} "
                  f"{nsig:+9.2f} [{ci[0]:+.4f},{ci[1]:+.4f}] {verdict:>12}")
            results.append({"run": name, "experiment": EXPERIMENT_OF[name], "desc": DESC.get(name),
                            "n": len(y), "auc": auc, "delta_auc": delta, "ci": list(ci),
                            "p_value": pval, "delta_over_sigma": nsig, "verdict": verdict})

    # --------------------------------- which class drives the macro-F1 variance
    # Macro-F1 averages classes equally, so a class with single-digit support can
    # swing the headline number far more than the model's behaviour actually moved.
    lib.header("E0.3  Per-class F1 spread across seeds (source of macro-F1 noise)")
    seed_files = {42: lib.CACHE / "world_v10_fourclass_v9", **{
        s: GPU / r for s, r in SEED_RUNS.items() if r}}
    for slice_file, slice_name in [("eval_metrics.json", "eval"),
                                   ("qual_eval_metrics.json", "qual_eval"),
                                   ("training_metrics.json", "test")]:
        per_class: dict[str, list[float]] = {}
        macro: list[float] = []
        for seed, d in sorted(seed_files.items()):
            p = d / slice_file
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            macro.append(j.get("f1"))
            for i, cname in enumerate(lib.CLASS_NAMES):
                v = j.get(f"f1_class{i}")
                if v is not None:
                    per_class.setdefault(cname, []).append(v)
        if len(macro) < 2:
            continue
        print(f"\n  {slice_name} (n_seeds={len(macro)}):")
        print(f"    {'macro-F1':<10} range {min(macro):.4f}-{max(macro):.4f}  "
              f"spread {max(macro)-min(macro):.4f}")
        for cname, vals in per_class.items():
            if len(vals) < 2:
                continue
            print(f"    {cname:<10} range {min(vals):.4f}-{max(vals):.4f}  "
                  f"spread {max(vals)-min(vals):.4f}")

    # ------------------------------------------- per-slice macro-F1 from metrics
    # The frozen benchmark answers farm-vs-not; the held-out slices are where
    # minority-type performance shows up, so report both.
    lib.header("Per-slice macro-F1 (from each run's own metrics files)")
    slices = [("training_metrics.json", "test"), ("eval_metrics.json", "eval"),
              ("generalization_metrics.json", "gen"), ("qual_eval_metrics.json", "qual_eval")]
    print(f"{'run':<22} " + " ".join(f"{s:>10}" for _, s in slices))
    print("-" * 68)
    per_slice = []
    for name in [r for r in SEED_RUNS.values() if r] + list(EXPERIMENT_OF):
        row = {"run": name}
        cells = []
        for fname, key in slices:
            p = GPU / name / fname
            if p.exists():
                try:
                    row[key] = json.loads(p.read_text()).get("f1")
                except json.JSONDecodeError:
                    row[key] = None
            else:
                row[key] = None
            cells.append(f"{row[key]:>10.4f}" if isinstance(row[key], (int, float)) else f"{'--':>10}")
        if any(row.get(k) is not None for _, k in slices):
            print(f"{name:<22} " + " ".join(cells))
            per_slice.append(row)

    # Production baseline for reference
    base = {}
    base_cells = []
    for fname, key in slices:
        p = lib.CACHE / "world_v10_fourclass_v9" / fname
        v = json.loads(p.read_text()).get("f1") if p.exists() else None
        base[key] = v
        base_cells.append(f"{v:>10.4f}" if isinstance(v, (int, float)) else f"{'--':>10}")
    print(f"{'production (seed 42)':<22} " + " ".join(base_cells))

    # ------------------------------------ per-slice sigma and lever verdicts
    # A lever's delta is only meaningful against run-to-run variance. The
    # difference of two single runs has variance 2*sigma^2, so the decision band
    # is 2*sqrt(2)*sigma, not 2*sigma.
    lib.header("Levers vs seed noise, per slice (macro-F1)")
    seed_names = [r for r in SEED_RUNS.values() if r]
    sigmas = {}
    for _, key in slices:
        vals = [base[key]] + [r[key] for r in per_slice
                              if r["run"] in seed_names and r.get(key) is not None]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) >= 3:
            sigmas[key] = float(np.std(vals, ddof=1))
    print("seed sigma:  " + "   ".join(
        f"{k}={sigmas[k]:.4f} (2*sqrt2*sig={2*np.sqrt(2)*sigmas[k]:.4f})" for k in sigmas))

    lever_rows = []
    print(f"\n{'run':<22} " + " ".join(f"{s:>18}" for _, s in slices))
    print("-" * 96)
    for r in per_slice:
        if r["run"] in seed_names:
            continue
        cells, rec = [], {"run": r["run"]}
        for _, key in slices:
            v, b, s = r.get(key), base.get(key), sigmas.get(key)
            if not isinstance(v, (int, float)) or not isinstance(b, (int, float)) or not s:
                cells.append(f"{'--':>18}")
                continue
            d = v - b
            band = 2 * np.sqrt(2) * s
            mark = "*" if abs(d) > band else " "
            cells.append(f"{d:+.4f}({d/s:+.1f}s){mark:>1}")
            rec[key] = {"delta": d, "d_over_sigma": d / s, "separates": bool(abs(d) > band)}
        print(f"{r['run']:<22} " + " ".join(cells))
        lever_rows.append(rec)
    print("\n  * = |delta| exceeds the 2*sqrt(2)*sigma band for that slice "
          "(i.e. distinguishable from seed noise)")

    lib.save("gpu_runs_analysis", {
        "frozen_n": len(ids),
        "seeds": seed_rows,
        "sigma_auc": sigma_auc, "sigma_macro_f1": sigma_f1,
        "levers": results,
        "per_slice_macro_f1": per_slice,
        "per_slice_sigma": sigmas,
        "per_slice_lever_verdicts": lever_rows,
    })


if __name__ == "__main__":
    main()
