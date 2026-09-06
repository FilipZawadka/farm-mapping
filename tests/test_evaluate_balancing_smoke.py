"""Smoke test for experiments/evaluate_balancing.py on synthetic scores.

Builds a fake master parquet plus fake collected runs with an injected ground
truth -- arm g ranks better than a, and arms a / j carry a region-driven score
offset (US negatives scored high, European negatives scored low) that the
balanced arms lack -- then runs the evaluator end-to-end and checks that the
injected ordering is recovered and every section renders. This is the same
kind of check that caught the empty-confirmatory-family bug in evaluate_r4
(EVAL_METHODS.md, "Estimands"). No GPU, no real data.

Run:  python tests/test_evaluate_balancing_smoke.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

# The layout of the real campaign: round_4 arm A has three seeds, every balanced
# arm has the single seed-44 run.
RUN_SEEDS = {"a": (42, 43, 44), "g": (44,), "h": (44,), "i": (44,), "j": (44,)}
# arm -> (skill, region offset strength); offset pushes scores up in the US/MEX
# and down in Europe regardless of the label.
TRUTH = {"a": (1.4, 1.2), "g": (2.0, 0.1), "h": (1.6, 0.3), "i": (1.7, 0.4), "j": (1.4, 1.1)}
FARM_LABELS = ["Farm: Poultry: Eggs", "Farm: Poultry: Meat Chickens", "Farm: Pigs", "Farm: Cattle"]


def build(tmp: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    rows = []
    plan = {
        "train": {"USA": (600, 0.8), "MEX": (200, 0.9), "RUS": (400, 0.0), "DEU": (150, 0.15),
                  "THA": (100, 0.6), "POL": (100, 0.5)},
        "val": {"USA": (150, 0.8), "MEX": (60, 0.9), "RUS": (120, 0.0), "DEU": (60, 0.15),
                "UKR": (40, 0.0), "POL": (40, 0.5), "THA": (40, 0.6)},
        "test": {"USA": (300, 0.8), "MEX": (100, 0.9), "THA": (60, 0.6), "BRA": (40, 0.6)},
        "eval": {"USA": (60, 0.7), "MEX": (60, 0.5), "THA": (60, 0.4), "BRA": (60, 0.5), "CHL": (60, 0.5)},
        "generalization": {"BGD": (170, 0.7), "NGA": (90, 0.6), "ALB": (140, 0.6), "COD": (45, 0.7),
                           "IND": (100, 0.6), "MAR": (95, 0.6)},
    }
    k = 0
    for split, countries in plan.items():
        for iso, (n, farm_rate) in countries.items():
            for _ in range(n):
                farm = rng.random() < farm_rate
                label = rng.choice(FARM_LABELS) if farm else "NotFarm"
                rows.append({"cluster_id": f"{iso}_cluster_{k}", "ADM0": iso, "final_label": label,
                             "cnn_split_assigned": split, "lat": 0.0, "lng": 0.0})
                k += 1
    # a handful of rows the evaluator must ignore
    rows.append({"cluster_id": "USA_cluster_amb", "ADM0": "USA", "final_label": "Ambiguous",
                 "cnn_split_assigned": "generalization", "lat": 0.0, "lng": 0.0})
    v10 = pd.DataFrame(rows)
    v10_path = tmp / "all_clusters_v10.parquet"
    v10.to_parquet(v10_path, index=False)

    europe = {"RUS", "DEU", "UKR", "POL"}
    us = {"USA", "MEX"}
    y = (v10.final_label != "NotFarm").to_numpy().astype(float)
    region = np.where(v10.ADM0.isin(us), 1.0, np.where(v10.ADM0.isin(europe), -1.0, 0.0))
    gpu = tmp / "gpu_results"
    for arm, (skill, offset) in TRUTH.items():
        for s in RUN_SEEDS[arm]:
            r = np.random.default_rng(1000 * s + ord(arm))
            logit = skill * (2 * y - 1) + offset * region + r.normal(0, 1.0, len(y)) + r.normal(0, 0.15)
            p_farm = 1 / (1 + np.exp(-logit))
            run = gpu / f"world_v10_fourclass_r4_{arm}_s{s}"
            run.mkdir(parents=True)
            pd.DataFrame({"candidate_id": v10.cluster_id, "prob_class0": 1 - p_farm,
                          "prob_class1": p_farm * 0.8, "prob_class2": p_farm * 0.15,
                          "prob_class3": p_farm * 0.05}).to_parquet(run / "scored_candidates.parquet", index=False)
            (run / "eval_metrics.json").write_text(json.dumps(
                {f"f1_class{i}": 0.5 + 0.01 * i for i in range(4)}))
            if arm != "a":
                (run / "sampling_report.json").write_text(json.dumps({
                    "scheme": {"g": "grouped_country", "h": "bucket", "i": "capped", "j": "grouped_country"}[arm],
                    "class_conditional": arm != "j", "n_groups": 7, "ess_ratio": 0.55,
                    "weight_max": 10.0, "clipped_frac": 0.01,
                    "label_region_dependence": {"nmi_natural": 0.45, "nmi_achieved": 0.2}}))
    return v10_path, gpu


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        v10_path, gpu = build(tmp)
        import lib
        import evaluate_balancing as eb
        lib.RESULTS = tmp / "results"
        lib.RESULTS.mkdir()
        sys.argv = ["evaluate_balancing.py", "--v10", str(v10_path), "--gpu-dir", str(gpu), "--boot", "300"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eb.main()
        out = buf.getvalue()
        print(out[-6000:])

        rep = json.loads((lib.RESULTS / "balancing_evaluation.json").read_text())
        gen = rep["generalization"]
        # injected ordering recovered on the primary slice
        assert gen["contrasts"]["a_vs_g"]["delta"] > 0.02, gen["contrasts"]["a_vs_g"]
        assert "g>a" in out and "confirmatory family (Holm-adjusted)" in out
        assert "exploratory (unadjusted): j>g" in out
        # the shortcut signature: arm a's negatives spread across buckets, arm g's do not
        spread = rep["val"]["buckets"]["p_neg_spread"]
        assert spread["a"] > spread["g"] + 0.1, spread
        assert "RUS" in rep["val"]["country_fpr"] and "UKR" in rep["val"]["country_fpr"]
        assert rep["val"]["country_fpr"]["RUS"]["fpr"]["a"] < rep["val"]["country_fpr"]["RUS"]["fpr"]["g"] + 1.0
        assert "within_country" in gen and set(gen["within_country"]) == set(TRUTH)
        assert len(rep["sampling_reports"]) == 4
        assert "measured on 1 arm(s) with >1 seed" in out       # sigma_seed from arm A's three seeds
        assert "arms with a single run: g, h, i, j" in out
        assert len(gen["per_arm"]["a"]) == 3 and len(gen["per_arm"]["g"]) == 1
        assert "per_class" in rep and rep["per_class"]
        print("\nevaluate_balancing smoke test passed")


if __name__ == "__main__":
    main()
