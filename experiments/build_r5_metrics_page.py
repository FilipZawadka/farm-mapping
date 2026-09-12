"""Build the round_5 metrics page: Rachel's per-country threshold metrics for
the four round_5 arms (A control, G, H, I; seed 44) with archived v9 as the
reference, packed into one self-contained HTML file.

Reads experiments/results/country_metrics/<run>/country_threshold_metrics.json
(written by evaluate_country_metrics.py: v11 labels, common clean rows) and
summary.json (for the lost/rescued-vs-v9 tables), trims the curves to what the
page needs, and injects them into experiments/r5_metrics_page_template.html at
the /*__DATA__*/ marker.

Run: python3 experiments/build_r5_metrics_page.py [--out /path/r5_metrics.html]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CM = HERE / "results" / "country_metrics"
TEMPLATE = HERE / "r5_metrics_page_template.html"

MODELS = [
    # key, run dir, label, description, categorical slot (1-4) or "ref"
    ("a", "world_v10_fourclass_r5_a_s44", "A · control", "round_5 labels, same recipe as round_4 A, no sampler", 1),
    ("g", "world_v10_fourclass_r5_g_s44", "G · grouped", "grouped countries (≥300 rows), uniform, class-conditional", 2),
    ("h", "world_v10_fourclass_r5_h_s44", "H · 3 buckets", "us / europe / rest buckets, class-conditional", 3),
    ("i", "world_v10_fourclass_r5_i_s44", "I · country cap", "per-country cap 20 %, class-conditional", 4),
    ("v9", "v9", "v9 · reference", "archived production model (round_3 labels)", "ref"),
]
CURVE_SPLITS = ("eval", "generalization", "test")


def r3(xs):
    return [round(float(x), 3) for x in xs]


def pack_task(e: dict) -> dict:
    curves = {}
    for c, by_split in e["pr_curves"].items():
        curves[c] = {sp: {"n": v["n"], "n_pos": v["n_pos"], "P": r3(v["precision"]), "R": r3(v["recall"])}
                     for sp, v in by_split.items() if sp in CURVE_SPLITS}
    th = e["threshold"]
    out = {
        "t_mean": round(th["mean_f1"], 4),
        "t_pooled": round(th["pooled_f1"], 4),
        "t_pfloor": round(th["precision_floor"]["t"], 4),
        "pfloor_met": bool(th["precision_floor"]["floor_met"]),
        "plateau": {k: r3(v) for k, v in th["plateau"].items()},
        "mean_f1_curve": r3(e["mean_f1_curve"]["mean_f1"]),
        "countries": e["countries"],
        "curves": curves,
        "per_country": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                        for row in e["per_country"]],
        "summary": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e["summary"].items()},
        "summary_loco": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e["summary_loco"].items()},
        "fp_fn_heldout": [{k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                          for row in e["fp_fn_heldout"]],
    }
    if "confusion_eval" in e:
        out["confusion_eval"] = e["confusion_eval"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "country_metrics" / "r5_metrics.html"))
    args = ap.parse_args()

    summary = json.loads((CM / "summary.json").read_text())
    data = {
        "generated": dt.date.today().isoformat(),
        "labels": summary.get("labels"),
        "n_blanked": summary.get("n_trained_blanked"),
        "grid": None,
        "models": [],
        "tasks": {"farm": {}, "poultry": {}},
        "vs_v9": {},
        "composition": None,
        "meta": None,
    }
    for key, run, label, desc, slot in MODELS:
        p = CM / run / "country_threshold_metrics.json"
        if not p.exists():
            print(f"! missing {p}")
            continue
        rep = json.loads(p.read_text())
        data["models"].append({"key": key, "run": run, "label": label, "desc": desc, "slot": slot})
        if data["grid"] is None:
            data["grid"] = r3(rep["tasks"]["farm"]["mean_f1_curve"]["grid"])
            data["composition"] = rep["eval_sets"]["composition"]
            data["meta"] = {k: rep["meta"][k] for k in ("focal_countries", "generalization_countries",
                                                         "excluded_countries", "n_rows", "n_labelled",
                                                         "n_heldout_labelled")}
        for task in ("farm", "poultry"):
            if task in rep["tasks"]:
                d = pack_task(rep["tasks"][task])
                d["headline"] = rep["headline"]
                data["tasks"][task][key] = d
        flat = summary["runs"].get(run, {})
        if "vs_v9" in flat:
            data["vs_v9"][key] = flat["vs_v9"]

    html = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(",", ":"))
    assert "/*__DATA__*/" in html
    html = html.replace("/*__DATA__*/", payload, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html) // 1024} KB; data {len(payload) // 1024} KB; models {[m['key'] for m in data['models']]})")


if __name__ == "__main__":
    main()
