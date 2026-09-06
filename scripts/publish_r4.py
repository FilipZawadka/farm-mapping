#!/usr/bin/env python3
"""Publish round_4 model runs to the website as individual datasets.

Each collected run becomes its own dataset dir under web/public/data/<id>/ plus a
registry entry. The frontend defaults to registry.datasets[0] (MapPage picks
`registry.datasets[0]?.id` when no ?ds= is given), and update_registry() sorts by
date descending with a *stable* sort -- so same-day entries keep insertion order.
Rather than depend on that, --default rewrites the registry afterwards to put the
chosen run first explicitly.

Usage:
  python3 scripts/publish_r4.py --dry-run --out /tmp/pub          # safe preview
  python3 scripts/publish_r4.py --default world_v10_fourclass_r4_d_s42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GPU = REPO / "experiments" / "gpu_results"
EXPORT = REPO / "web" / "scripts" / "export_dataset.py"
WEB_DATA = REPO / "web" / "public" / "data"

ARMS = {
    "a": "baseline (v9/v6 recipe)",
    "b": "freeze0 only",
    "c": "6 bands only",
    "d": "freeze0 + 6 bands",
    "e": "DenseNet-121 (ImageNet init, +freeze0 +6 bands)",
    "f": "freeze5 + full-LR unfreeze",
}


def arm_of(run: str) -> tuple[str, str]:
    """('b', '42') from world_v10_fourclass_r4_b_s42[_score]."""
    tail = run.rsplit("_r4_", 1)[-1].removesuffix("_score")   # e.g. 'b_s42'
    arm, _, seed = tail.partition("_s")
    return arm, seed


def runs_available() -> list[Path]:
    """Full-world scoring dirs only (`*_score`, ~152k points).

    The training dirs of the same name also hold a scored_candidates.parquet, but
    it covers only the 29,734 split-assigned rows (inference.labeled_only=true) --
    publishing those would put a partial map on the site under a name that looks
    like a full release.
    """
    return sorted(d for d in GPU.glob("world_v10_fourclass_r4_*_score")
                  if (d / "scored_candidates.parquet").exists())


def export_one(d: Path, out: Path, when: str, dry: bool) -> bool:
    run = d.name.removesuffix("_score")      # dataset id = the model's name
    arm, seed = arm_of(d.name)
    train_dir = d.parent / run               # metrics live with the training run
    desc = ARMS.get(arm, arm)
    cmd = [sys.executable, str(EXPORT),
           "--parquet", str(d / "scored_candidates.parquet"),
           "--id", run,
           "--version", f"Round 4 — arm {arm.upper()}: {desc} (seed {seed})",
           "--date", when,
           "--label-mode", "four_class",
           "--slim-geojson",
           "--out", str(out),
           "--notes", f"round_4 labels; arm {arm.upper()} = {desc}; seed {seed}."]
    for opt, fname in (("--metrics", "training_metrics.json"),
                       ("--per-country", "eval_metrics_per_country.json"),
                       ("--config", "config.yaml")):
        src = d / fname if (d / fname).exists() else train_dir / fname
        if src.exists():
            cmd += [opt, str(src)]
    if dry:
        print("  would run:", " ".join(cmd[2:6]), "...")
        return True
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  FAILED {run}:\n{(p.stdout + p.stderr)[-800:]}")
        return False
    print("  " + [l for l in p.stdout.splitlines() if "wrote" in l][-1][:150])
    return True


def set_default(registry: Path, default_id: str) -> None:
    """Move default_id to index 0; the frontend treats that as the default."""
    data = json.loads(registry.read_text())
    items = data.get("datasets", [])
    hit = [d for d in items if d["id"] == default_id]
    if not hit:
        print(f"  ! default '{default_id}' not in registry -- leaving order as-is")
        return
    data["datasets"] = hit + [d for d in items if d["id"] != default_id]
    registry.write_text(json.dumps(data, indent=2))
    print(f"  default dataset -> {default_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WEB_DATA)
    ap.add_argument("--default", help="run id to place first (site default)")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="publish only these run names")
    args = ap.parse_args()

    runs = runs_available()
    if args.only:
        want = set(args.only)
        runs = [d for d in runs if d.name in want or d.name.removesuffix("_score") in want]
    if not runs:
        print("no collected runs with scored_candidates.parquet"); return
    print(f"publishing {len(runs)} run(s) -> {args.out}")

    # Publish the default first so it also leads the same-date group naturally.
    if args.default:
        runs.sort(key=lambda d: d.name.removesuffix("_score") != args.default)

    ok = sum(export_one(d, args.out, args.date, args.dry_run) for d in runs)
    print(f"exported {ok}/{len(runs)}")

    reg = args.out / "registry.json"
    if args.default and reg.exists() and not args.dry_run:
        set_default(reg, args.default)


if __name__ == "__main__":
    main()
