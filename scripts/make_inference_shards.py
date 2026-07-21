"""Generate country-sharded, inference-only configs from a trained model's config.

Scoring every candidate we have (~136k) on one GPU is loader-bound and slow, so
the work is split across N pods by country. Sharding is by country because
``data.countries`` (consumed by ``training.inference._load_candidates_csv``) is
the only built-in row filter, and candidate CSVs are written one-per-country.

Each shard config differs from the base only in:
  * ``run_name`` / filename -- a distinct config stem, so shards write to
    distinct ``data/output/{stem}/`` dirs. This matters: the scored-parquet
    write is NOT atomic, so same-stem pods would corrupt each other's output.
  * ``data.countries`` -- its slice of the work (country_key values, e.g.
    ``united_states``, NOT ADM0 codes like ``USA``).
  * ``inference.norm_stats_stem`` -- points back at the training config, since
    norm stats / split assignments are keyed by stem and only written at train
    time.

Plus the scoring-run settings applied to every shard: keep every row
(``keep_unscorable_labels``, ``include_unlabeled``, ``labeled_only: false``)
and inference-tuned loader/precision knobs.

Usage::

    python scripts/make_inference_shards.py \
        --base configs/rachel_clusters/world_v10_fourclass_softcon.yaml \
        --parquet data/rachel_geometry_candidates/all_countries/all_clusters_v5.parquet \
        --shards 4 --name world_v10_fourclass_scoreall
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.rachel_to_candidates import _ADM0_TO_KEY  # noqa: E402


def country_keys(parquet: Path) -> pd.Series:
    """Row counts per country_key -- mirrors rachel_to_candidates.convert()."""
    df = pd.read_parquet(parquet, columns=["ADM0"])
    key = df["ADM0"].map(_ADM0_TO_KEY).fillna(df["ADM0"].str.lower())
    return key.value_counts()


def partition(counts: pd.Series, n: int) -> list[list[str]]:
    """Greedy longest-processing-time partition: assign the largest country to
    the currently-smallest shard. Countries are atomic (one CSV each), so a
    single dominant country sets the floor on the largest shard."""
    shards: list[list[str]] = [[] for _ in range(n)]
    totals = [0] * n
    for name, cnt in counts.items():
        i = totals.index(min(totals))
        shards[i].append(str(name))
        totals[i] += int(cnt)
    return shards, totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                     help="config to copy model/patch/training settings from")
    ap.add_argument("--model-stem",
                     help="config stem that actually TRAINED the checkpoint, if different from "
                          "--base (e.g. scoring a new data version with an existing model). "
                          "Determines inference.checkpoint and inference.norm_stats_stem, since "
                          "both the checkpoint and the norm-stats/splits files live under the "
                          "training run's stem. Defaults to --base's stem.")
    ap.add_argument("--parquet", required=True, type=Path, help="all_clusters parquet (for country counts)")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--name", required=True, help="config stem prefix, e.g. world_v10_fourclass_scoreall")
    ap.add_argument("--candidates-dir", default="data/rachel_geometry_candidates/candidates_world_v10_scoreall",
                     help="SHARED across shards -- written once by the candidates step, read-only to shards")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("configs/rachel_clusters"))
    args = ap.parse_args()

    base = yaml.safe_load(args.base.read_text())
    base_stem = args.model_stem or args.base.stem

    counts = country_keys(args.parquet)
    shards, totals = partition(counts, args.shards)
    print(f"{len(counts)} country files, {int(counts.sum()):,} rows -> {args.shards} shards "
          f"(balance {max(totals)/max(min(totals),1):.2f}x)")

    for i, (countries, total) in enumerate(zip(shards, totals)):
        cfg = yaml.safe_load(args.base.read_text())  # fresh copy per shard
        stem = f"{args.name}_shard{i}"
        cfg["run_name"] = stem

        cfg["data"]["candidates_dir"] = args.candidates_dir
        cfg["data"]["countries"] = sorted(countries)
        # Score everything: keep unlabeled rows AND rows whose label has no slot
        # in the 4-class taxonomy (Ambiguous / Mixed / Other / Unknown /
        # PigsOrPoultry). Both arrive as label=-1 and are scored, never trained.
        cfg["data"]["include_unlabeled"] = True
        cfg["data"]["keep_unscorable_labels"] = True

        inf = cfg.setdefault("inference", {})
        # Explicit path to the ALREADY-TRAINED checkpoint (stem-independent).
        inf["checkpoint"] = f"data/output/{base_stem}/best_model.pt"
        inf["labeled_only"] = False
        inf["norm_stats_stem"] = base_stem
        inf["batch_size"] = args.batch_size
        inf["num_workers"] = args.num_workers
        inf["mixed_precision"] = True

        cfg["mlflow"]["experiment_name"] = stem
        cfg["visualization"]["output_dir"] = f"output/maps_{stem}"

        header = (
            f"# ============================================================================\n"
            f"# {stem} -- inference-only shard {i+1}/{args.shards} of a full scoring pass.\n"
            f"#\n"
            f"# GENERATED by scripts/make_inference_shards.py -- do not hand-edit; regenerate.\n"
            f"# Scores the checkpoint trained by {base_stem}.yaml over every candidate we\n"
            f"# have, including rows training drops (Ambiguous + Mixed/Other/Unknown/\n"
            f"# PigsOrPoultry -> label=-1, scored but never trained on) and all unlabeled\n"
            f"# rest-of-world rows.\n"
            f"#\n"
            f"# This shard: {len(countries)} countries, ~{total:,} candidates.\n"
            f"# Run with:  --steps inference   (candidates step runs ONCE, separately --\n"
            f"# convert() writes ALL country CSVs, so concurrent shards would race on it.)\n"
            f"# ============================================================================\n"
        )
        path = args.out_dir / f"{stem}.yaml"
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=100))
        print(f"  {path}  ({len(countries)} countries, ~{total:,} rows)")


if __name__ == "__main__":
    main()
