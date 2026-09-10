#!/usr/bin/env python3
"""Generate the region-balancing campaign configs (arms G/H/I, optional J) for a data round.

Every arm is the round's anchor recipe (configs/rachel_clusters/
world_v10_fourclass_{r4,r5}.yaml == arm A, byte-identical v9/v6 recipe) with
ONE block changed -- training.region_balancing -- so a difference against arm A
is attributable to the sampler. Default: round_5 (Rachel's split-level cap),
ONE run per scheme at seed 44 plus the round_5 baseline arm A (no sampler),
which is the control (same data, same seed => same init and augmentation
stream). Single runs support artifact-level claims only (Dietterich Q3); add
--seeds 42 43 44 for the three-seed recipe-level design. `--round r4` rebuilds
the round_4 arms (control = the existing world_v10_fourclass_r4_a_s44).

  G  grouped_country : countries with >= 300 train rows are their own group,
                       smaller ones pool into rest_<macro-region>; uniform share
  H  bucket          : us / europe / rest, uniform share
  I  capped          : natural shares, no country above 20% of an epoch
  J  (optional, --with-ablation) : G's share law WITHOUT class conditioning --
                       isolates what the anti-shortcut term adds

All of G/H/I condition the class mix inside each group on the global prior
(class_conditional: true) -- that is the term that actually breaks the
region->label shortcut; see docs/COUNTRY_BALANCING_PLAN.md.

Run:  python3 experiments/gen_balancing_configs.py --selftest   # round_5 configs + bash -n
      python3 experiments/gen_balancing_configs.py --round r4    # round_4 arms
Then: bash scripts/run_round5_campaign.sh                        # sync, merge, launch, collect
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs" / "rachel_clusters"

# Rounds: the anchor config (arm A recipe on that round's data) and whether the
# baseline run already exists. round_4's arm A was trained in the round_4
# campaign; round_5 (Rachel's split-level cap, new eval sets, label fixes) has
# no baseline yet, so arm A is generated with the balanced arms.
ROUNDS = {
    "r4": {"base": "world_v10_fourclass_r4.yaml", "generate_baseline": False},
    "r5": {"base": "world_v10_fourclass_r5.yaml", "generate_baseline": True},
}
DEFAULT_ROUND = "r5"

# The sampler code lives on this branch; a pod cloning an older branch would
# silently ignore the unknown block and train an UNBALANCED baseline under a
# balanced run name. train.py also refuses to start a run whose config asks
# for balancing but whose dataset carries no weights.
BRANCH = "develop"
DEFAULT_SEEDS = (44,)          # pairs with round_4 arm A seed 44 (the control)
CORE_ARMS = ("g", "h", "i")
MAX_WEIGHT = 10.0

ARMS: dict[str, tuple[str, dict]] = {
    "a": ("baseline: same data and recipe, NO sampler (the control)",
          {"enabled": False}),
    "g": ("grouped countries (>=300 rows own group, rest pooled by macro-region), "
          "uniform share, class-conditional",
          {"enabled": True, "scheme": "grouped_country", "min_country_rows": 300,
           "class_conditional": True, "class_axis": "binary", "max_weight": MAX_WEIGHT}),
    "h": ("3 buckets us / europe / rest, uniform share, class-conditional",
          {"enabled": True, "scheme": "bucket",
           "buckets": {"us": ["USA"], "europe": ["EUROPE"], "rest": ["*"]},
           "class_conditional": True, "class_axis": "binary", "max_weight": MAX_WEIGHT}),
    "i": ("per-country cap: no country above 20% of an epoch (pro-rata redistribution), "
          "class-conditional",
          {"enabled": True, "scheme": "capped", "max_share": 0.20,
           "class_conditional": True, "class_axis": "binary", "max_weight": MAX_WEIGHT}),
    "j": ("ABLATION: arm G share law WITHOUT class conditioning (region marginals only)",
          {"enabled": True, "scheme": "grouped_country", "min_country_rows": 300,
           "class_conditional": False, "class_axis": "binary", "max_weight": MAX_WEIGHT}),
}


def deep_merge(base: dict, delta: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_name(rnd: str, arm: str, seed: int) -> str:
    return f"world_v10_fourclass_{rnd}_{arm}_s{seed}"


def make(rnd: str, arm: str, seed: int, base: dict, base_path: Path) -> Path:
    desc, block = ARMS[arm]
    cfg = deep_merge(base, {
        "run_name": run_name(rnd, arm, seed),
        "training": {"seed": seed, "region_balancing": block,
                     # belt and braces: the legacy samplers stay off
                     "upsample_minority_regions": False,
                     "balanced_country_splits": False,
                     "balanced_class_sampling": False},
        "runpod": {"github_branch": BRANCH},
    })
    control = run_name(rnd, "a", seed)
    header = (
        f"# balancing campaign ({rnd}) arm {arm.upper()} seed {seed}: {desc}\n"
        f"# Single-lever delta vs {rnd} arm A: {{'training': {{'region_balancing': {block}}}}}\n"
        f"# Generated by experiments/gen_balancing_configs.py from {base_path.relative_to(REPO)}.\n"
        + (f"# Control: {control} (same data, same recipe, same seed, no balancing).\n" if arm != "a" else
           f"# This IS the control for the {rnd} balanced arms (no sampler).\n")
        + f"# runpod.github_branch={BRANCH}: the sampler code lives there -- an older branch\n"
        f"# would ignore the block and train an unbalanced run under this name.\n\n"
    )
    path = OUT / f"{run_name(rnd, arm, seed)}.yaml"
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=sorted(ROUNDS), default=DEFAULT_ROUND,
                    help=f"which round's data/anchor config to build on (default {DEFAULT_ROUND})")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="seeds per arm (default: 44)")
    ap.add_argument("--with-ablation", action="store_true",
                    help="also generate arm J (grouped countries without class conditioning)")
    ap.add_argument("--selftest", action="store_true", help="bash -n the pod startup scripts")
    args = ap.parse_args()
    rnd = args.round
    base_path = OUT / ROUNDS[rnd]["base"]
    arms = (["a"] if ROUNDS[rnd]["generate_baseline"] else []) + list(CORE_ARMS) \
        + (["j"] if args.with_ablation else [])
    seeds = tuple(args.seeds)
    order_file = REPO / "experiments" / f"balancing_order_{rnd}.txt"

    base = yaml.safe_load(base_path.read_text())
    written = []
    for arm in arms:
        for seed in seeds:
            written.append(make(rnd, arm, seed, base, base_path))
    # Baseline first: its candidates step builds the round's candidates dir,
    # which the balanced arms read (see scripts/run_round5_campaign.sh).
    order = [run_name(rnd, a, s) for a in arms for s in seeds]
    order_file.write_text("\n".join(order) + "\n")
    print(f"wrote {len(written)} configs -> {OUT.relative_to(REPO)} and {order_file.relative_to(REPO)}")
    for p in written:
        print("  ", p.name)

    # Validate every config through the real loader: the block must parse, be
    # enabled on balanced arms (and off on the baseline), and point at the
    # branch that carries the sampler.
    sys.path.insert(0, str(REPO))
    from training.config import load_config
    bad = 0
    for p in written:
        try:
            c = load_config(p)
            rb = c.training.region_balancing
            is_baseline = "_a_s" in p.name
            assert rb.enabled != is_baseline, f"region_balancing.enabled={rb.enabled}"
            assert c.runpod.github_branch == BRANCH, f"github_branch={c.runpod.github_branch}"
            assert not c.training.upsample_minority_regions
            assert c.data.candidates_dir == base["data"]["candidates_dir"]
            if not is_baseline:
                assert rb.max_weight == MAX_WEIGHT
        except Exception as exc:
            print(f"  *** {p.name}: {type(exc).__name__}: {str(exc)[:160]}"); bad += 1
    print("all campaign configs valid" if not bad else f"*** {bad} INVALID ***")

    if args.selftest:
        _selftest_startup_scripts(written)


def _selftest_startup_scripts(paths: list[Path]) -> None:
    """bash -n the startup script the launcher would send for each config.

    A malformed fragment kills the pod's script on line 1 and the pod sits idle
    and billing with no visible error (see experiments/gen_configs.py).
    """
    import subprocess
    import tempfile
    from training.config import load_config, resolve_paths
    try:
        from training import runpod_launch as rl
    except ImportError as exc:                 # e.g. no `runpod` package on this machine
        print(f"selftest skipped: cannot import training.runpod_launch ({exc})")
        return
    bad = []
    for p in paths:
        cfg = resolve_paths(load_config(p))
        rel = str(p.relative_to(REPO))
        script = rl._build_startup_script(cfg, rel, steps=["train", "inference"])
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as fh:
            fh.write(script)
            fh.flush()
            r = subprocess.run(["bash", "-n", fh.name], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append((p.name, r.stderr.strip()[:200]))
        if BRANCH not in script:
            bad.append((p.name, f"startup script does not check out {BRANCH}"))
    for n, e in bad:
        print(f"  SYNTAX ERROR {n}: {e}")
    if bad:
        raise SystemExit(f"{len(bad)} startup scripts fail")
    print(f"startup-script self-test: {len(paths)} configs OK")


if __name__ == "__main__":
    main()
