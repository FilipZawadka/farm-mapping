"""Run the full pipeline: candidates → patch_extraction → train → inference → visualize.

Usage::

    python -m training.run_pipeline --config configs/chicken_eggs_united_states.yaml

Optional::

    python -m training.run_pipeline --config configs/chicken_eggs_united_states.yaml --max-patches 100
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _steps() -> list[tuple[str, list[str]]]:
    py = sys.executable
    return [
        ("candidates", [py, "-m", "training.candidates"]),
        ("patch_extraction", [py, "-m", "training.patch_extraction"]),
        ("train", [py, "-m", "training.train"]),
        ("inference", [py, "-m", "training.inference"]),
        ("visualize", [py, "-m", "training.visualize"]),
    ]


def _archive_outputs(config_path: str, run_dir: Path) -> None:
    """Copy training outputs into the run directory."""
    from training.config import load_config, resolve_paths
    cfg = resolve_paths(load_config(config_path))

    # Config-specific output directory
    output_dir = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem
    for name in ("best_model.pt", "training_metrics.json", "scored_candidates.parquet"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
            log.info("Archived %s", name)

    # Split assignments
    splits_path = Path(cfg.patches.output_dir).parent / "patches" / "splits" / f"{cfg._config_stem}.csv"
    if not splits_path.exists():
        patches_root = Path(cfg.patches.output_dir)
        splits_path = patches_root / "splits" / f"{cfg._config_stem}.csv"
    if splits_path.exists():
        shutil.copy2(splits_path, run_dir / "split_assignments.csv")
        log.info("Archived split_assignments.csv")

    # Prediction map
    vis_dir = Path(cfg.visualization.output_dir)
    for html in vis_dir.glob("*.html"):
        shutil.copy2(html, run_dir / html.name)
        log.info("Archived %s", html.name)


def _setup_run_dir(config_path: str, run_name: str = "") -> Path:
    """Create a timestamped run directory on the network volume and save the config.

    Structure: runs/{config_stem}/pipeline/{run_name}_{timestamp}/
    """
    config_stem = Path(config_path).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaf = f"{run_name}_{timestamp}" if run_name else timestamp
    run_dir = Path("/workspace/farm-mapping/runs") / config_stem / "pipeline" / leaf
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Symlink latest
    latest = run_dir.parent.parent / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(run_dir)

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: candidates → patches → train → inference → visualize"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-patches", type=int, default=None)
    all_step_names = [s[0] for s in _steps()]
    parser.add_argument("--steps", nargs="+", choices=all_step_names, default=None,
        help="Run only these steps (default: all). E.g. --steps train inference visualize")
    args = parser.parse_args()

    # Load config to get run_name
    from training.config import load_config
    cfg = load_config(args.config)

    # Set up run directory and logging
    run_dir = _setup_run_dir(args.config, run_name=cfg.run_name)
    log_file = run_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    log.info("Run directory: %s", run_dir)
    log.info("Config: %s", args.config)

    steps_to_run = set(args.steps) if args.steps else set(all_step_names)
    config_arg = ["--config", args.config]

    for name, cmd in _steps():
        if name not in steps_to_run:
            log.info("Skipping %s", name)
            continue
        full_cmd = cmd + config_arg
        if name == "patch_extraction" and args.max_patches is not None:
            full_cmd += ["--max-patches", str(args.max_patches)]
        log.info("=== Step: %s ===", name)
        step_log = run_dir / f"{name}.log"
        with open(step_log, "w") as f:
            result = subprocess.run(full_cmd, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            log.error("Step %s failed (exit %d). See %s", name, result.returncode, step_log)
            return result.returncode
        log.info("Step %s completed", name)

    # Copy outputs into the run directory for persistence
    _archive_outputs(args.config, run_dir)

    log.info("Pipeline completed successfully. Logs at %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
