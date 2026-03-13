"""Run the full pipeline: candidates → patch_extraction → train → inference → visualize.

Usage::

    python -m training.run_pipeline --config configs/chicken_eggs_united_states.yaml

Optional::

    python -m training.run_pipeline --config configs/chicken_eggs_united_states.yaml --max-patches 100
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: candidates → patches → train → inference → visualize"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--skip", nargs="+", choices=[s[0] for s in _steps()], default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config_arg = ["--config", args.config]
    skip = set(args.skip)

    for name, cmd in _steps():
        if name in skip:
            log.info("Skipping %s", name)
            continue
        full_cmd = cmd + config_arg
        if name == "patch_extraction" and args.max_patches is not None:
            full_cmd += ["--max-patches", str(args.max_patches)]
        log.info("=== Step: %s ===", name)
        result = subprocess.run(full_cmd)
        if result.returncode != 0:
            log.error("Step %s failed with exit code %d", name, result.returncode)
            return result.returncode

    log.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
