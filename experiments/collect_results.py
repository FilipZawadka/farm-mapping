"""Pull experiment outputs off the RunPod network volume.

The volume is only reachable through a running pod, and training pods
auto-terminate when they finish -- so this must run while at least one pod is
alive. It picks any RUNNING pod with SSH exposed and rsyncs from the shared
volume (every pod mounts the same /workspace).

Usage:
  python3 experiments/collect_results.py            # pull all finished runs
  python3 experiments/collect_results.py --watch    # keep pulling until fleet is done
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEST = REPO / "experiments" / "gpu_results"
REMOTE_OUT = "/workspace/farm-mapping/data/output"

log = logging.getLogger("collect")

WANT = [
    "training_metrics.json", "eval_metrics.json", "generalization_metrics.json",
    "qual_eval_metrics.json", "inspected_metrics.json",
    "eval_metrics_per_country.json", "generalization_metrics_per_country.json",
    "qual_eval_metrics_per_country.json",
    "scored_candidates.parquet", "config.yaml",
]


def _api(query: str) -> dict:
    from training.env_loader import load_dotenv
    load_dotenv()
    import os
    key = os.environ["RUNPOD_API_KEY"]
    out = subprocess.run(
        ["curl", "-s", "-H", f"Authorization: Bearer {key}", "-H", "Content-Type: application/json",
         "-X", "POST", "https://api.runpod.io/graphql", "-d", json.dumps({"query": query})],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["data"]


def live_pod() -> tuple[str, int] | None:
    """SSH endpoint of any running pod (all mount the same network volume)."""
    d = _api("query { myself { pods { id name desiredStatus runtime { "
             "ports { ip publicPort privatePort isIpPublic } } } } }")
    for p in d["myself"]["pods"]:
        if p["desiredStatus"] != "RUNNING":
            continue
        for port in ((p.get("runtime") or {}).get("ports") or []):
            if port["privatePort"] == 22 and port.get("isIpPublic"):
                return port["ip"], port["publicPort"]
    return None


def pull(host: str, port: int, names: list[str]) -> list[str]:
    """Stream wanted files back via tar over SSH.

    The RunPod pytorch image has no rsync, so tar is the portable option: it is
    present in every image and needs nothing installed on the pod.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    pulled = []
    for name in names:
        # Build the file list on the pod; missing files must not fail the tar.
        globs = " ".join(f"{name}/{w}" for w in WANT)
        remote = (
            f"cd {REMOTE_OUT} 2>/dev/null && "
            f"ls -d {globs} 2>/dev/null | tar czf - -T - 2>/dev/null || true"
        )
        proc = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
             "-p", str(port), f"root@{host}", remote],
            capture_output=True, timeout=600,
        )
        if proc.returncode != 0 or not proc.stdout:
            continue
        untar = subprocess.run(
            ["tar", "xzf", "-", "-C", str(DEST)],
            input=proc.stdout, capture_output=True,
        )
        if untar.returncode != 0:
            log.warning("  %-22s untar failed: %s", name, untar.stderr.decode()[:160])
            continue
        got = sorted(p.name for p in (DEST / name).glob("*")) if (DEST / name).exists() else []
        if got:
            pulled.append(name)
            log.info("  %-22s %s", name, ", ".join(got))
    return pulled


def expected_runs() -> list[str]:
    from experiments.launch_fleet import ORDER  # noqa
    return list(ORDER)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll", type=int, default=600)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "experiments"))
    from launch_fleet import ORDER

    while True:
        ep = live_pod()
        if not ep:
            log.error("no RUNNING pod with public SSH -- cannot reach the network volume. "
                      "Launch one pod to collect results.")
            return
        host, port = ep
        log.info("collecting via %s:%d", host, port)
        pulled = pull(host, port, ORDER)
        complete = [n for n in ORDER
                    if (DEST / n / "qual_eval_metrics.json").exists()]
        log.info("runs with completed metrics: %d/%d", len(complete), len(ORDER))

        if not args.watch or len(complete) == len(ORDER):
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
