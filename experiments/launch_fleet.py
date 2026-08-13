"""Launch experiment configs on RunPod with a concurrency cap and a budget guard.

Pods auto-terminate when their pipeline finishes, so this keeps at most
--max-concurrent alive and tops the fleet up as slots free.

Usage:
  python3 experiments/launch_fleet.py --list
  python3 experiments/launch_fleet.py --max-concurrent 5 --budget 30
  python3 experiments/launch_fleet.py --status
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
STATE = Path(__file__).resolve().parent / "results" / "fleet_state.json"

# Ordered by value: seed variance first (it calibrates every other delta),
# then the levers most likely to change the production recipe.
ORDER = [
    "e03_seed43", "e03_seed44", "e03_seed45", "e03_seed46",   # E0.3 seed variance
    "e12_crop128", "e12_crop48",                               # E1.2 context
    "e16_ssl4eo",                                              # E1.6 backbone
    "e11_6bands", "e11_rgb", "e11_rgb_nir", "e11_rgb_ndwi",    # E1.1 bands
    "e11_recompute_idx",
    "e17_val_loss",                                            # E1.7 checkpoint metric
    "e19_three_class",                                         # E1.9 taxonomy
    "e15_cutout_only", "e15_no_photometric", "e15_geometric_only",  # E1.5 augmentation
    "e14_lr3e-5", "e14_lr3e-4", "e14_freeze0",                 # E1.4 optimiser
]

log = logging.getLogger("fleet")


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


def account() -> dict:
    d = _api("query { myself { clientBalance currentSpendPerHr "
             "pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }")
    return d["myself"]


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"launched": {}, "spend_estimate": 0.0}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def launch_one(name: str) -> str | None:
    cfg = f"configs/experiments/{name}.yaml"
    if not (REPO / cfg).exists():
        log.error("missing config %s", cfg)
        return None
    log.info("launching %s ...", name)
    p = subprocess.run(
        [sys.executable, "-m", "training.runpod_launch", "--config", cfg,
         "--steps", "train", "inference"],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    for line in (p.stdout + p.stderr).splitlines():
        if "Pod created" in line or "Pod launched" in line:
            log.info("  %s", line.strip())
    for line in (p.stdout + p.stderr).splitlines():
        if "Pod launched:" in line:
            return line.split("Pod launched:")[1].strip()
    log.error("  launch failed for %s:\n%s", name, (p.stdout + p.stderr)[-1500:])
    return None


def cmd_status() -> None:
    me = account()
    pods = me["pods"]
    print(f"balance ${me['clientBalance']:.2f}   spend ${me['currentSpendPerHr']:.3f}/hr   "
          f"running pods {len(pods)}")
    for p in pods:
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        print(f"  {p['id']:<18} {p['name']:<46} {p['desiredStatus']:<8} "
              f"${p.get('costPerHr', 0):.3f}/hr  up {up//60}m")
    st = load_state()
    done = [k for k, v in st["launched"].items() if v.get("pod_id")]
    print(f"\nlaunched so far: {len(done)}/{len(ORDER)}")
    remaining = [n for n in ORDER if n not in st["launched"]]
    if remaining:
        print(f"remaining: {', '.join(remaining)}")


def cmd_run(max_concurrent: int, budget: float, poll: int) -> None:
    st = load_state()
    queue = [n for n in ORDER if n not in st["launched"]]
    log.info("queue: %d configs, max_concurrent=%d, budget=$%.2f", len(queue), max_concurrent, budget)

    while queue:
        me = account()
        balance, running = me["clientBalance"], len(me["pods"])
        hourly = me["currentSpendPerHr"]

        if balance < budget:
            log.warning("balance $%.2f below reserve $%.2f -- stopping launches", balance, budget)
            break

        slots = max_concurrent - running
        if slots <= 0:
            log.info("%d/%d pods busy ($%.3f/hr, balance $%.2f) -- %d queued; sleeping %ds",
                     running, max_concurrent, hourly, balance, len(queue), poll)
            time.sleep(poll)
            continue

        for _ in range(min(slots, len(queue))):
            name = queue.pop(0)
            pod_id = launch_one(name)
            st["launched"][name] = {"pod_id": pod_id, "ts": time.time()}
            save_state(st)
            if pod_id:
                time.sleep(20)  # stagger so concurrent git syncs don't collide

    log.info("all configs launched; %d pods still running", len(account()["pods"]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--budget", type=float, default=15.0,
                    help="stop launching when balance falls below this reserve")
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n in ORDER:
            print(n)
        return
    if args.status:
        cmd_status()
        return
    cmd_run(args.max_concurrent, args.budget, args.poll)


if __name__ == "__main__":
    main()
