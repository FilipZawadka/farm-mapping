#!/usr/bin/env python3
"""Start a bare CPU pod with the network volume mounted, and print host:port.

The campaign has a chicken-and-egg: `sync`, `upload` and `wait` all need a
RUNNING pod to reach the volume, but the first pod the campaign would launch is
the baseline GPU run -- which cannot start until the data those stages produce
is already on the volume.

This starts the cheapest CPU pod that mounts the volume and leaves it idle.
RunPod CPU pods ignore docker_args (the prep path drives them over SSH), so
creating one without sending a startup script is enough to get a shell on the
volume. Terminate it when the data stages are done:

    python3 scripts/start_volume_pod.py            # create, wait for SSH, print
    python3 scripts/start_volume_pod.py --stop     # terminate any idle volume pod
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NAME = "farm-volume-shell"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rachel_clusters/world_v10_fourclass_r5_a_s44.yaml")
    ap.add_argument("--stop", action="store_true", help="terminate the idle volume pod")
    args = ap.parse_args()

    from training.env_loader import load_dotenv
    load_dotenv()                                  # .env holds RUNPOD_API_KEY
    from training.config import load_config
    from training import runpod_launch as R

    cfg = load_config(args.config)
    runpod = R._init_runpod(cfg)

    if args.stop:
        for p in runpod.get_pods():
            if p.get("name", "").startswith(NAME):
                runpod.terminate_pod(p["id"])
                print(f"terminated {p['id']} ({p['name']})")
        return

    for p in runpod.get_pods():
        if p.get("name", "").startswith(NAME) and p.get("desiredStatus") == "RUNNING":
            print(f"already running: {p['id']}")
            break
    else:
        p = None
        for instance_id in [cfg.runpod.cpu_instance_id, *cfg.runpod.cpu_fallbacks]:
            kwargs = R._build_prep_kwargs(cfg, args.config, instance_id)
            kwargs["name"] = NAME
            kwargs.pop("docker_args", None)          # leave it idle; no work to run
            try:
                p = runpod.create_pod(**kwargs)
                print(f"created {p['id']} on {instance_id}")
                break
            except Exception as exc:                  # instance type unavailable
                print(f"  {instance_id}: {str(exc)[:110]}")
        if p is None:
            sys.exit("no CPU instance type available")

    host, port = R._wait_for_ssh(p["id"], runpod)
    print(f"POD_HOST={host}\nPOD_PORT={port}")


if __name__ == "__main__":
    main()
