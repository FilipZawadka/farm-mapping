"""Run training on a local-network GPU box (e.g. desktop with RTX 3080) over SSH.

Mirrors :mod:`training.runpod_launch` semantics, but skips the provider API
entirely — the machine is always there, you just SSH in and start a job in
``tmux``.

Setup
-----
1. The remote box reachable over SSH (LAN or Tailscale).
2. Add to ``.env``::

       LOCAL_HOST=desktop.tail-scale.ts.net   # or LAN IP
       LOCAL_PORT=22
       LOCAL_USER=filip
       LOCAL_CODE_DIR=/home/filip/farm-mapping
       LOCAL_VENV=/home/filip/farm-mapping/.venv
       LOCAL_SSH_KEY=~/.ssh/id_ed25519        # optional, defaults to id_ed25519

3. Run ``scripts/setup_local_box.sh`` on the target box once to provision
   CUDA / venv / repo clone.

Usage
-----
::

    python -m training.local_launch --config configs/rachel_clusters/baseline_v2_multiclass.yaml
    python -m training.local_launch --config <yaml> --steps train inference visualize
    python -m training.local_launch --list           # list running tmux train sessions
    python -m training.local_launch --kill <name>    # kill a tmux train session
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .env_loader import load_dotenv

log = logging.getLogger(__name__)


def _read_local_env() -> dict[str, str]:
    """Pull LOCAL_* connection settings from .env, raise if any are missing."""
    load_dotenv()
    required = ["LOCAL_HOST", "LOCAL_CODE_DIR"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing in .env: {', '.join(missing)}. See training/local_launch.py docstring."
        )
    host = os.environ["LOCAL_HOST"]
    code_dir = os.environ["LOCAL_CODE_DIR"]
    return {
        "host": host,
        "port": os.environ.get("LOCAL_PORT", "22"),
        "user": os.environ.get("LOCAL_USER", os.environ.get("USER", "filip")),
        "code_dir": code_dir,
        "venv": os.environ.get("LOCAL_VENV", f"{code_dir}/.venv"),
        "ssh_key": os.path.expanduser(
            os.environ.get("LOCAL_SSH_KEY", "~/.ssh/id_ed25519")
        ),
    }


def _ssh_base(env: dict[str, str]) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-i", env["ssh_key"],
        "-p", env["port"],
        f"{env['user']}@{env['host']}",
    ]


def _build_startup_script(env: dict[str, str], config_name: str, run_name: str,
                          steps: list[str] | None) -> str:
    """Bash script that the remote box runs in tmux."""
    code_dir = env["code_dir"]
    venv = env["venv"]
    py = f"{venv}/bin/python"
    stem = config_name.removesuffix(".yaml")
    leaf = f"{run_name or stem}_$(date -u +%Y%m%d_%H%M%S)"

    steps_arg = f" --steps {' '.join(steps)}" if steps else ""

    return "\n".join([
        "set -euo pipefail",
        f"cd {code_dir}",
        # Pull latest main
        "git config --global --add safe.directory $(pwd) 2>/dev/null || true",
        "git fetch origin && git reset --hard origin/main",
        # Run dir layout matches runpod's so check_logs.sh works unchanged
        f"export RUN_DIR={code_dir}/runs/{stem}/pipeline/{leaf}",
        "mkdir -p $RUN_DIR",
        f"cp {code_dir}/configs/{config_name} $RUN_DIR/config.yaml",
        f"ln -sfn $RUN_DIR {code_dir}/runs/{stem}/latest",
        # Mirror runpod's "shared latest log" symlink so check_logs.sh tails work
        f"mkdir -p {code_dir}/runs",
        f"ln -sf $RUN_DIR/startup.log {code_dir}/runs/_latest_startup.log",
        # Venv: create if missing, otherwise reuse
        f"if [ ! -d {venv} ]; then",
        f"  echo '=== creating venv at {venv} ==='",
        f"  python3 -m venv {venv}",
        f"  {venv}/bin/pip install --upgrade pip",
        f"  {venv}/bin/pip install -r requirements-train.txt",
        "fi",
        f"echo '=== running pipeline ({config_name}) ==='",
        f"{py} -u -m training.run_pipeline --config configs/{config_name}{steps_arg}"
        f" 2>&1 | tee $RUN_DIR/startup.log",
        "echo '=== DONE ==='",
    ])


def _check_no_existing_session(env: dict[str, str], session_name: str) -> None:
    """Refuse to launch if the named tmux session already exists (single GPU)."""
    res = subprocess.run(
        _ssh_base(env) + [f"tmux has-session -t {session_name} 2>/dev/null && echo EXISTS || echo OK"],
        capture_output=True, text=True, check=False,
    )
    if "EXISTS" in res.stdout:
        raise RuntimeError(
            f"Tmux session '{session_name}' already running on {env['host']}. "
            f"Wait for it, kill it (--kill {session_name}), or pass --session <other>."
        )


def _other_train_sessions(env: dict[str, str]) -> list[str]:
    """Return names of currently-active tmux sessions that look like training runs."""
    res = subprocess.run(
        _ssh_base(env) + ["tmux list-sessions -F '#{session_name}' 2>/dev/null || true"],
        capture_output=True, text=True, check=False,
    )
    return [s for s in res.stdout.split() if s.startswith("train_")]


def launch(config_path: str, steps: list[str] | None = None,
           session: str | None = None, force: bool = False) -> None:
    env = _read_local_env()
    cfg = load_config(config_path)
    config_name = Path(config_path).name
    run_name = cfg.run_name or config_name.removesuffix(".yaml")
    session = session or f"train_{run_name}"

    if not force:
        existing = _other_train_sessions(env)
        if existing:
            log.warning(
                "Other training sessions are already running on %s: %s. "
                "Pass --force to launch anyway (single GPU → they'll compete).",
                env["host"], existing,
            )
            return
        _check_no_existing_session(env, session)

    script = _build_startup_script(env, config_name, run_name, steps)
    script_b64 = base64.b64encode(script.encode()).decode()

    remote_path = f"/tmp/{session}_startup.sh"
    write_cmd = (
        f"echo '{script_b64}' | base64 -d > {remote_path} && chmod +x {remote_path}"
    )
    setup_cmd = "which tmux >/dev/null 2>&1 || (sudo apt-get update -qq && sudo apt-get install -y -qq tmux)"
    run_cmd = f"tmux new-session -d -s {session} 'bash {remote_path}'"

    ssh = _ssh_base(env)
    log.info("Connecting to %s@%s:%s ...", env["user"], env["host"], env["port"])
    subprocess.run(ssh + [write_cmd], check=True)
    subprocess.run(ssh + [setup_cmd], check=True)
    subprocess.run(ssh + [run_cmd], check=True)

    attach = f"ssh -t -i {env['ssh_key']} -p {env['port']} {env['user']}@{env['host']} 'tmux attach -t {session}'"
    follow = f"./scripts/check_logs.sh -f   # (set LOCAL_HOST/LOCAL_PORT as POD_HOST/POD_PORT in .env)"
    print(f"\nLaunched '{session}' on {env['host']}.")
    print(f"  Attach : {attach}")
    print(f"  Follow : ssh -i {env['ssh_key']} -p {env['port']} {env['user']}@{env['host']} 'tail -f {env['code_dir']}/runs/_latest_startup.log'")


def list_sessions() -> None:
    env = _read_local_env()
    res = subprocess.run(
        _ssh_base(env) + ["tmux list-sessions 2>/dev/null || echo '(no sessions)'"],
        capture_output=True, text=True, check=False,
    )
    print(f"Sessions on {env['host']}:")
    print(res.stdout.rstrip())


def kill_session(name: str) -> None:
    env = _read_local_env()
    subprocess.run(
        _ssh_base(env) + [f"tmux kill-session -t {name}"], check=True,
    )
    print(f"Killed session '{name}' on {env['host']}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Launch training on a local GPU box over SSH.")
    parser.add_argument("--config", help="Path to pipeline YAML config")
    parser.add_argument("--steps", nargs="+", default=None,
                        help="Pipeline steps to run (default: all)")
    parser.add_argument("--session", default=None,
                        help="tmux session name (default: train_<run_name>)")
    parser.add_argument("--force", action="store_true",
                        help="Launch even if another training session is active")
    parser.add_argument("--list", action="store_true",
                        help="List active tmux sessions on the remote and exit")
    parser.add_argument("--kill", default=None,
                        help="Kill the named tmux session on the remote and exit")
    args = parser.parse_args()

    if args.list:
        list_sessions(); return
    if args.kill:
        kill_session(args.kill); return
    if not args.config:
        parser.error("--config is required (or use --list / --kill)")

    launch(args.config, steps=args.steps, session=args.session, force=args.force)


if __name__ == "__main__":
    main()
