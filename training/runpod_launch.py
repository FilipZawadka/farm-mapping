"""RunPod API helper -- provision a GPU pod, launch training, download results.

Requires the ``runpod`` Python package and a valid API key (set via the env
var named in ``config.runpod.api_key_env``).

Usage::

    python -m training.runpod_launch --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from .config import PipelineConfig, RunPodConfig, load_config, resolve_paths

log = logging.getLogger(__name__)


def _get_api_key(cfg: RunPodConfig) -> str:
    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise EnvironmentError(
            f"RunPod API key not found. Set the {cfg.api_key_env} environment variable."
        )
    return key


_DOCKER_ARGS = "/bin/bash -lc 'eval $STARTUP_SCRIPT'"
"""RunPod/tini expects the first docker_args token to be an executable.
Use bash as the executable and pass a tiny command string to ``-lc``.
The real startup script lives in STARTUP_SCRIPT and is executed via eval."""


_CPU_IMAGE = "runpod/base:1.0.2-ubuntu2404"
"""RunPod base image (Ubuntu 24.04, Python 3.12) for CPU-only data-prep pods."""


def _build_clone_steps(cfg: PipelineConfig) -> list[str]:
    """Git clone/update steps shared by both prep and train startup scripts."""
    code_dir = getattr(cfg.runpod, "code_dir", "/workspace/farm-mapping")
    repo = getattr(cfg.runpod, "github_repo", "")
    branch = getattr(cfg.runpod, "github_branch", "main")
    parts: list[str] = []
    if repo:
        parts.append("apt-get update -qq && apt-get install -y -qq git")
        parts.append(
            f"git clone --branch {branch} --single-branch {repo} {code_dir}"
            f" || (cd {code_dir} && git fetch origin && git reset --hard origin/{branch})"
        )
    parts.append(f"cd {code_dir}")
    return parts


_LOAD_RUNPOD_ENV = (
    "while IFS= read -r -d $'\\0' _v; do"
    " case \"$_v\" in GEE_*|RUNPOD_*|GOOGLE_*) export \"$_v\" ;; esac;"
    " done < /proc/1/environ"
)
"""One-liner that exports RunPod secrets from the init process into the current shell.

Needed because tmux sessions don't inherit env vars set by the RunPod container runtime.
"""


def _build_prep_script(cfg: PipelineConfig, config_name: str) -> str:
    """Startup script for a CPU-only pod: candidates generation only.

    Skips venv creation and pip install when the venv already exists on the
    network volume (e.g. /workspace/farm-venv-cpu pre-installed).
    """
    code_dir = getattr(cfg.runpod, "code_dir", "/workspace/farm-mapping")
    venv = "/workspace/farm-venv-cpu"
    py = f"{venv}/bin/python"

    parts = [
        "set -euxo pipefail",
        _LOAD_RUNPOD_ENV,
        f"cd {code_dir}",
        f"echo '=== pod started, config={config_name} ==='",
    ]
    parts.extend(_build_clone_steps(cfg))
    parts.append(
        f"[ -d {venv} ]"
        f" && echo 'farm-venv-cpu found, skipping install'"
        f" || (echo 'farm-venv-cpu not found, installing...' && python -m venv {venv}"
        f" && {venv}/bin/pip install --no-cache-dir -r requirements-cpu.txt)"
    )
    parts.append(f"echo '=== running candidates ==='")
    parts.append(f"{py} -u -m training.candidates --config configs/{config_name}")
    parts.append(f"echo '=== DONE: candidates saved to {code_dir}/data/candidates/ ==='")
    if getattr(cfg.runpod, "auto_terminate", True):
        parts.append(
            f"{py} -c \"import runpod, os; runpod.api_key=os.environ['RUNPOD_API_KEY'];"
            f" runpod.terminate_pod(os.environ['RUNPOD_POD_ID']);"
            f" print('=== pod self-terminating ===')\""
        )
    return " && ".join(parts)


def _build_patch_script(cfg: PipelineConfig, config_name: str) -> str:
    """Startup script for a CPU-only pod: patch extraction only.

    Assumes candidates CSVs already exist on the network volume.
    """
    code_dir = getattr(cfg.runpod, "code_dir", "/workspace/farm-mapping")
    venv = "/workspace/farm-venv-cpu"
    py = f"{venv}/bin/python"

    parts = [
        "set -euxo pipefail",
        _LOAD_RUNPOD_ENV,
        f"cd {code_dir}",
        f"echo '=== pod started (patch extraction), config={config_name} ==='",
    ]
    parts.extend(_build_clone_steps(cfg))
    parts.append(
        f"[ -d {venv} ]"
        f" && echo 'farm-venv-cpu found, skipping install'"
        f" || (echo 'farm-venv-cpu not found, installing...' && python -m venv {venv}"
        f" && {venv}/bin/pip install --no-cache-dir -r requirements-cpu.txt)"
    )
    parts.append(f"echo '=== running patch extraction ==='")
    parts.append(f"{py} -u -m training.patch_extraction --config configs/{config_name}")
    parts.append(f"echo '=== DONE: patches saved to {code_dir}/data/patches/ ==='")
    if getattr(cfg.runpod, "auto_terminate", True):
        parts.append(
            f"{py} -c \"import runpod, os; runpod.api_key=os.environ['RUNPOD_API_KEY'];"
            f" runpod.terminate_pod(os.environ['RUNPOD_POD_ID']);"
            f" print('=== pod self-terminating ===')\""
        )
    return " && ".join(parts)


def _build_startup_script(cfg: PipelineConfig, config_name: str) -> str:
    """Startup script for a GPU pod: install deps, prep if needed, then train."""
    venv = "/workspace/farm-venv"
    py = f"{venv}/bin/python"

    parts = ["set -euxo pipefail"]
    parts.extend(_build_clone_steps(cfg))
    parts.append(f"[ -d {venv} ] || python -m venv {venv}")
    parts.append(f"{venv}/bin/pip install --no-cache-dir -r requirements-train.txt")
    parts.append(
        f"CAND_DIR=$({py} -c \"from training.config import "
        f"load_config, resolve_paths; cfg=resolve_paths(load_config('configs/{config_name}')); "
        "print(cfg.data.candidates_dir)\")"
    )
    parts.append(
        f"PATCHES_ROOT=$({py} -c \"from pathlib import Path; "
        f"from training.config import load_config, resolve_paths; "
        f"cfg=resolve_paths(load_config('configs/{config_name}')); "
        "od=Path(cfg.patches.output_dir); "
        "print(next((p for p in [od]+list(od.parents) if p.name=='patches'), od.parent))\")"
    )
    parts.append(
        "if ! ls \"$CAND_DIR\"/*.csv 1>/dev/null 2>&1 || [ ! -f \"$PATCHES_ROOT/patch_meta.csv\" ]; "
        f"then {py} -m training.candidates --config configs/{config_name} "
        f"&& {py} -m training.patch_extraction --config configs/{config_name}; "
        "fi"
    )
    parts.append(f"{py} -m training.train --config configs/{config_name}")
    return " && ".join(parts)


def _network_volume_id(cfg: PipelineConfig) -> str | None:
    return (
        getattr(cfg.runpod, "network_volume_id", None)
        or os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )


_RUNPOD_SECRETS_ENV: dict[str, str] = {
    "GEE_SERVICE_ACCOUNT": "{{ RUNPOD_SECRET_GEE_SERVICE_ACCOUNT }}",
    "GEE_PRIVATE_KEY_JSON": "{{ RUNPOD_SECRET_GEE_PRIVATE_KEY_JSON }}",
    "GOOGLE_MAPS_API_KEY": "{{ RUNPOD_SECRET_GOOGLE_MAPS_API_KEY }}",
}
"""Map of env vars → RunPod secret references injected into every pod."""


def _build_create_kwargs(cfg: PipelineConfig, gpu_type: str, config_name: str) -> dict:
    """Build the kwargs dict for runpod.create_pod (GPU training)."""
    volume_mount = cfg.runpod.volume_mount
    network_volume_id = _network_volume_id(cfg)
    cloud_type = getattr(cfg.runpod, "cloud_type", "ALL")

    kwargs: dict = {
        "name": "farm-detection-training",
        "image_name": cfg.runpod.docker_image,
        "gpu_type_id": gpu_type,
        "container_disk_in_gb": 20,
        "volume_mount_path": volume_mount,
        "docker_args": _DOCKER_ARGS,
        "env": {
            **_RUNPOD_SECRETS_ENV,
            "STARTUP_SCRIPT": _build_startup_script(cfg, config_name),
        },
    }
    if cloud_type != "ALL":
        kwargs["cloud_type"] = cloud_type
    if network_volume_id:
        kwargs["network_volume_id"] = network_volume_id
        kwargs["volume_in_gb"] = 0
    else:
        kwargs["volume_in_gb"] = 50
    return kwargs


def _build_prep_kwargs(cfg: PipelineConfig, config_name: str, instance_id: str) -> dict:
    """Build the kwargs dict for a CPU-only data-prep pod."""
    volume_mount = cfg.runpod.volume_mount
    network_volume_id = _network_volume_id(cfg)

    kwargs: dict = {
        "name": f"farm-prep-{config_name.removesuffix('.yaml')}",
        "image_name": _CPU_IMAGE,
        "gpu_count": 0,
        "instance_id": instance_id,
        "min_vcpu_count": 4,
        "min_memory_in_gb": 16,
        "container_disk_in_gb": 10,
        "volume_mount_path": volume_mount,
        "docker_args": _DOCKER_ARGS,
        "env": {
            **_RUNPOD_SECRETS_ENV,
            "RUNPOD_API_KEY": _get_api_key(cfg.runpod),
        },
    }
    if network_volume_id:
        kwargs["network_volume_id"] = network_volume_id
        kwargs["volume_in_gb"] = 0
    else:
        kwargs["volume_in_gb"] = 50
    return kwargs


def _init_runpod(cfg: PipelineConfig):
    import runpod
    runpod.api_key = _get_api_key(cfg.runpod)
    return runpod


def _wait_for_ssh(pod_id: str, runpod, timeout: int = 120) -> tuple[str, int]:
    """Poll until the pod exposes an SSH port. Returns (host, port)."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        for port_info in (pod.get("runtime") or {}).get("ports", []):
            if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                host = port_info["ip"]
                port = port_info["publicPort"]
                # Wait until TCP port actually accepts connections
                try:
                    with socket.create_connection((host, port), timeout=5):
                        return host, port
                except OSError:
                    pass
        time.sleep(10)
    raise TimeoutError(f"SSH not available on pod {pod_id} after {timeout}s")


def _ssh_run_startup(host: str, port: int, script: str) -> None:
    """SSH into the pod and run the startup script inside a detached tmux session."""
    import subprocess, base64
    remote_script = "/tmp/prep_startup.sh"
    # Encode as base64 to avoid all quoting/escaping issues over SSH
    script_b64 = base64.b64encode(script.encode()).decode()
    write_cmd = f"echo '{script_b64}' | base64 -d > {remote_script} && chmod +x {remote_script}"
    run_cmd = f"tmux new-session -d -s prep 'bash {remote_script} 2>&1 | tee /tmp/startup.log'"

    ssh_base = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "-p", str(port), f"root@{host}",
    ]
    log.info("Uploading startup script to pod (%s:%d) ...", host, port)
    subprocess.run(ssh_base + [write_cmd], check=True)
    subprocess.run(ssh_base + [run_cmd], check=True)
    log.info(
        "Script running in tmux session 'prep'.\n"
        "  Attach : ssh root@%s -p %d  →  tmux attach -t prep\n"
        "  Logs   : /tmp/startup.log",
        host, port,
    )


def launch_prep_pod(cfg: PipelineConfig, config_name: str = "us_egg_farms.yaml") -> dict:
    """Launch a CPU-only pod that generates farm candidates.

    Tries ``cpu_instance_id`` first, then each entry in ``cpu_fallbacks`` until
    a pod is created successfully. Because RunPod CPU pods ignore docker_args,
    the startup script is triggered via SSH once the pod is ready.
    """
    runpod = _init_runpod(cfg)

    cpu_candidates = [cfg.runpod.cpu_instance_id] + list(cfg.runpod.cpu_fallbacks)

    last_error = None
    for instance_id in cpu_candidates:
        log.info("Trying CPU instance %r (image=%s) ...", instance_id, _CPU_IMAGE)
        try:
            pod = runpod.create_pod(**_build_prep_kwargs(cfg, config_name, instance_id))
            log.info("Prep pod created: id=%s instance=%r", pod.get("id"), instance_id)
            break
        except runpod.error.QueryError as exc:
            log.warning("CPU instance %r unavailable: %s", instance_id, exc)
            last_error = exc
    else:
        raise RuntimeError(
            f"No CPU instances available. Tried: {cpu_candidates}"
        ) from last_error

    pod_id = pod["id"]
    startup_script = _build_prep_script(cfg, config_name)
    log.info("Waiting for SSH on pod %s ...", pod_id)
    host, port = _wait_for_ssh(pod_id, runpod)
    _ssh_run_startup(host, port, startup_script)
    log.info("To watch live:  ssh root@%s -p %d -t 'tmux attach -t prep'", host, port)
    return pod


def launch_patch_pod(cfg: PipelineConfig, config_name: str = "us_egg_farms.yaml") -> dict:
    """Launch a CPU-only pod that runs patch extraction.

    Assumes candidate CSVs already exist on the network volume.
    """
    runpod = _init_runpod(cfg)

    cpu_candidates = [cfg.runpod.cpu_instance_id] + list(cfg.runpod.cpu_fallbacks)

    last_error = None
    for instance_id in cpu_candidates:
        log.info("Trying CPU instance %r (image=%s) ...", instance_id, _CPU_IMAGE)
        try:
            kwargs = _build_prep_kwargs(cfg, config_name, instance_id)
            kwargs["name"] = f"farm-patches-{config_name.removesuffix('.yaml')}"
            pod = runpod.create_pod(**kwargs)
            log.info("Patch pod created: id=%s instance=%r", pod.get("id"), instance_id)
            break
        except runpod.error.QueryError as exc:
            log.warning("CPU instance %r unavailable: %s", instance_id, exc)
            last_error = exc
    else:
        raise RuntimeError(
            f"No CPU instances available. Tried: {cpu_candidates}"
        ) from last_error

    pod_id = pod["id"]
    startup_script = _build_patch_script(cfg, config_name)
    log.info("Waiting for SSH on pod %s ...", pod_id)
    host, port = _wait_for_ssh(pod_id, runpod)
    _ssh_run_startup(host, port, startup_script)
    log.info(
        "Script running in tmux session 'prep'.\n"
        "  Attach : ssh root@%s -p %d  ->  tmux attach -t prep\n"
        "  Logs   : /tmp/startup.log",
        host, port,
    )
    return pod


def launch_pod(cfg: PipelineConfig, config_name: str = "us_egg_farms.yaml") -> dict:
    """Provision a RunPod GPU pod and start the training container.

    Tries ``gpu_type`` first, then each entry in ``gpu_fallbacks`` until a pod
    is created successfully. Returns the pod info dict from the RunPod API.
    """
    runpod = _init_runpod(cfg)

    gpu_candidates = [cfg.runpod.gpu_type] + list(getattr(cfg.runpod, "gpu_fallbacks", []))

    last_error = None
    for gpu_type in gpu_candidates:
        log.info("Trying GPU %s (image=%s) ...", gpu_type, cfg.runpod.docker_image)
        try:
            pod = runpod.create_pod(**_build_create_kwargs(cfg, gpu_type, config_name))
            log.info("Pod created: id=%s gpu=%s status=%s",
                     pod.get("id"), gpu_type, pod.get("desiredStatus"))
            return pod
        except runpod.error.QueryError as exc:
            last_error = exc
            log.warning("GPU %s unavailable: %s", gpu_type, exc)

    raise RuntimeError(
        f"No instances available for any of {gpu_candidates}. "
        "Try different GPU types or wait and retry."
    ) from last_error


def wait_for_completion(pod_id: str, cfg: PipelineConfig, poll_interval: int = 60) -> dict:
    """Poll RunPod until the pod completes or fails."""
    runpod = _init_runpod(cfg)

    log.info("Waiting for pod %s to finish ...", pod_id)
    while True:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "UNKNOWN")
        runtime_status = pod.get("runtime", {}).get("status", "UNKNOWN") if pod.get("runtime") else "UNKNOWN"
        log.info("  Pod %s: desired=%s runtime=%s", pod_id, status, runtime_status)

        if runtime_status in ("EXITED", "COMPLETED"):
            log.info("Pod finished.")
            return pod
        if runtime_status in ("ERROR", "FAILED"):
            log.error("Pod failed: %s", pod)
            return pod
        if status == "TERMINATED":
            log.info("Pod terminated.")
            return pod

        time.sleep(poll_interval)


def terminate_pod(pod_id: str, cfg: PipelineConfig) -> None:
    """Terminate a RunPod pod."""
    runpod = _init_runpod(cfg)
    runpod.terminate_pod(pod_id)
    log.info("Terminated pod %s", pod_id)


def main() -> None:
    from .env_loader import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Launch training on RunPod")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--wait", action="store_true", help="Wait for pod to finish")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prep", action="store_true",
        help="CPU pod: generate candidates only")
    mode.add_argument("--patches", action="store_true",
        help="CPU pod: run patch extraction (candidates must already exist)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    config_name = os.path.basename(args.config)

    if args.prep:
        pod = launch_prep_pod(cfg, config_name=config_name)
    elif args.patches:
        pod = launch_patch_pod(cfg, config_name=config_name)
    else:
        pod = launch_pod(cfg, config_name=config_name)

    pod_id = pod.get("id")

    if args.wait and pod_id:
        result = wait_for_completion(pod_id, cfg)
        terminate_pod(pod_id, cfg)
        print(f"Pod finished with status: {result.get('runtime', {}).get('status', 'unknown')}")
    else:
        print(f"Pod launched: {pod_id}")
        print("Use --wait to block until training completes.")


if __name__ == "__main__":
    main()
