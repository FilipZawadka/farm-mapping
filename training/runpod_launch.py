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


def _build_prep_script(cfg: PipelineConfig, config_name: str) -> str:
    """Startup script for a CPU-only pod: candidates + patch extraction."""
    code_dir = getattr(cfg.runpod, "code_dir", "/workspace/farm-mapping")
    venv = "/workspace/farm-venv-cpu"
    py = f"{venv}/bin/python"

    parts = ["set -euxo pipefail"]
    parts.extend(_build_clone_steps(cfg))
    parts.append(f"[ -d {venv} ] || python -m venv {venv}")
    parts.append(f"{venv}/bin/pip install --no-cache-dir -r requirements-cpu.txt")
    parts.append(f"{py} -m training.candidates --config configs/{config_name}")
    parts.append(f"{py} -m training.patch_extraction --config configs/{config_name}")
    parts.append(f"echo '=== data prep finished, patches in {code_dir}/data/ ==='")
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
        f"PATCH_DIR=$({py} -c \"from training.config import "
        f"load_config, resolve_paths; cfg=resolve_paths(load_config('configs/{config_name}')); "
        "print(cfg.patches.output_dir)\")"
    )
    parts.append(
        "if [ ! -f \"$PATCH_DIR/candidates.parquet\" ] || [ ! -f \"$PATCH_DIR/patch_meta.parquet\" ]; "
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


def _build_prep_kwargs(cfg: PipelineConfig, config_name: str) -> dict:
    """Build the kwargs dict for a CPU-only data-prep pod."""
    volume_mount = cfg.runpod.volume_mount
    network_volume_id = _network_volume_id(cfg)

    kwargs: dict = {
        "name": "farm-data-prep",
        "image_name": _CPU_IMAGE,
        "gpu_type_id": "NVIDIA GeForce RTX 4090",
        "gpu_count": 0,
        "container_disk_in_gb": 10,
        "volume_mount_path": volume_mount,
        "docker_args": _DOCKER_ARGS,
        "env": {
            **_RUNPOD_SECRETS_ENV,
            "STARTUP_SCRIPT": _build_prep_script(cfg, config_name),
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


def launch_prep_pod(cfg: PipelineConfig, config_name: str = "us_egg_farms.yaml") -> dict:
    """Launch a cheap CPU-only pod that generates candidates + patches.

    The resulting data lives on the network volume so a later GPU training
    pod can pick it up without re-extracting.
    """
    runpod = _init_runpod(cfg)
    log.info("Launching CPU data-prep pod (image=%s) ...", _CPU_IMAGE)
    pod = runpod.create_pod(**_build_prep_kwargs(cfg, config_name))
    log.info("Prep pod created: id=%s", pod.get("id"))
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
    parser.add_argument(
        "--prep-only", action="store_true",
        help="Launch a cheap CPU pod that only generates candidates + patches",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    config_name = os.path.basename(args.config)

    if args.prep_only:
        pod = launch_prep_pod(cfg, config_name=config_name)
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
