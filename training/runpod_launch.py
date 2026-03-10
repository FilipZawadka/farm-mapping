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


_DOCKER_ARGS = (
    "python3 -c "
    "__import__('os').execvp('bash',['bash','-c',__import__('os').environ['STARTUP_SCRIPT']])"
)
"""RunPod/tini splits docker_args by whitespace with no shell-style quoting,
so ``bash -c "script with spaces"`` is impossible.  Instead we pass the real
script through the STARTUP_SCRIPT env-var and use a space-free Python
one-liner that execs bash with the correct argv."""


def _build_startup_script(cfg: PipelineConfig, config_name: str) -> str:
    """Build the bash script that will be stored in the STARTUP_SCRIPT env var."""
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
    parts.append("pip install --no-cache-dir -r requirements-train.txt")
    parts.append(f"python -m training.train --config configs/{config_name}")
    return " && ".join(parts)


def _build_create_kwargs(cfg: PipelineConfig, gpu_type: str, config_name: str) -> dict:
    """Build the kwargs dict for runpod.create_pod."""
    volume_mount = cfg.runpod.volume_mount
    network_volume_id = (
        getattr(cfg.runpod, "network_volume_id", None)
        or os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )
    cloud_type = getattr(cfg.runpod, "cloud_type", "ALL")

    kwargs: dict = {
        "name": "farm-detection-training",
        "image_name": cfg.runpod.docker_image,
        "gpu_type_id": gpu_type,
        "container_disk_in_gb": 20,
        "volume_mount_path": volume_mount,
        "docker_args": _DOCKER_ARGS,
        "env": {"STARTUP_SCRIPT": _build_startup_script(cfg, config_name)},
    }
    if cloud_type != "ALL":
        kwargs["cloud_type"] = cloud_type
    if network_volume_id:
        kwargs["network_volume_id"] = network_volume_id
        kwargs["volume_in_gb"] = 0
    else:
        kwargs["volume_in_gb"] = 50
    return kwargs


def launch_pod(cfg: PipelineConfig, config_name: str = "us_egg_farms.yaml") -> dict:
    """Provision a RunPod GPU pod and start the training container.

    Tries ``gpu_type`` first, then each entry in ``gpu_fallbacks`` until a pod
    is created successfully. Returns the pod info dict from the RunPod API.
    """
    try:
        import runpod  # noqa: F811
    except ImportError:
        log.error("'runpod' package not installed. Run: pip install runpod")
        raise

    api_key = _get_api_key(cfg.runpod)
    runpod.api_key = api_key

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
    try:
        import runpod
    except ImportError:
        raise

    api_key = _get_api_key(cfg.runpod)
    runpod.api_key = api_key

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
    try:
        import runpod
    except ImportError:
        raise

    api_key = _get_api_key(cfg.runpod)
    runpod.api_key = api_key
    runpod.terminate_pod(pod_id)
    log.info("Terminated pod %s", pod_id)


def main() -> None:
    from .env_loader import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Launch training on RunPod")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--wait", action="store_true", help="Wait for pod to finish")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))

    config_name = os.path.basename(args.config)
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
