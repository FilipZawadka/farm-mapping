# 10 — Deployment (RunPod)

Every substantive training run happens on a RunPod GPU pod with our shared
network volume mounted. The launcher, startup script, and self-heal git
path all live in
[training/runpod_launch.py](../../training/runpod_launch.py).

## Entry point

```bash
python -m training.runpod_launch --config configs/rachel_clusters/<name>.yaml
```

Optional flags:

- `--prep` — CPU pod that only runs the candidates step.
- `--patches` — CPU pod that runs candidates + patch_extraction.
- `--steps train inference visualize` — restrict to specific pipeline
  steps.
- `--wait` — block until pod terminates (else launcher returns
  immediately after the pod is running).

## What the launcher does

```
1. Parse config; pick gpu_type from cfg.runpod.gpu_type + fallbacks.
2. For each candidate GPU:
     runpod.create_pod(**_build_create_kwargs)
     - name = "farm-train-<config_stem>"
     - image_name = cfg.runpod.docker_image
     - gpu_type_id = candidate
     - container_disk_in_gb = 20
     - volume_mount_path = /workspace  (cfg.runpod.volume_mount)
     - network_volume_id = cfg.runpod.network_volume_id
     - support_public_ip = True
     - env = { GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY_JSON /
               GOOGLE_MAPS_API_KEY }  (RunPod secret refs)
   Retry on QueryError until one succeeds or the list is exhausted.
3. Wait for SSH on the pod (typically 10-30 s).
4. Build startup script (see below).
5. Upload the script to /tmp/prep_startup.sh via SSH (base64-encoded).
6. Start a detached tmux session `prep` running the script under stdbuf
   tee → /tmp/startup.log + /workspace/farm-mapping/runs/_latest_startup.log.
7. Return immediately (or block for --wait).
```

## Startup script

Assembled by `_build_startup_script`:

```bash
set -uxo pipefail
_on_err() { echo "SCRIPT FAILED (exit $?)"; ... }
trap '_on_err' ERR
set -e

# Load RunPod secrets from /proc/1/environ
while IFS= read -r -d $'\0' _v; do
  case "$_v" in GEE_*|RUNPOD_*|GOOGLE_*) export "$_v" ;; esac
done < /proc/1/environ

git config --global --add safe.directory /workspace/farm-mapping
cd /workspace/farm-mapping

# --- Self-heal git ---
# Try normal fetch+reset; on failure (corrupted .git, missing .git, or
# disk-quota exceeded during fetch), rebuild .git via /tmp and mv onto
# the volume.
(cd /workspace/farm-mapping && git fetch origin \
  && git reset --hard origin/$(git symbolic-ref --short HEAD 2>/dev/null || echo main)) \
|| (echo 're-cloning /workspace/farm-mapping from <repo> via /tmp' \
  && rm -rf /workspace/farm-mapping/.git /tmp/__repo_tmp \
  && git clone --branch main --single-branch --no-checkout <repo> /tmp/__repo_tmp \
  && mv /tmp/__repo_tmp/.git /workspace/farm-mapping/.git \
  && rm -rf /tmp/__repo_tmp \
  && cd /workspace/farm-mapping && git reset --hard origin/main)

# Create runs/<config>/pipeline/<run_name>_<ts>/  for logs
export RUN_DIR=/workspace/farm-mapping/runs/<config>/pipeline/<run_name>_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

# venv check — reuse if present
[ -d /workspace/farm-venv ] \
  && echo 'farm-venv found, skipping install' \
  || (python -m venv /workspace/farm-venv \
      && /workspace/farm-venv/bin/pip install --no-cache-dir -r requirements-train.txt)

# Auto-rebuild master parquet if the config references all_clusters_v4 and
# the file is missing (uses data_seed/ for BGD/NGA).
if [ ! -f data/rachel_geometry_candidates/all_countries/all_clusters_v4.parquet ] \
    && grep -q all_clusters_v4 configs/<config>; then
  /workspace/farm-venv/bin/python scripts/merge_clusters_v4.py
fi

echo '=== running pipeline ==='
/workspace/farm-venv/bin/python -u -m training.run_pipeline \
    --config configs/<config> \
    [--steps <list>]
echo '=== DONE ==='

# Always try to auto-terminate (uses ; so this runs even on pipeline failure)
; /workspace/farm-venv/bin/python -m training.auto_terminate
```

The `; auto_terminate` at the end fires regardless of pipeline success or
failure. Pod termination is via RunPod API from
[training/auto_terminate.py](../../training/auto_terminate.py).

## Two-phase secret injection

Secrets are declared at pod-creation time as RunPod secret references:

```python
_RUNPOD_SECRETS_ENV = {
    "GEE_SERVICE_ACCOUNT": "{{ RUNPOD_SECRET_GEE_SERVICE_ACCOUNT }}",
    "GEE_PRIVATE_KEY_JSON": "{{ RUNPOD_SECRET_GEE_PRIVATE_KEY_JSON }}",
    "GOOGLE_MAPS_API_KEY": "{{ RUNPOD_SECRET_GOOGLE_MAPS_API_KEY }}",
}
```

RunPod expands them into the container's PID-1 environ. The startup
script's first non-preamble command reads `/proc/1/environ` to hoist them
into the tmux shell (tmux doesn't inherit RunPod's injected env).

Then [training/env_loader.py](../../training/env_loader.py)
`get_gee_credentials` materializes `GEE_PRIVATE_KEY_JSON` (raw or
base64) into a temp file the S2 provider reads.

## Directory layout on the pod

```
/workspace/
├── farm-mapping/                   # network volume
│   ├── .git/                       # commit tracked on the volume
│   ├── configs/
│   ├── training/
│   ├── data/
│   │   ├── patches/                # ~140k .npy + patch_meta.csv
│   │   ├── output/                 # per-config: best_model.pt + JSONs
│   │   └── rachel_geometry_candidates/all_countries/
│   ├── output/                     # per-config maps HTML
│   ├── mlruns/                     # MLflow store
│   └── runs/                       # per-run log archives
│       ├── _latest_startup.log     # last pod's startup log (0-byte if boot failed)
│       └── <config_stem>/pipeline/<run_name>_<ts>/
│           ├── candidates.log
│           ├── patch_extraction.log
│           ├── train.log
│           ├── inference.log
│           ├── pipeline.log        # step transitions
│           └── config.yaml         # snapshot
├── farm-venv/                      # GPU venv (torch + cuda)
└── farm-venv-cpu/                  # CPU venv (no torch/cuda)
```

## Relay pod

Independent of the training pods, a small CPU pod with just the network
volume mounted is very useful:

- Tail live logs while training runs elsewhere.
- Run `scripts/post_hoc_evaluate.py` on an existing best_model.pt.
- Serve the MLflow UI (`mlflow ui --port 5000 --backend-store-uri ./mlruns`)
  behind an SSH tunnel.
- Clean up disk quota, rename artefacts, etc.

Launch manually (bypassing the config-driven launcher):

```python
runpod.create_pod(
    name="farm-relay",
    image_name="runpod/base:1.0.2-ubuntu2404",
    instance_id="cpu3g-4-16",
    volume_mount_path="/workspace",
    network_volume_id="r8nyom4e4e",
    ports="22/tcp",
    support_public_ip=True,
)
```

Then keep the SSH details in `POD_HOST` / `POD_PORT` in your `.env`.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Pod created, tmux session runs, `_latest_startup.log` is 0 bytes | Volume disk quota exceeded → `git fetch` fails silently. Self-heal path rebuilds .git into /tmp, then can't `mv` back due to quota. | Free space (delete `last_ckpt.pt`, mlflow artifacts). Then re-launch. |
| All 8 SECURE GPUs unavailable | Peak demand. | Retry in 5-15 min; the launcher polls when wrapped in `until ... 'Pod launched:'; do sleep 300; done`. |
| Pod terminated within minutes of `Step: train` | Community-cloud preemption. | Set `cloud_type: SECURE`. |
| Fresh pod says `GEE key file not found: /root/gee-key.json` | RunPod secret not configured for the account. | Add `GEE_PRIVATE_KEY_JSON` (raw or base64) as a RunPod secret. |
| Pod dies at `mlflow.log_artifact(best_path)` | 270 MB artifact eats remaining quota. | Set `mlflow.log_model: false` or free 300+ MB. |

## Terminating pods

Auto-terminate is on by default. Force-terminate:

```python
import runpod
runpod.terminate_pod(pod_id)
```

Or find all running pods:

```python
for p in runpod.get_pods():
    if p["desiredStatus"] == "RUNNING":
        print(p["id"], p["name"])
```

## Startup script tuning knobs

Everything in `_build_startup_script` is a plain Python string, so tweaks
are trivial:

- Skip the venv install for a specific run: comment out the venv line.
- Pin a git ref: replace `git reset --hard origin/main` with a commit SHA.
- Run additional shell commands: append to the `parts` list before the
  final pipeline invocation.
