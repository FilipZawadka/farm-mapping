# RunPod persistent storage (same data between runs)

By default, the launcher creates a new pod with ephemeral storage; when the pod is terminated, data is lost. To keep patches, config, and checkpoints between runs, use a **RunPod Network Volume** and attach it every time.

## 1. Create a Network Volume

1. Open [RunPod Console](https://www.runpod.io/console) and sign in.
2. Go to **Storage** → **Network Volumes**.
3. Click **Create Network Volume**.
4. Choose:
   - **Name** (e.g. `farm-mapping-data`)
   - **Size** (e.g. 100 GB; enough for patches + checkpoints)
   - **Data Center** (pick one where you usually run GPU pods, e.g. same as your GPU region).
5. Create the volume and copy the **Volume ID** (e.g. `abc123xyz`).

## 2. Attach the volume to your config

In your config YAML (e.g. `configs/us_egg_farms.yaml`), set the volume ID under `runpod`:

```yaml
runpod:
  gpu_type: "NVIDIA A40"
  docker_image: farm-mapping-train:latest
  volume_mount: /workspace/data
  api_key_env: RUNPOD_API_KEY
  network_volume_id: "YOUR_VOLUME_ID"   # paste the ID from step 1
```

When `network_volume_id` is set, the launcher attaches this volume to every new pod at `volume_mount` (`/workspace/data`). No ephemeral disk is created.

## 3. Put config and data on the volume (first run)

The container runs:

```text
python -m training.train --config /workspace/data/config.yaml
```

So the volume must contain at least:

- **`config.yaml`** — a copy of your resolved config (paths can be relative to `/workspace/data`).
- **Patch data** — e.g. `patches_us_egg/` with `candidates.parquet`, `patch_meta.parquet`, and the `.npy` files. Your config’s `patches.output_dir` should point to a path under `/workspace/data` (e.g. `data/patches_us_egg` → `/workspace/data/patches_us_egg` when the project root is `/workspace`).

Ways to populate the volume the first time:

**Option A: Run a one-off pod with the same volume**

1. Create a pod from the RunPod UI that uses your **same** Docker image and **attaches this Network Volume** at `/workspace/data`.
2. Start the pod, open the web terminal or Jupyter.
3. Copy your local `config.yaml` and patch directory into `/workspace/data` (e.g. via upload, or `scp`/`rsync` from your machine if you have SSH).
4. Stop the pod. The volume keeps the data.

**Option B: Build data inside the pod**

1. Launch a pod with the volume attached (e.g. with `runpod_launch` after setting `network_volume_id`).
2. In the RunPod template/startup, run candidates + patch extraction so that output goes under `/workspace/data` (and write `/workspace/data/config.yaml`). That requires Earth Engine and any other credentials to be available in the pod.
3. Then run training (manually or by setting the container command to do extraction then training).

**Option C: Use the data cache (S3/GCS) and sync on the pod**

If you use the pipeline’s cache with S3 or GCS, you can run extraction locally (or elsewhere), push to the cache, then on RunPod run a script that pulls from the cache into `/workspace/data` before starting training. The volume then holds the synced data for future runs.

## 4. Launch training

From your machine:

```bash
export RUNPOD_API_KEY="your-key"   # or use .env
python -m training.runpod_launch --config configs/us_egg_farms.yaml --wait
```

Every run will use the **same** network volume, so anything you left in `/workspace/data` (config, patches, previous checkpoints) is still there. Set `patches.output_dir` and other paths so they live under `/workspace/data` when the project root is `/workspace` in the container.

## Notes

- Network volumes are billed separately; check RunPod pricing.
- The volume must be in a data center that can be used with your GPU pods.
- If the RunPod Python SDK uses a different parameter name for the volume (e.g. `volume_id` vs `network_volume_id`), the launcher in `training/runpod_launch.py` passes it as `volume_id` to `create_pod()`; adjust if your SDK version differs.
