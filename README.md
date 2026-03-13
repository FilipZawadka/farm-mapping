# Farm Mapping — Satellite-based Farm Detection

Detect livestock farms from Sentinel-2 satellite imagery using a ResNet-50 classifier trained on labelled candidates from Farm Transparency Project and OpenStreetMap.

## Project Structure

```
farm-mapping/
├── configs/                  # YAML pipeline configs (one per experiment)
│   ├── us_egg_farms.yaml     # US egg/layer farms — main config
│   ├── us_chicken_meat.yaml  # US broiler farms
│   ├── thailand_chicken_meat.yaml
│   ├── smoke_test.yaml       # tiny config for quick local testing
│   └── default.yaml
├── training/                 # all pipeline code
│   ├── candidates.py         # step 1: build labelled candidate set
│   ├── patch_extraction.py   # step 2: extract Sentinel-2 patches via Earth Engine
│   ├── train.py              # step 3: train ResNet-50 classifier
│   ├── inference.py          # step 4: run inference on new candidates
│   ├── visualize.py          # step 5: generate prediction maps
│   ├── config.py             # Pydantic config models
│   ├── dataset.py            # PyTorch dataset + train/val/test splits
│   ├── model.py              # model builder (HuggingFace hub)
│   ├── env_loader.py         # .env / secrets loading
│   ├── runpod_launch.py      # RunPod pod launcher (GPU + CPU modes)
│   ├── osm_negatives.py      # OSM-based negative sampling
│   ├── osm_farm_finder.py    # OSM farm discovery
│   ├── imagery/              # imagery providers (Sentinel-2, Sentinel-1)
│   └── storage/              # cache backends (local, S3, GCS, RunPod volume)
├── src/                      # shared utilities
│   ├── config.py             # country definitions + Earth Engine init
│   ├── data_sources.py       # load known farms from CSVs
│   └── detection.py          # Sentinel-2 compositing helpers
├── notebooks/                # exploration / analysis notebooks
├── data/                     # local data (gitignored)
├── requirements-train.txt    # GPU training deps (torch, mlflow, etc.)
├── requirements-cpu.txt      # CPU-only deps (candidates + patch extraction)
├── .env.example              # template for secrets
└── .env                      # your secrets (gitignored)
```

## Quick Start (Local)

### 1. Environment Setup

```bash
# Create conda env (or venv)
conda create -n farm-mapping python=3.11 -y
conda activate farm-mapping
pip install -r requirements-train.txt

# Copy and fill in secrets
cp .env.example .env
# Edit .env with your keys (see "Secrets" section below)
```

### 2. Generate Training Data

```bash
# Build candidate set (positives from Farm Transparency + negatives from OSM)
python -m training.candidates --config configs/us_egg_farms.yaml

# Extract Sentinel-2 image patches via Earth Engine
python -m training.patch_extraction --config configs/us_egg_farms.yaml

# For a quick test, limit to 50 patches:
python -m training.patch_extraction --config configs/us_egg_farms.yaml --max-patches 50
```

### 3. Train

```bash
python -m training.train --config configs/us_egg_farms.yaml
```

Training logs go to MLflow (`./mlruns`). View them with:

```bash
mlflow ui
```

### 4. Inference + Visualization

```bash
python -m training.inference --config configs/us_egg_farms.yaml
python -m training.visualize --config configs/us_egg_farms.yaml
```

## RunPod (Cloud GPU)

### Prerequisites

1. A [RunPod](https://www.runpod.io) account with API key
2. A **Network Volume** for persistent storage between pods
3. Secrets configured (see below)

### Secrets Setup

Go to **RunPod Console → Settings → Secrets** and create:

| Secret Name | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT` | Your GEE service account email |
| `GEE_PRIVATE_KEY_JSON` | Base64-encoded GEE JSON key (see below) |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key (optional) |

To base64-encode your key:

```bash
base64 -w0 /path/to/your-service-account-key.json
```

Then in your pod template's **Environment Variables**, add:

| Key | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT` | `{{ RUNPOD_SECRET_GEE_SERVICE_ACCOUNT }}` |
| `GEE_PRIVATE_KEY_JSON` | `{{ RUNPOD_SECRET_GEE_PRIVATE_KEY_JSON }}` |
| `GOOGLE_MAPS_API_KEY` | `{{ RUNPOD_SECRET_GOOGLE_MAPS_API_KEY }}` |

### Option A: Automated Launch (from your machine)

```bash
# Step 1 — Generate data on a cheap CPU pod
python -m training.runpod_launch --config configs/us_egg_farms.yaml --prep-only --wait

# Step 2 — Train on a GPU pod (picks up data from network volume)
python -m training.runpod_launch --config configs/us_egg_farms.yaml --wait
```

### Option B: Manual SSH on RunPod

**Data prep (CPU pod):**

```bash
git config --global --add safe.directory /workspace/farm-mapping
cd /workspace/farm-mapping
git fetch origin && git reset --hard origin/main

python3 -m venv /workspace/farm-venv-cpu
/workspace/farm-venv-cpu/bin/pip install --no-cache-dir -r requirements-cpu.txt

/workspace/farm-venv-cpu/bin/python -m training.candidates --config configs/us_egg_farms.yaml
/workspace/farm-venv-cpu/bin/python -m training.patch_extraction --config configs/us_egg_farms.yaml
```

**Training (GPU pod):**

```bash
git config --global --add safe.directory /workspace/farm-mapping
cd /workspace/farm-mapping
git fetch origin && git reset --hard origin/main

python3 -m venv /workspace/farm-venv
/workspace/farm-venv/bin/pip install --no-cache-dir -r requirements-train.txt

/workspace/farm-venv/bin/python -m training.train --config configs/us_egg_farms.yaml
```

### RunPod Images

| Purpose | Image |
|---|---|
| Data prep (CPU) | `runpod/base:1.0.2-ubuntu2404` |
| Training (GPU) | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

## Secrets Reference

| Variable | Where | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | `.env` only | RunPod API access |
| `RUNPOD_NETWORK_VOLUME_ID` | `.env` only | Persistent storage volume ID |
| `GEE_SERVICE_ACCOUNT` | `.env` + RunPod Secret | Earth Engine service account email |
| `GEE_KEY_FILE` | `.env` only (local) | Path to GEE JSON key file |
| `GEE_PRIVATE_KEY_JSON` | RunPod Secret | GEE JSON key (raw or base64) |
| `GOOGLE_MAPS_API_KEY` | `.env` + RunPod Secret | Google Maps / Places API key |

## Pipeline Steps

```
candidates.py → patch_extraction.py → train.py → inference.py → visualize.py
     │                  │                 │             │              │
     ▼                  ▼                 ▼             ▼              ▼
  candidates.parquet  patch_meta.parquet  best_model.pt  scored.parquet  maps/
  (pos + neg labels)  + .npy patches     + mlruns/
```

1. **Candidates** — loads known farms (Farm Transparency CSVs) as positives, samples negatives from OSM buildings (warehouses, hangars, etc.), assigns geographic regions for train/val/test splits.

2. **Patch Extraction** — for each candidate, fetches a Sentinel-2 median composite (6 bands + 3 indices = 9 channels) from Google Earth Engine and saves as `.npy`.

3. **Training** — ResNet-50 (pretrained, from HuggingFace) fine-tuned as a binary farm/not-farm classifier. Logs metrics to MLflow. Supports mixed precision, cosine LR schedule, early stopping.

4. **Inference** — runs the trained model on patches and outputs scored candidates.

5. **Visualization** — generates interactive HTML maps of predictions.

## Config Files

Each YAML config defines the full pipeline: data regions, patch settings, model architecture, training hyperparameters, and RunPod settings. See `configs/us_egg_farms.yaml` for the most complete example.

To create a new experiment, copy an existing config and adjust the regions, species filter, and output paths.
