# Farm Mapping — Satellite-based Farm Detection

Detect livestock farms from Sentinel-2 satellite imagery using a ResNet-50 classifier trained on labelled candidates from Farm Transparency Project and OpenStreetMap.

## Project Structure

```
farm-mapping/
├── configs/                          # YAML pipeline configs (one per experiment)
│   ├── global_all_farms.yaml         # Multi-country, all farm types
│   ├── chicken_eggs_united_states.yaml # US egg/layer farms
│   ├── all_countries_candidates.yaml # Candidate generation for all countries
│   ├── smoke_test.yaml               # Tiny config for quick local testing
│   └── default.yaml
├── training/                         # All pipeline code
│   ├── candidates.py                 # Step 1: build labelled candidate set
│   ├── patch_extraction.py           # Step 2: extract Sentinel-2 patches via Earth Engine
│   ├── train.py                      # Step 3: train ResNet-50 classifier
│   ├── inference.py                  # Step 4: run inference on candidates
│   ├── visualize.py                  # Step 5: generate prediction maps
│   ├── run_pipeline.py              # Orchestrator: runs all steps in sequence
│   ├── runpod_launch.py             # RunPod pod launcher (GPU + CPU modes)
│   ├── auto_terminate.py            # Self-terminate RunPod pod after completion
│   ├── config.py                     # Pydantic config models
│   ├── dataset.py                    # PyTorch dataset + train/val/test splits
│   ├── model.py                      # Model builder (HuggingFace hub)
│   ├── env_loader.py                 # .env / secrets loading
│   ├── osm_negatives.py              # OSM-based negative sampling
│   ├── osm_farm_finder.py            # OSM farm discovery
│   ├── imagery/                      # Imagery providers (Sentinel-2, Sentinel-1)
│   └── storage/                      # Cache backends (local, S3, GCS, RunPod volume)
├── src/                              # Shared utilities
│   ├── config.py                     # Country definitions + Earth Engine init
│   ├── data_sources.py               # Load known farms from CSVs
│   └── detection.py                  # Sentinel-2 compositing helpers
├── scripts/
│   └── reorganize_patches.py         # One-off: migrate patches to {country}/{state} layout
├── data/                             # Local data (gitignored)
├── requirements-train.txt            # GPU training deps (torch, mlflow, etc.)
├── requirements-cpu.txt              # CPU-only deps (candidates + patch extraction)
├── .env.example                      # Template for secrets
└── .env                              # Your secrets (gitignored)
```

## Supported Countries

| Country | Key | Farm Transparency Data | Role |
|---------|-----|----------------------|------|
| United States | `united_states` | 30k+ facilities | Train/Val/Test (by state) |
| Thailand | `thailand` | 2,100 facilities | Train |
| Brazil | `brazil` | 1,000 facilities | Train |
| Mexico | `mexico` | 177 facilities | Train |
| Chile | `chile` | 229 facilities | Train |
| United Kingdom | `united_kingdom` | 1,000 facilities | Validation |
| Australia | `australia` | 6,400 facilities | Validation |
| Argentina | `argentina` | 11,000 facilities | Test (held out) |
| Canada | `canada` | 560 facilities | Test (held out) |

## Quick Start (Local)

### 1. Environment Setup

```bash
conda create -n farm-mapping python=3.11 -y
conda activate farm-mapping
pip install -r requirements-train.txt

cp .env.example .env
# Edit .env with your keys (see "Secrets" section below)
```

### 2. Run the Full Pipeline

```bash
python -m training.run_pipeline --config configs/global_all_farms.yaml
```

This runs: candidates → patch_extraction → train → inference → visualize.

Skip steps that are already done:

```bash
python -m training.run_pipeline --config configs/global_all_farms.yaml --skip candidates patch_extraction
```

### 3. Or Run Steps Individually

```bash
python -m training.candidates --config configs/global_all_farms.yaml
python -m training.patch_extraction --config configs/global_all_farms.yaml
python -m training.train --config configs/global_all_farms.yaml
python -m training.inference --config configs/global_all_farms.yaml
python -m training.visualize --config configs/global_all_farms.yaml
```

### 4. View Training Metrics

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

## RunPod (Cloud GPU)

### Prerequisites

1. A [RunPod](https://www.runpod.io) account with API key
2. A **Network Volume** (ID in config under `runpod.network_volume_id`) for persistent storage
3. Secrets configured in RunPod Console (see below)

### Three-Step Launch

```bash
# Step 1 — Generate candidates on a CPU pod (~2 min)
python -m training.runpod_launch --config configs/global_all_farms.yaml --prep

# Step 2 — Extract satellite patches on a CPU pod (~2-4 hours)
python -m training.runpod_launch --config configs/global_all_farms.yaml --patches

# Step 3 — Train + inference + visualize on a GPU pod
python -m training.runpod_launch --config configs/global_all_farms.yaml
```

Each step launches a pod, runs via SSH+tmux (CPU pods) or docker_args (GPU pods), and auto-terminates when done (configurable via `runpod.auto_terminate` in the config).

### Monitoring

CPU pods (candidates, patches):
```bash
# Attach to the live tmux session
ssh -t root@<host> -p <port> 'tmux attach -t prep'

# Or check the log
ssh root@<host> -p <port> 'tail -20 /tmp/startup.log'
```

GPU pods (training):
```bash
# Check RunPod web console → Logs tab
# Or SSH via RunPod proxy
ssh <pod_host_id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Secrets Setup

Go to **RunPod Console → Settings → Secrets** and create:

| Secret Name | Value |
|---|---|
| `GEE_SERVICE_ACCOUNT` | Your GEE service account email |
| `GEE_PRIVATE_KEY_JSON` | Raw GEE JSON key content |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key (optional) |

These are automatically injected into pods via RunPod's secret system.

### RunPod Images

| Purpose | Image |
|---|---|
| Data prep (CPU) | `runpod/base:1.0.2-ubuntu2404` |
| Training (GPU) | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |

### CPU Instance Types

| Flavor | ID Example | Description |
|--------|-----------|-------------|
| General Purpose | `cpu3g-4-16` | 4 vCPU, 16GB RAM |
| Compute-Optimized | `cpu3c-8-16` | 8 vCPU, 16GB RAM |
| Memory-Optimized | `cpu3m-4-32` | 4 vCPU, 32GB RAM |

Format: `{flavor}-{vcpu}-{memoryGb}`. Configure in `runpod.cpu_instance_id` with fallbacks in `runpod.cpu_fallbacks`.

## Pipeline Steps

```
candidates → patch_extraction → train → inference → visualize
    │              │                │         │           │
    ▼              ▼                ▼         ▼           ▼
 CSV files    patch_meta.csv   best_model.pt  scored     prediction_map.html
 (pos+neg)    + .npy patches   + mlruns/      .parquet
```

1. **Candidates** — Loads known farms from Farm Transparency CSVs as positives. Samples negatives via `random_rural` (random points away from farms) or `osm_buildings` (warehouses, hangars, etc.). Assigns geographic regions for train/val/test splits.

2. **Patch Extraction** — For each candidate, fetches a Sentinel-2 median composite (6 bands + 3 spectral indices = 9 channels, 128x128px at 10m resolution) from Google Earth Engine. Saves as `.npy` files. Resumes from `patch_meta.csv` — already-extracted patches are skipped across configs.

3. **Training** — ResNet-50 (pretrained from HuggingFace) fine-tuned as a binary farm/not-farm classifier. Supports mixed precision, cosine LR schedule, backbone freezing, class weighting, and early stopping. Logs to MLflow.

4. **Inference** — Runs the trained model on all candidates. Outputs `scored_candidates.parquet` with predictions, confidence scores, and split assignments.

5. **Visualization** — Generates an interactive Leaflet HTML map with layers for TP/FP/FN/TN, filterable by train/val/test split.

## Data Layout on Network Volume

```
/workspace/farm-mapping/
├── data/
│   ├── candidates/{config_name}/          # One CSV per country
│   │   └── united_states.csv
│   ├── patches/                           # Shared patch store
│   │   ├── patch_meta.csv                 # Global registry (all configs share this)
│   │   ├── split_assignments.csv          # train/val/test per candidate
│   │   ├── scored_candidates.parquet      # Latest inference predictions
│   │   └── {country}/{state}/{hash}/*.npy # Actual patch files
│   ├── cache/                             # OSM query caches
│   ├── farms/                             # Merged farm CSVs
│   └── farm_transparency_maps/            # Raw source CSVs
├── output/
│   ├── best_model.pt                      # Latest model checkpoint
│   └── maps_{config}/prediction_map.html  # Latest prediction map
├── mlruns/                                # MLflow experiment tracking
│   └── {experiment_id}/{run_id}/
│       ├── params/                        # Hyperparameters
│       ├── metrics/                       # Per-epoch metrics
│       └── artifacts/                     # Model + test metrics JSON
└── runs/                                  # Persistent run logs
    └── {config_name}/
        ├── latest -> (symlink to most recent run)
        ├── candidates/{run_name}_{timestamp}/
        │   ├── config.yaml                # Config snapshot
        │   └── startup.log                # Full console output
        ├── patches/{run_name}_{timestamp}/
        │   ├── config.yaml
        │   └── startup.log
        └── pipeline/{run_name}_{timestamp}/
            ├── config.yaml
            ├── pipeline.log               # Orchestrator log
            ├── train.log                  # Per-step logs
            ├── inference.log
            ├── visualize.log
            ├── best_model.pt              # Archived model
            ├── training_metrics.json      # Test results
            └── prediction_map.html        # Archived map
```

**Patch sharing:** All configs share `data/patches/patch_meta.csv`. If the same candidate was already extracted (same `candidate_id` + `imagery_config_hash`), it won't be re-downloaded regardless of which config triggered it.

## Config Files

Each YAML config defines the full pipeline. Key sections:

```yaml
run_name: "resnet50_lr5e5"        # Optional tag for the run directory name

data:
  countries: [united_states, thailand, ...]
  species_filter: [Chickens, Pigs, ...]
  categories_include: [Farm, "Farm (meat)", "Farm (eggs)", "Farm (dairy)"]
  train_regions:                   # Geographic splits (country or country/state)
    - united_states/NY
    - thailand                     # Entire country
  val_regions: [...]
  test_regions: [...]
  negative_sampling:
    strategy: random_rural         # or osm_buildings
    ratio: 1.0

patches:
  output_dir: data/patches         # Shared across configs
  bands: [B2, B3, B4, B8, B11, B12]
  indices: [NDVI, NDBI, NDWI]

model:
  architecture: resnet50
  hub_name: microsoft/resnet-50
  freeze_backbone_epochs: 5

training:
  epochs: 50
  learning_rate: 0.00005
  class_weight: [1.0, 2.0]        # [neg, pos] — penalize missed farms
  early_stopping_patience: 10

runpod:
  gpu_type: "NVIDIA RTX 4000 Ada Generation"
  gpu_fallbacks: [...]
  network_volume_id: r8nyom4e4e
  auto_terminate: true             # Kill pod after script completes
```

See `configs/global_all_farms.yaml` for the most complete example.

## Secrets Reference

| Variable | Where | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | `.env` only | RunPod API access |
| `GEE_SERVICE_ACCOUNT` | `.env` + RunPod Secret | Earth Engine service account email |
| `GEE_KEY_FILE` | `.env` only (local) | Path to GEE JSON key file |
| `GEE_PRIVATE_KEY_JSON` | RunPod Secret | GEE JSON key (raw or base64) |
| `GOOGLE_MAPS_API_KEY` | `.env` + RunPod Secret | Google Maps / Places API key |
