# 09 — Configuration reference

Every run is fully described by a single YAML file under
`configs/rachel_clusters/`. This doc lists every field, its default, and
what it does.

## Root shape

```yaml
run_name: str                # optional label for logs / run dirs

data: DataConfig
patches: PatchConfig
model: ModelConfig
training: TrainingConfig
mlflow: MLflowConfig
runpod: RunPodConfig
cache: CacheConfig
inference: InferenceConfig
visualization: VizConfig
```

Every subsection is a Pydantic model in
[training/config.py](../../training/config.py). Missing fields fall back
to defaults.

## `data`

Governs the master parquet, candidate CSVs, and split routing.

| Field | Default | Purpose |
|---|---|---|
| `countries` | `[]` | Legacy: filter candidates to these country_keys. Empty = all. |
| `species_filter` | `[]` | Legacy: filter to species. Empty = all. |
| `categories_include` | `[]` | Legacy: FT category whitelist. |
| `train_regions`, `val_regions`, `test_regions` | `null` | Legacy region-based split. Not used by world_v* runs. |
| **`training_countries`** | `[]` | ADM0 whitelist. Non-listed labelled rows are demoted to `unlabeled`. Empty = no restriction. |
| **`generalization_countries`** | `[]` | ADM0s held out as OOD (`generalization` split). |
| **`dmv_force_to_train_only`** | `false` | Pin DMV rows to train; strip from val/test/inspected. |
| `candidates_dir` | `data/candidates` | Where the candidates step writes CSVs. |
| **`parquet_source`** | `null` | Master parquet (`all_clusters_v4.parquet` for world_v*). |
| `extra_parquet_sources` | `[]` | Additional parquets to merge in. |
| **`include_unlabeled`** | `false` | Keep rows with `final_label` NULL so inference can score them. |
| `inspected_as_test` | `false` | Hold out `viz_status == inspected` rows. |
| `inspected_only` | `false` | Keep only inspected rows (for one-off audits). |
| **`label_mode`** | `binary` | `binary` \| `poultry` \| `three_class` \| `multiclass`. |
| `exclude_labels` | `[]` | Drop rows where `label_col` value is in this list. |
| `exclude_osm_farms` | `false` | Drop OSM-tagged farm rows (noisy). |
| `negative_sampling` | see below | Legacy — random rural / OSM buildings / hard negative sampling. |
| `building_footprints` | see below | Legacy — Google Open Buildings / MS ML buildings source. |

## `patches`

Imagery + patch caching.

| Field | Default | Purpose |
|---|---|---|
| `patch_size_px` | 128 | Window in pixels. 128×10 m = 1.28 km context. |
| `resolution_m` | 10 | Sentinel-2 native. |
| `bands` | `[B2, B3, B4, B8, B11, B12]` | S2 bands stacked into the patch. |
| `indices` | `[NDVI, NDBI, NDWI]` | Spectral indices, appended after the bands. |
| `composite` | `median` | `median` or `least_cloudy`. |
| `date_range` | `[2023-01-01, 2023-12-31]` | S2 collection window. |
| `max_cloud_cover` | 15 | Per-image cloud-cover cap. |
| `output_dir` | `data/patches` | Shared across configs. |
| `num_workers` | 4 | Parallel EE requests. |
| `retry_failed` | `false` | Retry `failed_patches.csv` IDs. |
| `imagery_sources` | `null` | Optional multi-provider list (e.g. S1 + S2). |

The property `n_channels` derives from `bands + indices` (or the
`imagery_sources` list).

## `model`

Architecture.

| Field | Default | Purpose |
|---|---|---|
| `architecture` | `resnet50` | `resnet50` \| `convnext_tiny` \| `efficientnet_b0` \| `swin_tiny` \| `resnet50_satlas`. |
| `hub_name` | `microsoft/resnet-50` | HuggingFace ID (or `SENTINEL2_SI_MS_SATLAS`). |
| `pretrained` | `true` | Load pretrained weights. |
| `num_classes` | 2 | 2 (binary), 3 (three_class), 7 (multiclass). |
| `input_channels` | 9 | Must match patch channels after `channel_subset` if set. |
| `freeze_backbone_epochs` | 3 | Epochs to keep the backbone frozen. |
| `class_names` | `null` | List of names for multi-class visualize palette. |
| `class_colors` | `null` | Matching hex colors. |

## `training`

Loop knobs.

| Field | Default | Purpose |
|---|---|---|
| `epochs` | 30 | Cap; early stop can beat it. |
| `batch_size` | 32 | GPU-side batch. |
| `learning_rate` | 1e-4 | AdamW LR. |
| `weight_decay` | 0.01 | AdamW WD. |
| `scheduler` | `cosine` | `cosine` \| `step` \| `plateau`. |
| `early_stopping_patience` | 5 | Epochs without val_loss improvement before stop. |
| `val_split` | 0.15 | Fraction of the labelled pool → val (random split path). |
| `test_split` | 0.15 | Same → test. |
| `seed` | 42 | RNG seed for splits + sampling. |
| `mixed_precision` | `true` | AMP on CUDA. |
| **`class_weight`** | `null` | `[w_0, w_1, w_2]` per-class CrossEntropyLoss weights. |
| **`upsample_minority_regions`** | `false` | Per-country sampler weights. |
| **`balanced_country_splits`** | `false` | Country-balanced val/test. |
| **`balanced_class_sampling`** | `false` | Per-class sampler weights (multiplies with region). |
| `channel_subset` | `null` | e.g. `[B2, B3, B4, NDWI]` — subset by name at train time. |
| `crop_center_px` | `null` | Center-crop patches at train time (64 = 640 m context). |
| `resume_from` | `null` | Path to a checkpoint to continue training from. |
| `resume_optimizer_state` | `true` | Load optimizer state as well as weights. |
| `augmentation` | see model | Per-aug config nested under `augmentation:`. |

## `mlflow`

| Field | Default | Purpose |
|---|---|---|
| `tracking_uri` | `./mlruns` | Filesystem backend. |
| `experiment_name` | `farm_detection_v1` | Groups runs. |
| `log_model` | `true` | Also log best_model.pt as an artifact (~270 MB per run). |

Set `log_model: false` when volume quota is tight.

## `runpod`

Governs the `runpod_launch.py` behaviour (not the training itself).

| Field | Default | Purpose |
|---|---|---|
| `gpu_type` | `NVIDIA A40` | Primary GPU to request. |
| `gpu_fallbacks` | `[]` | Ordered fallbacks. |
| `cloud_type` | `ALL` | `ALL` \| `COMMUNITY` \| `SECURE`. Set to `SECURE` for stability. |
| `docker_image` | RunPod PyTorch image | Full base image. |
| `volume_mount` | `/workspace` | Where the network volume lands. |
| `code_dir` | `/workspace/farm-mapping` | Where the repo lives on the pod. |
| `github_repo` | `""` | URL used for fresh clones and self-heal. |
| `github_branch` | `main` | Branch. |
| `api_key_env` | `RUNPOD_API_KEY` | Env var carrying the RunPod API key. |
| `network_volume_id` | `null` | Volume to attach. |
| `cpu_instance_id` | `cpu3g-4-16` | Default CPU pod flavor. |
| `cpu_fallbacks` | `[]` | Ordered CPU fallbacks. |
| `auto_terminate` | `true` | Self-terminate after the pipeline. |

## `cache`

Legacy: optional S3 / GCS / RunPod cache for shared patch stores. Not
used by world_v* runs.

## `inference`

| Field | Default | Purpose |
|---|---|---|
| `checkpoint` | `output/best_model.pt` | Path to load. |
| `threshold` | 0.5 | Binary decision threshold. |
| `confidence_tiers` | high/med/low | Score buckets attached to output. |
| **`labeled_only`** | `false` | Skip unlabelled rows (~8× faster; no world map). Toggle via `--labeled-only` CLI. |

## `visualization`

| Field | Default | Purpose |
|---|---|---|
| `output_dir` | `output/maps` | Where the HTML lands. |
| `show_true_positives` | `true` | TP layer visible on load. |
| `show_false_positives` | `true` | FP visible on load. |
| `show_false_negatives` | `true` | FN visible on load. |
| `show_true_negatives` | `false` | Hidden by default. |

## Full example (world_v5)

See
[configs/rachel_clusters/world_v5_three_class.yaml](../../configs/rachel_clusters/world_v5_three_class.yaml)
for the canonical strict-framework three-class recipe.

Excerpts:

```yaml
data:
  parquet_source: data/rachel_geometry_candidates/all_countries/all_clusters_v4.parquet
  inspected_as_test: true
  include_unlabeled: true
  label_mode: three_class
  training_countries: [USA, BRA, CHL, MEX, THA]
  generalization_countries: [BGD, NGA]
  dmv_force_to_train_only: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  balanced_country_splits: true          # v5: on
  balanced_class_sampling: true          # v5: on
  channel_subset: [B2, B3, B4, NDWI]
  crop_center_px: 64

inference:
  labeled_only: false                    # world map
```

The v6 config flips both balancing knobs to `false`; the v7 config adds
`class_weight: [1.0, 0.7, 2.0]` and `inference.labeled_only: true`.

## Loading a config from Python

```python
from training.config import load_config, resolve_paths
cfg = resolve_paths(load_config("configs/rachel_clusters/world_v5_three_class.yaml"))
```

`resolve_paths` promotes every relative path to absolute, and sets
`cfg._config_stem` (the file basename without `.yaml`) which the pipeline
uses to name output directories.

## Config precedence + hashing

- `imagery_config_hash` = hash of `patches.{bands, indices, composite,
  max_cloud_cover, date_range}` + `imagery_sources` if set. A change here
  invalidates the patch cache silently.
- `cache_key` = hash of `(data, patches)` for optional shared caches
  (unused in world_v*).
