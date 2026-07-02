# 03 — Patch extraction

Fetch a Sentinel-2 median composite from Earth Engine for every candidate,
save as `(C, H, W) float32 .npy` on the network volume.

## Entry point

```bash
python -m training.patch_extraction --config configs/rachel_clusters/<name>.yaml
```

Source:
[training/patch_extraction.py](../../training/patch_extraction.py).

## Prerequisites

- Every candidate CSV from step 02 present under `cfg.data.candidates_dir`.
- Earth Engine credentials. Two accepted forms
  ([training/env_loader.py](../../training/env_loader.py) `get_gee_credentials`):
  - **File**: `GEE_SERVICE_ACCOUNT` + `GEE_KEY_FILE=/root/gee-key.json`
    (default local & pod setup).
  - **Inline**: `GEE_SERVICE_ACCOUNT` + `GEE_PRIVATE_KEY_JSON=<raw or base64>`
    (RunPod secret injection — the launcher passes this into every pod).

## Flow

```
Read all CSVs from candidates_dir
  │
  ▼
Load existing patch_meta.csv (resume support)
  │  skip candidate_ids already present
  ▼
Load failed_patches.csv (unless retry_failed=true)
  │  skip previously failed candidate_ids
  ▼
For each candidate:
  ├── Resolve imagery provider(s) — S2 by default, optional S1
  ├── Call ee.computePixels() for a 128×128 window centered on lat/lng
  ├── Stack bands (B2, B3, B4, B8, B11, B12) + indices (NDVI, NDBI, NDWI)
  │     → 9-channel float32 (C, H, W)
  ├── Write .npy to data/patches/{ADM0}/{state}/{imagery_hash}/{id}.npy
  └── Append row to patch_meta.csv (candidate_id, lat, lng, state,
                                     n_channels, height, width,
                                     clear_pixel_fraction, patch_path,
                                     imagery_config_hash, bands,
                                     date_range, composite, provider)
```

Failures append to `failed_patches.csv` with the EE error message ("no
patch in previous runs", "extraction failed", etc.).

## Imagery config hash

Twelve-char hex hash of the imagery-affecting fields
([training/config.py](../../training/config.py) `imagery_config_hash`):

- `bands`, `indices` (or `imagery_sources` if present)
- `composite` (`median` or `least_cloudy`)
- `max_cloud_cover`
- `date_range`

Both the `.npy` path and the `patch_meta.csv` row carry this hash, so:

- Multiple imagery configs can coexist in `data/patches/` without
  clobbering each other.
- Inference reads `patch_meta.csv` and filters to
  `imagery_config_hash == current_config_hash` before scoring — a config
  change silently invalidates its cache and forces re-extraction.

## Resume behavior

- `patch_meta.csv` is checkpointed every ~200 successful extractions.
- On restart, existing `candidate_id` rows are skipped.
- Failed IDs skip too (see `_load_failed_ids`) unless
  `patches.retry_failed: true`, which clears `failed_patches.csv` and
  retries them.

## Config knobs

From `PatchConfig` in
[training/config.py](../../training/config.py):

| Field | Default | Purpose |
|---|---|---|
| `patch_size_px` | 128 | Window size in pixels (1.28 km at 10 m/px). |
| `resolution_m` | 10 | Sentinel-2 native resolution. |
| `bands` | `[B2, B3, B4, B8, B11, B12]` | S2 SR bands to include. |
| `indices` | `[NDVI, NDBI, NDWI]` | Spectral indices computed from `bands`. |
| `composite` | `median` | `median` or `least_cloudy`. |
| `date_range` | `[2023-01-01, 2023-12-31]` | S2 image collection window. |
| `max_cloud_cover` | 15 | `%` cap per S2 image included in the composite. |
| `output_dir` | `data/patches` | Shared across configs — same cache. |
| `num_workers` | 4 | Parallel EE requests. |
| `retry_failed` | `false` | Retry candidates in `failed_patches.csv`. |
| `imagery_sources` | `null` | Optional multi-provider list (S1 + S2, etc.). |

## Directory layout

```
data/patches/
├── patch_meta.csv          # global registry, one row per patch
├── failed_patches.csv      # skipped IDs + reason
├── AFG/nan/cc5a6ebb502a/
│   ├── AFG_cluster_1.npy
│   ├── AFG_cluster_168.npy
│   └── ...
├── Brazil/nan/cc5a6ebb502a/
├── United States/AL/cc5a6ebb502a/
│   └── ...
└── splits/                # one CSV per config, written by build_splits
    ├── world_v4_three_class.csv
    ├── world_v5_three_class.csv
    └── ...
```

The `state` layer is only non-empty for USA (state code inferred from lat/lng).

## Typical timings

- Full v4 world (~135k patches, first run): 1-2 h on a 16-worker CPU pod.
- v5 → v6 re-run (mostly cached): a few minutes for the diff (~78 new IDs).
- Retry of 3,376 failed patches: ~5 min; recovered ~1,300 transient
  failures, remaining ~2,000 are persistent "no-clear-pixel" failures.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `Image.select: Band pattern 'B2' was applied to an Image with no bands` | No S2 image satisfies `max_cloud_cover` in the date range | Relax cloud cover to 30-50 or widen the date range. Note: changes `imagery_config_hash`, so patches land in a new bucket that current inference won't see. |
| Errors then process dies | EE quota / transient network | Retry with `patches.retry_failed: true`. |
| Log ends abruptly at "Loaded checkpoint" | Volume disk quota | Free space — see [runpod-storage.md](../runpod-storage.md). |

## What the next stage needs

- `data/patches/patch_meta.csv` — every candidate that will feed the
  splitter.
- `.npy` files at the paths listed in `patch_meta.csv`.

If a candidate is in the CSV but its patch is missing, the splitter drops
it — see the "Filtered to N patches" log line in
[04_splits.md](04_splits.md).
