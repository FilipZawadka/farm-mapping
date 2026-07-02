# 01 — Data sources

## Rachel's `_for_analysis.parquet` files

The single upstream source of truth. Rachel produces one per country in her
own notebook pipeline
([rachel_notebooks/09_consolidate.ipynb](../../rachel_notebooks/09_consolidate.ipynb))
and uploads them to Google Drive under `{Country}/`.

We currently sync these countries:

| ISO | Path on Drive | Role |
|---|---|---|
| USA | `USA/USA_selected_clusters_for_analysis.parquet` | training |
| BRA | `Brazil/BRA_selected_clusters_for_analysis.parquet` | training |
| CHL | `Chile/CHL_selected_clusters_for_analysis.parquet` | training |
| MEX | `Mexico/MEX_selected_clusters_for_analysis.parquet` | training |
| THA | `Thailand/THA_selected_clusters_for_analysis.parquet` | training |
| BGD | `Bangladesh/BGD_selected_clusters_for_analysis.parquet` | generalization |
| NGA | `Nigeria/NGA_selected_clusters_for_analysis.parquet` | generalization |

Every file has these columns (per
[rachel_notebooks/config.py](../../rachel_notebooks/config.py) `REQUIRED_COLS`):

| Column | Purpose |
|---|---|
| `cluster_id` | Unique per-cluster ID (the join key) |
| `geometry` | Cluster polygon (WKB), used to compute lat/lng centroid |
| `hex3_id`, `SMOD_L2` | Spatial index + urban/rural class (not used by CNN) |
| `farm_id`, `distance_m` | Source-farm cross-reference (DMV detection uses this) |
| `original_label`, `standardized_label` | Source + normalized taxonomy label |
| `visual_label` | Human-eyeballed label from manual review |
| **`final_label`** | `visual_label` if present else `standardized_label` — **the modelling target** |
| `label_source` | Provenance string (`DMV_barns`, `Visual inspection`, `OSM`, …) |
| `notes` | Free-text annotations |
| `random_sample` | True for the ~100 clusters used to seed the eval set |
| `eval_set` | True for Rachel's representative-sample holdout |

## Syncing from Drive

Two paths, depending on whether the sync destination is the network volume
or the local machine.

### Direct rclone into the volume via the RunPod S3 API

Runs from your laptop. Requires the `runpod:` remote configured via env
vars (see `.env` — `RUNPOD_S3_*`).

```bash
rclone copy drive: "runpod:${RUNPOD_NETWORK_VOLUME_ID}/farm-mapping/data/rachel_geometry_candidates/all_countries/" \
  --include "*_selected_clusters_for_analysis.parquet" \
  --s3-acl="" --transfers=4
```

Caveats:

- RunPod's S3 rejects the `x-amz-acl` header, so `--s3-acl=""` is
  mandatory.
- Directory listings > 1000 files sometimes truncate; single-file copies
  are safe.
- The full [scripts/sync_rachel.sh](../../scripts/sync_rachel.sh) does a
  two-phase pull (Drive → S3, then unzip on the pod) — mostly used for
  the giant building-footprint zips.

### Local pull + scp to the pod

Works around every S3 quirk. Slower but bullet-proof.

```bash
# 1. Pull to /tmp
rclone copy drive: /tmp/rachel_for_analysis/ \
  --include "*_selected_clusters_for_analysis.parquet"

# 2. Push to pod
cd /tmp/rachel_for_analysis
for d in USA Brazil Chile Mexico Thailand Bangladesh Nigeria; do
  scp -P $POD_PORT -q "$d"/*.parquet \
      "root@$POD_HOST:/workspace/farm-mapping/data/rachel_geometry_candidates/all_countries/$d/"
done
```

### Seed copies committed to the repo

For the small generalization files (BGD ~110 KB, NGA ~250 KB), a copy is
under [data_seed/](../../data_seed/). Startup script auto-uses these if
the volume copy is missing, so a fresh pod can rebuild the master without
needing the Drive sync to succeed first.

## Master parquet build — `all_clusters_v4.parquet`

Consumers of the pipeline don't read `_for_analysis` files directly. They
read the master file at
`data/rachel_geometry_candidates/all_countries/all_clusters_v4.parquet`.

Builder:
[scripts/merge_clusters_v4.py](../../scripts/merge_clusters_v4.py).

Source composition:

```
Source A: all_clusters_v2.parquet, filtered to ADM0 ∉ {USA, BRA, CHL, MEX,
          THA, BGD, NGA} — 64,597 rest-of-world rows, mostly unlabelled.

Source B: USA/BRA/CHL/MEX/THA _selected_clusters_for_analysis.parquet
          — 71,322 rows total, ~14,988 labelled.

Source C: BGD/NGA _selected_clusters_for_analysis.parquet
          — 301 rows total, all labelled.

Output:  concat + dedup on cluster_id (Source B/C wins on collision)
         = 135,919 rows / 109 ADM0s.
```

Normalized schema (`UNION_COLS` in the script):

```
ADM0, cluster_id,
original_label, standardized_label, visual_label, final_label,
label_source, eval_set, random_sample,
viz_status, viz_label,
template_score_if, dmv,
geometry
```

Missing columns get filled with `None` (or `False` for the two bool
columns). `eval_set`/`random_sample` are cast to bool.

Running the builder:

```bash
python scripts/merge_clusters_v4.py
```

The startup script for a training pod auto-runs this if the master file
is missing and the config references `all_clusters_v4` (see
[training/runpod_launch.py](../../training/runpod_launch.py) `_build_startup_script`).

## `data_seed/` fallback

`merge_clusters_v4.py` checks `data_seed/{Country}/{ISO}_selected_clusters_for_analysis.parquet`
before failing — if the volume copy is missing but the seed exists, it
copies the seed to the volume and continues. Seed files are small enough
(~360 KB total for BGD + NGA) to live in git.

## What the pipeline expects

By the end of this stage, the volume should have:

```
data/rachel_geometry_candidates/all_countries/
├── all_clusters_v4.parquet             # master (109 ADM0s, 135,919 rows)
├── USA/USA_selected_clusters_for_analysis.parquet
├── Brazil/BRA_selected_clusters_for_analysis.parquet
├── Chile/CHL_selected_clusters_for_analysis.parquet
├── Mexico/MEX_selected_clusters_for_analysis.parquet
├── Thailand/THA_selected_clusters_for_analysis.parquet
├── Bangladesh/BGD_selected_clusters_for_analysis.parquet
└── Nigeria/NGA_selected_clusters_for_analysis.parquet
```

If `all_clusters_v4.parquet` is missing, the next stage
([02_candidates.md](02_candidates.md)) will refuse to run.
