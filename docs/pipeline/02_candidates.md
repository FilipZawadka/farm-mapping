# 02 — Candidates step

Convert the master parquet into per-country candidate CSVs the rest of the
pipeline consumes.

## Entry point

```bash
python -m training.rachel_to_candidates --config configs/rachel_clusters/<name>.yaml
```

Or inside `run_pipeline`:

```
=== Step: candidates ===
```

Source:
[training/rachel_to_candidates.py](../../training/rachel_to_candidates.py)
`convert()`.

## What it does — flow

```
all_clusters_v4.parquet
        │
        │  pd.read_parquet + label-column selection
        ▼
label_col = "final_label" if present else "modified_label"
        │
        │  optional filters (inspected_only, exclude_labels, exclude_osm_farms,
        │  include_unlabeled)
        ▼
label mapping (binary / poultry / three_class / multiclass)
        │
        ▼
compute centroid (lat, lng) from WKB geometry
map ADM0 → country_key (USA→united_states, BRA→brazil, …)
propagate diagnostic cols (visual_label, label_source, eval_set,
                            random_sample, viz_status, notes)
infer US state from lat/lng (uses osm_negatives.US_STATE_BOUNDS)
build region string ("united_states/CA" or just "brazil")
        │
        ▼
write per-country CSVs to
data/rachel_geometry_candidates/{cfg.data.candidates_dir}/{country_key}.csv
```

## Label modes

Controlled by `cfg.data.label_mode`.

| Mode | Mapping | Notes |
|---|---|---|
| `binary` (default) | `NotFarm → 0`, anything else → `1`, NULL → `-1` | Every farm row becomes positive. |
| `poultry` | `contains "Poultry" → 1`, else → `0`, NULL → `-1` | Poultry-vs-rest (used by legacy runs). |
| `three_class` | `NotFarm=0, Poultry=1, OtherFarm=2`; drops `Farm: Mixed / Other / Unknown / PigsOrPoultry` | Bumps ambiguous rows. |
| `multiclass` | Full 7-class taxonomy (see `_MULTICLASS_MAP` at line ~183) | For fine-grained species work. |

## Columns emitted into the CSV

From the `keep` list at line 244:

```
id, name, lat, lng, species, category, source, country, state,
label, region, viz_status,
visual_label, label_source, eval_set, random_sample,
num_bldgs, total_area_m2, median_area, template_score_if
```

`id` == `cluster_id` from the parquet — this is the join key for all
downstream stages (patch names, splits CSV, scored parquet).

## Diagnostic columns (v4+)

Added in the "label propagation" work (slide 10 of the meeting deck).
`_attach_labels` in inference forwards these into `scored_candidates.parquet`
so downstream reviewers can audit "bad labels" without re-joining Rachel's
files.

Handling:

- `viz_status` — filled with `""` where NULL.
- `visual_label` — filled with `""`.
- `label_source` — filled with `""`.
- `eval_set` — filled with `False` → cast to `0/1 int`.
- `random_sample` — same as `eval_set`.

## `include_unlabeled` semantics

- `false` (default) — drops rows where `label_col` is NULL. Output CSVs
  contain only labelled candidates.
- `true` — keeps NULL-labelled rows. Their `label` becomes `-1` and they
  flow through to inference as unlabelled predictions (UP/UN on the map).

Every `world_v*` config sets `include_unlabeled: true` so the map covers
the whole world.

## OSM farm exclusion

`cfg.data.exclude_osm_farms: true` drops rows where
`original_label` contains "OSM" *and* the row looks like a farm (via
`standardized_label` or OSM keyword match). Used to strip noisy OSM-only
"farm" tags that are actually houses / warehouses.

## Output

Per-country CSVs land under `data/rachel_geometry_candidates/{cfg.data.candidates_dir}/`.
Example (three-class v4):

```
data/rachel_geometry_candidates/candidates_world_v4_three_class/
├── afg.csv
├── ago.csv
├── albania.csv
├── ...
├── united_states.csv       # 35,867 rows
├── ven.csv
└── zwe.csv
```

Log line at end:

```
INFO Saved 35867 candidates to .../united_states.csv (pos=8073, neg=2051, unlabeled=25743, eval_set=102)
```

## What the next stage needs

The patch_extraction step reads every CSV in the directory, so anything
you drop there gets scored. A common workflow bug: leaving stale CSVs
from a previous config (e.g. copying files by hand) — the pipeline will
happily patch-extract candidates that shouldn't be in the run.

To avoid: use `candidates_dir` in the config to point at a fresh directory
per run (all `world_v*` configs do this).
