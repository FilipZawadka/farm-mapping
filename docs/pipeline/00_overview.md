# 00 — System overview

## What we build

A ResNet-50-based classifier that scores building clusters from
Sentinel-2 satellite imagery as:

- Binary: `Farm` vs `NotFarm`
- Three-class: `NotFarm` / `Poultry` / `OtherFarm` (pigs+cattle)
- Seven-class: full species taxonomy (`Poultry: Meat`, `Poultry: Eggs`,
  `Pigs`, `Cattle`, …)

The output is a per-cluster score plus an interactive world map showing
predictions colored by prediction class (TP/FP/FN/TN for labeled slices;
UP/UN for unlabeled countries).

## System map

```
Rachel's Drive          Network volume (RunPod)              GPU/CPU pod                 Local
──────────────         ──────────────────────────           ─────────────────           ─────────
{ISO}_selected  ─rclone→  data/rachel_geometry_cand    ──→  training.candidates   ──→  candidates_dir/*.csv
   _clusters                /all_countries/                         │
   _for_analysis           /{Country}/                              ▼
                                                       ┌── training.patch_extraction ──┐
                                                       │      (Earth Engine call)      │
                                                       │                               │
                                                       │        Sentinel-2 ─────┐      │
                                                       │                       ▼      │
                                                       └── data/patches/{ADM0}/*.npy──┘
                                                                          │
                                                                          ▼
                                                          training.dataset.build_splits
                                                          (train / val / test /
                                                           inspected / eval /
                                                           generalization / unlabeled)
                                                                          │
                                                                          ▼
                                                             training.train
                                                             (ResNet-50 + MLflow)
                                                                          │
                                                                          ▼
                                                             training.inference
                                                             (scored_candidates.parquet)
                                                                          │
                                                                          ▼
                                                             training.visualize
                                                             (prediction_map.html) ────→  local `output/maps_*/`
```

## Pipeline stages

The whole run is orchestrated by [training/run_pipeline.py](../../training/run_pipeline.py)
in this order:

1. **candidates** — Convert Rachel's parquet(s) to per-country CSVs.
2. **patch_extraction** — Fetch a Sentinel-2 median composite from Earth
   Engine for every candidate. Skips already-cached patches.
3. **train** — Fit the CNN. Writes `best_model.pt`, `training_metrics.json`,
   `inspected_metrics.json`, `eval_metrics.json` + `_per_country.json`,
   `generalization_metrics.json` + `_per_country.json`.
4. **inference** — Score every candidate (or labelled-only, see step 07)
   with the best checkpoint → `scored_candidates.parquet`.
5. **visualize** — Turn the scored parquet into an interactive HTML map
   with per-split + per-country stats panels.

Each stage can also be invoked directly (e.g., `python -m training.inference
--config <cfg>`), and `run_pipeline` accepts `--steps ...` to skip finished
stages when resuming.

## Country roles

Per [docs/EVAL_FRAMEWORK.md](../EVAL_FRAMEWORK.md), each labelled ADM0 has
exactly one role:

| Role | ADM0s | What they do |
|---|---|---|
| Training | USA, BRA, CHL, MEX, THA | CNN train / val / test / inspected / eval_set holdout |
| Generalization | BGD, NGA (extensible) | Held-out OOD slice — never enters train |
| Inference-only | ~104 rest-of-world | Not used for training or metrics; scored for the map |

The strict whitelist is enforced by `data.training_countries` +
`data.generalization_countries` in the config; any labelled row in a
non-listed country gets demoted to the `unlabeled` split (see step 04).

## Splits summary (from `data/patches/splits/<config>.csv`)

| Split | Origin | Purpose |
|---|---|---|
| `train` | training countries, labelled, not eval_set / gen / DMV-in-val-test | gradient updates |
| `val` | training countries, labelled, not eval_set / gen | early-stopping + LR-schedule signal |
| `test` | training countries, labelled, not eval_set / gen | headline holdout (Eval 1 in the meeting deck) |
| `inspected` | `viz_status == inspected`, training countries only | secondary holdout — hand-picked hard cases |
| `eval` | `eval_set == True`, training countries only | Rachel's representative sample (Eval 2) |
| `generalization` | ADM0 ∈ `generalization_countries` | OOD held-out (Eval 3) |
| `unlabeled` | `final_label` NULL | never trained on; scored at inference for the map |
| `unknown` / `unassigned` | edge cases | usually zero rows |

## Glossary

- **Cluster** — a set of adjacent buildings detected by Rachel's geometric
  filter. The unit of prediction.
- **`_for_analysis.parquet`** — Rachel's per-country labelled cluster
  export; the source of truth for labels + `eval_set` membership.
- **`all_clusters_v{N}.parquet`** — the "master" file our pipeline reads.
  Built by [scripts/merge_clusters_v4.py](../../scripts/merge_clusters_v4.py)
  from the per-country `_for_analysis` files plus rest-of-world
  unlabelled rows carried from v2.
- **Candidate** — one row of the candidate CSV; carries `id, lat, lng,
  label, region, eval_set, ...`. One candidate per cluster.
- **Patch** — one `.npy` file `(C, H, W)` at `data/patches/{ADM0}/{state}/
  {imagery_hash}/{candidate_id}.npy`. Usually 9 channels (`B2, B3, B4, B8,
  B11, B12, NDVI, NDBI, NDWI`) at 128×128 px / 10 m = 1.28 km context.
- **DMV** — Rachel's Delmarva Peninsula poultry-barn dataset. Very clean
  labels used to fit her Isolation Forest, so our CNN must not evaluate
  itself on them (config: `data.dmv_force_to_train_only: true`).
- **Imagery config hash** — a stable 12-char hash of the bands / indices /
  date range / cloud cover / composite type. Written into `patch_meta.csv`
  and used by inference to only score patches whose imagery matches the
  current config.

## Run flavors

Runs are named after the training data + task, e.g. `world_v5_three_class`,
`world_v6_three_class`. Everything under `configs/rachel_clusters/` is one
of these. The `world_v{N}` prefix bumps whenever the *composition of the
labelled data or splits* changes. Model architecture changes usually stay
inside a version and get suffixed differently.

Recent Δ history:

| Config | vs previous | Purpose |
|---|---|---|
| v2 (all_clusters) | first world master | Legacy binary |
| v3 | inspected-as-test | Add held-out inspected slice |
| v4 | adds BGD/NGA in master, `generalization_countries` in code, label-col propagation | Rachel's OOD framework |
| v5 | strict `training_countries` whitelist + `dmv_force_to_train_only` | Adherence to Rachel's framework |
| v6 | all sampler balancing OFF | Fix v5's Poultry ↔ OtherFarm confusion |
| v7 | v6 + light class weight `[1, 0.7, 2.0]` | Try to rescue OtherFarm (failed — weight too light) |
| v8 | v6 recipe + split bugfix + val-F1 selection + per-channel norm | See below — v6/v7 results are INVALID |

> **⚠ v6/v7 correction (2026-07-02).** The old split code stratified by
> `label==1`/`label==0` only, so with `balanced_country_splits: false`
> (v6/v7) every OtherFarm (`label==2`) row was silently dropped from
> train/val/test. The model never saw one OtherFarm example — F1_class2=0
> was inevitable, and v7's class weight had nothing to act on. Both runs'
> conclusions about "natural distribution" are void. Fixed in
> `training/dataset.py` (`_stratified_class_split`); `build_splits` now
> hard-fails when any class has zero train rows. See
> [docs/EXPERIMENTS_v8.md](../EXPERIMENTS_v8.md).
