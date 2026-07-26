# Pipeline — step-by-step reference

End-to-end walkthrough of the farm-mapping pipeline. Read top-to-bottom to
understand what happens between "Rachel drops a new `_for_analysis.parquet`
on Drive" and "we have a global prediction map."

Each numbered file below covers one stage; the code references are file:line
so you can jump straight to the implementation.

## Stage order

| # | Doc | What happens | Main code |
|---|---|---|---|
| 0 | [00_overview.md](00_overview.md) | System map, glossary, run flavors | — |
| 1 | [01_data_sources.md](01_data_sources.md) | Rachel's `_for_analysis` parquets, master build | [scripts/merge_clusters_v4.py](../../scripts/merge_clusters_v4.py) |
| 2 | [02_candidates.md](02_candidates.md) | Parquet → per-country candidate CSVs | [training/rachel_to_candidates.py](../../training/rachel_to_candidates.py) |
| 3 | [03_patch_extraction.md](03_patch_extraction.md) | Sentinel-2 patch download from Earth Engine | [training/patch_extraction.py](../../training/patch_extraction.py) |
| 4 | [04_splits.md](04_splits.md) | Train / val / test / eval / generalization / unlabeled routing | [training/dataset.py](../../training/dataset.py) `build_splits` |
| 5 | [05_training.md](05_training.md) | ResNet-50 fine-tune, MLflow logging, best checkpoint | [training/train.py](../../training/train.py) |
| 6 | [06_post_train_eval.md](06_post_train_eval.md) | Inspected / eval / generalization evaluators | [training/train.py](../../training/train.py) tail + [scripts/post_hoc_evaluate.py](../../scripts/post_hoc_evaluate.py) |
| 7 | [07_inference.md](07_inference.md) | Score every candidate → `scored_candidates.parquet` | [training/inference.py](../../training/inference.py) |
| 8 | [08_visualization.md](08_visualization.md) | Leaflet HTML map + per-split panel | [training/visualize.py](../../training/visualize.py) |
| 9 | [09_configuration.md](09_configuration.md) | YAML config schema + every knob | [training/config.py](../../training/config.py) |
| 10 | [10_deployment.md](10_deployment.md) | RunPod launch + startup script + self-heal git | [training/runpod_launch.py](../../training/runpod_launch.py) |
| 11 | [11_audit_and_debug.md](11_audit_and_debug.md) | Standard sanity checks + how to relaunch a broken run | various |

## Companion docs

- [EVAL_FRAMEWORK.md](../EVAL_FRAMEWORK.md) — Rachel's three-role framework
  (training / generalization / labeled countries) and what each metrics
  file actually measures.
- [CODEBASE.md](../CODEBASE.md) — repo-level overview: directories,
  experiment catalog, config cheat-sheet.
- [runpod-storage.md](../runpod-storage.md) — network-volume layout on
  RunPod, common quota / cleanup operations.

## Reading order for common tasks

- **New collaborator**: 00 → 01 → 02 → 04 → 05 → 07 → 09.
- **Debugging a bad metric**: 04 → 06 → 11.
- **Spinning up a new experiment**: 09 → 10.
- **Producing a world prediction map**: 07 → 08 (with `inference.labeled_only: false`).
- **Rebuilding master parquet**: 01.
