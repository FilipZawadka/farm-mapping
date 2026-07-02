# 06 — Post-training evaluators

After `_save_test_results` writes `training_metrics.json`, the train script
runs three more evaluators inside the same `with mlflow.start_run()`
block. All three follow the same pattern: reload `best_model.pt`, build a
`DataLoader`, call `_evaluate`, write a JSON, then compute the per-country
breakdown.

## Location

Source:
[training/train.py](../../training/train.py), immediately after
`_save_test_results`.

## Inspected pass

```python
if inspected_loader:
    model.load_state_dict(load_checkpoint(best_path, device)["model_state_dict"])
    _, insp_metrics = _evaluate(model, inspected_loader, criterion, device)
    mlflow.log_metrics({f"inspected_{k}": v for k, v in insp_metrics.items() if isinstance(v, (int, float))})
    (output_dir / "inspected_metrics.json").write_text(json.dumps(insp_metrics, indent=2))
```

Writes `inspected_metrics.json` for rows whose `viz_status == "inspected"`
in the training countries. This is the "hand-picked hard cases" slice —
Rachel manually reviewed these before merging into her labelling pipeline.

## Eval pass — Rachel's representative sample

```python
if eval_ds:
    model.load_state_dict(load_checkpoint(best_path, device)["model_state_dict"])
    eval_loader = DataLoader(eval_ds, batch_size=bs, shuffle=False, num_workers=0)
    _, eval_metrics = _evaluate(model, eval_loader, criterion, device)
    (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    _write_per_country_metrics(
        eval_ds, model, criterion, device, bs,
        output_dir / "eval_metrics_per_country.json", cfg,
    )
```

Writes `eval_metrics.json` + `eval_metrics_per_country.json`. This is Eval 2
in the meeting deck — the representative sample stratified by
LABEL_GROUPS (see [rachel_notebooks/label_code.py](../../rachel_notebooks/label_code.py)
`build_eval_set`). 100-150 rows per training country.

Because `eval_set` and `generalization_countries` overlap for BGD/NGA,
`build_splits` explicitly reassigns those rows from `eval` to
`generalization` (see step 04). So the eval slice is training-country
only.

## Generalization pass — OOD

```python
if gen_ds:
    model.load_state_dict(load_checkpoint(best_path, device)["model_state_dict"])
    gen_loader = DataLoader(gen_ds, batch_size=bs, shuffle=False, num_workers=0)
    _, gen_metrics = _evaluate(model, gen_loader, criterion, device)
    (output_dir / "generalization_metrics.json").write_text(json.dumps(gen_metrics, indent=2))
    _write_per_country_metrics(
        gen_ds, model, criterion, device, bs,
        output_dir / "generalization_metrics_per_country.json", cfg,
    )
```

Writes `generalization_metrics.json` + `generalization_metrics_per_country.json`.
This is Eval 3 in the meeting deck — held-out OOD countries (BGD, NGA).

## Metrics structure

Each `*_metrics.json` file:

```json
{
  "accuracy": 0.547,
  "precision": 0.578,
  "recall": 0.607,
  "f1": 0.547,
  "f1_class0": 0.624,
  "f1_class1": 0.583,
  "f1_class2": 0.435,
  "loss": 0.967
}
```

Metrics are computed by `_compute_metrics` in
[training/train.py](../../training/train.py) using
`sklearn.metrics.precision_recall_fscore_support(average="weighted")` for
overall + per-class F1. `loss` is the average CrossEntropyLoss over the
slice.

## Per-country breakdown

`_write_per_country_metrics` (line ~100):

```python
meta = eval_ds.meta                        # patch-level meta
candidates = _load_candidates_csv(...)      # candidate CSVs with country
cid_to_country = dict(zip(candidates.id, candidates.country))

for country in sorted(unique countries):
    idx = [i for i, cid in enumerate(meta.candidate_id)
           if cid_to_country[str(cid)] == country]
    sub = Subset(eval_ds, idx)
    loader = DataLoader(sub, batch_size=bs, shuffle=False)
    _, m = _evaluate(model, loader, criterion, device)
    m["n"] = len(idx)
    out[country] = m
```

Output shape:

```json
{
  "Brazil":  {"accuracy": 0.339, "precision": 0.436, ..., "n": 92},
  "Chile":   {..., "n": 135},
  "Mexico":  {..., "n": 121},
  "Thailand":{..., "n": 212},
  "United States": {..., "n": 92}
}
```

**Note**: `n` here is patches, not unique candidates. When a candidate has
multiple patches under the same imagery hash (rare, but happens), the
patch count exceeds the candidate count. To get unique-candidate counts,
join to `scored_candidates.parquet` and `.drop_duplicates("candidate_id")`.

## `scripts/post_hoc_evaluate.py`

If the training script crashed mid-run (common at v4 when the volume
quota hit `mlflow.log_artifact`), the JSONs above may be missing. Re-run
them from an existing `best_model.pt` with:

```bash
python -m scripts.post_hoc_evaluate --config configs/rachel_clusters/<name>.yaml
```

Source:
[scripts/post_hoc_evaluate.py](../../scripts/post_hoc_evaluate.py).

The script mirrors the train.py post-training block: calls
`build_splits`, loads the checkpoint at
`data/output/{stem}/best_model.pt`, and writes the three metric files +
per-country breakdowns. Idempotent — safe to re-run.

## MLflow

Each metric family gets logged with a prefix so you can compare in the
MLflow UI:

- `test_*` from `_save_test_results`
- `inspected_*` from the inspected block
- `eval_*` from the eval block
- `gen_*` from the generalization block

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `eval_metrics.json` and later files missing | Volume quota | Free space, then run `post_hoc_evaluate`. |
| `eval_metrics_per_country.json` counts don't sum to overall | Patches > candidates (multiple `.npy` per cluster) | Normalize with `drop_duplicates("candidate_id")`. |
| `generalization_metrics_per_country.json` includes non-gen countries | Old dataset.py bug (fixed pre-v5) | Update to `main`; rerun `build_splits`. |
