# 07 — Inference

Score patches with the trained checkpoint. Produces `scored_candidates.parquet`
plus a dated sibling.

## Entry point

```bash
python -m training.inference --config configs/rachel_clusters/<name>.yaml [--labeled-only]
```

Source:
[training/inference.py](../../training/inference.py) `score_candidates`.

## What it does

```
1. Read data/patches/patch_meta.csv
2. Filter to current imagery_config_hash
3. Load candidate CSVs from cfg.data.candidates_dir
4. If cfg.inference.labeled_only (or --labeled-only CLI flag):
     drop candidates with label == -1  (skip the ~124k rest-of-world rows)
5. Filter meta to candidate_ids present in the candidate CSVs
6. Build PatchDataset with the SAME channel_subset + crop_size the model
   was trained on.
7. Load best_model.pt from cfg.inference.checkpoint.
8. Score every patch with torch.no_grad + softmax.
   - Binary (num_classes=2): score = P(class=1); pred = score >= threshold.
   - Multi-class: score = top-1 prob; pred = argmax; also record every
     per-class probability as prob_class{k}.
9. Attach:
   - true_label (from candidate.label)
   - source, country (from candidate)
   - split (from data/patches/splits/{config_stem}.csv)
   - Diagnostic cols: original_label, standardized_label, visual_label,
     label_source, notes, eval_set, random_sample, viz_status
10. Wrap in GeoDataFrame with Point geometry.
11. Write scored_candidates.parquet + scored_candidates_{YYYYMMDD_HHMM}.parquet
    (dated sibling for audit).
```

## `--labeled-only` mode

Default is to score every candidate in the CSVs — including ~124k rest-of-
world rows that only contribute UP/UN points to the world map. On slow
MooseFS reads (~100 sleeping threads on the L4 SECURE pod), this can take
30-60 min.

`--labeled-only` or `cfg.inference.labeled_only: true` drops the unlabelled
candidates before scoring. Only the ~17k labelled slices (train + val +
test + inspected + eval + generalization) are scored. ~8× faster; the
metric slices are untouched but the prediction map lacks the rest-of-world
UP/UN layers.

Rule of thumb:
- When you only want the metrics or a labelled-countries map, use
  `--labeled-only`.
- When you want the full global map for demo/deployment, leave the flag off.

All `world_v7+` configs default to `labeled_only: true`.

## Confidence tiers

`cfg.inference.confidence_tiers = {high: 0.9, medium: 0.7, low: 0.4}`.
Adds a `confidence_tier` column to the scored parquet: `high`, `medium`,
`low`, `very_low`. Not used by the map coloring but useful for post-hoc
filtering ("keep only high-confidence predictions").

## Split column

The step reads `data/patches/splits/{config_stem}.csv` (produced by
`build_splits`) and attaches the `split` value to each row. Values:
`train`, `val`, `test`, `inspected`, `eval`, `generalization`,
`unlabeled`, `unknown`.

## Diagnostic column propagation

`_attach_labels` (v4+) forwards these columns from the candidate CSVs to
the scored parquet:

- `original_label`
- `standardized_label`
- `visual_label`
- `label_source`
- `notes`
- `eval_set` (0/1)
- `random_sample` (0/1)
- `viz_status`

Motivation: Rachel's meeting-deck slide 10 asked us to "propagate original
cols" so she could audit "bad labels" without a manual join back to her
`_for_analysis.parquet`.

## Output shape

Columns in `scored_candidates.parquet`:

```
candidate_id, lat, lng,
predicted_score, predicted_label, confidence_tier,
prob_class0, prob_class1, prob_class2,     # only when num_classes >= 3
true_label, source, country,
split,
original_label, standardized_label, visual_label,
label_source, notes, eval_set, random_sample, viz_status,
geometry
```

Row count = number of patches (after imagery-hash filter + candidate
filter). Since some clusters have multiple patches, `len(df) >=
df.candidate_id.nunique()`.

## Common queries against the scored parquet

Unique-candidate counts per split:

```python
df = gpd.read_parquet("data/output/world_v7_three_class/scored_candidates.parquet")
df.drop_duplicates("candidate_id").split.value_counts()
```

Per-country eval:

```python
e = df[df.split == "eval"].copy()
e["tb"] = e.true_label.isin([1, 2]).astype(int)   # binary rollup
e["pb"] = e.predicted_label.isin([1, 2]).astype(int)
for c, sub in e.groupby("country"):
    tp = ((sub.tb == 1) & (sub.pb == 1)).sum()
    ...
```

3-class confusion matrix:

```python
NAMES = {0: "NotFarm", 1: "Poultry", 2: "OtherFarm"}
s = df[df.split == "eval"]
pd.crosstab(s.true_label.map(NAMES), s.predicted_label.map(NAMES), margins=True)
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `No patches with imagery_config_hash=...` | Config changed after patch extraction | Re-run patch_extraction or revert the changed field. |
| Inference takes 30-60 min on L4 | Network-mount patch reads bottleneck GPU | Use `--labeled-only`; or run inference on a CPU pod (labeled-only takes ~2 min). |
| `scored_candidates.parquet` missing after inference | Pod terminated between train and inference | `python -m training.run_pipeline --steps inference visualize --config <cfg>`. |
| `split` column all `"unknown"` | Splits CSV missing | Ran inference before training on this config — re-run training or produce a splits CSV manually. |

## What the next stage needs

- `scored_candidates.parquet` in `data/output/{cfg._config_stem}/`.
- (Optional) same file with dated sibling for audit trail.
