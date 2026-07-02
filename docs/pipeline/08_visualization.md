# 08 — Visualization

Turn `scored_candidates.parquet` into an interactive Leaflet HTML map with
per-split + per-country metric panels.

## Entry point

```bash
python -m training.visualize --config configs/rachel_clusters/<name>.yaml
```

Source:
[training/visualize.py](../../training/visualize.py) `visualize`.

## What it produces

Under `cfg.visualization.output_dir` (defaults to `output/maps_{config_stem}`):

- `prediction_map.html` — canonical latest.
- `prediction_map_{YYYYMMDD_HHMM}.html` — dated sibling.

Same dated-archive convention as the scored parquet.

## What the map contains

### Layers (toggleable)

Binary and multiclass runs use different layer palettes.

**Binary / three-class / multiclass with labels**:

| Layer | Meaning | Color |
|---|---|---|
| TP | True positive (labelled + predicted farm) | green `#2ecc71` |
| FP | False positive | red `#e74c3c` |
| FN | False negative | orange `#f39c12` |
| TN | True negative | gray `#95a5a6` |
| UP | Predicted farm, no label (rest-of-world) | blue `#3498db` |
| UN | Predicted not-farm, no label | light gray `#d0d3d4` |

**Multi-class extras** (when `num_classes >= 3`): each true-class ×
predicted-class combination gets its own toggleable layer with a
distinct color from the class palette.

### Panels (top-right corner)

- **Overall metrics** table: accuracy / precision / recall / F1 / TP / FP
  / FN / TN.
- **Per-Split** table: rows for `train`, `val`, `test`, `inspected`,
  `eval`, `generalization`.
- **Per-Country** table: precision / recall / TP / FP / FN / UP / UN per
  country (sorted alphabetically).

The per-split rows are only populated when the scored parquet carries
those splits — a `--labeled-only` run doesn't have `unlabeled` rows, so
that split doesn't appear.

## Split-membership routing

Points are colored by their prediction class (TP/FP/FN/TN etc.) and
filtered by split checkbox. `_classify_predictions` picks one of the
tags per row:

- `true_label ∈ {0, 1}` (or `0..k-1` for multiclass) and matches
  `predicted_label` → TP or TN.
- Mismatch → FP or FN.
- `true_label = -1` (unlabelled) → UP or UN.

`_build_split_layers` groups rows by `split` first, then colors by
`pred_class`. That way you can hide the training set while still seeing
generalization + eval as separate layers.

## Unlabelled world (map-only) mode

When `inference.labeled_only: false`, ~124k UP/UN points cover the whole
world. The HTML file balloons from ~1 MB to ~80 MB.

When `labeled_only: true`, only labelled slices show — the map is small
(~1 MB) and focused on the metric slices.

## Confidence tiers on the map

`predicted_score` (or `confidence_tier`) is included in each Feature's
`properties`, so mousing over a point shows the score. Not currently used
for stratified filtering, but easy to add via a slider layer.

## Config knobs

Under `cfg.visualization`:

| Field | Default | Effect |
|---|---|---|
| `output_dir` | `output/maps_{config_stem}` | Where the HTML lands. |
| `show_true_positives` | `true` | TP layer visible on load. |
| `show_false_positives` | `true` | FP visible on load. |
| `show_false_negatives` | `true` | FN visible on load. |
| `show_true_negatives` | `false` (some configs) | Hidden by default. |

`UP` / `UN` always render when unlabelled rows are present.

## Multi-class palettes

For 7-class multiclass runs, the color scheme is fixed in
[training/visualize.py](../../training/visualize.py) `MULTICLASS_COLORS`:

- 0 NotFarm — gray `#95a5a6`
- 1 Poultry: Meat — orange `#e67e22`
- 2 Poultry: Eggs — yellow `#f1c40f`
- 3 Poultry: Unspecified — red `#e74c3c`
- 4 Pigs — blue `#3498db`
- 5 Cattle — green `#27ae60`
- 6 Other — purple `#8e44ad`

For 3-class runs, the config can override with `model.class_names` +
`model.class_colors`.

## Typical timings

- Full ~140k-point HTML: ~30-60 s to generate; 80 MB file.
- Labeled-only (~17k points): ~5 s; 1-5 MB file.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `No scored_candidates.parquet found -- run inference first` | Inference didn't produce it | Check pipeline log; re-run inference. |
| Per-split panel missing eval / generalization rows | Old visualize.py before slot v5 fix | Ensure code is at `main`. |
| Points at Lat 0, Lng 0 | Missing lat/lng in candidate CSV | Rebuild candidates step. |
| Blank map with no points | Wrong `split` values (all `unknown`) | Re-run training so the splits CSV lands. |

## What's next

Nothing — visualize is the last stage. If you want richer analytics,
join `scored_candidates.parquet` to a notebook (see
[notebooks/](../../notebooks/)) or the `_metrics_per_country.json` files
directly.
