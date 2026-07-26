# 05 — Training

Fit the CNN. Writes `best_model.pt`, `training_metrics.json`, and (usually)
the four post-training metric files.

## Entry point

```bash
python -m training.train --config configs/rachel_clusters/<name>.yaml
```

Source:
[training/train.py](../../training/train.py) `train()`.

## What happens

```
1. Resolve config, seed RNGs.
2. Load / cache patches (patches_dir from cfg.patches.output_dir).
3. build_splits (see 04) → train_ds, val_ds, test_ds, inspected_ds,
                            eval_ds, gen_ds.
4. Build DataLoaders:
   - Optional WeightedRandomSampler on train_loader when sampling knobs
     are on.
   - val_loader, test_loader, inspected_loader always shuffle=False.
5. build_model(cfg.model):
   - HuggingFace ResNet-50 (or ConvNeXt / EfficientNet / Swin / Satlas).
   - Adapt first conv 3ch → 4ch (or 9ch) by tiling the RGB weights.
   - Replace 1000-class head with num_classes (2, 3, or 7).
   - Freeze backbone for freeze_backbone_epochs.
6. Resume-from-checkpoint (optional): load prior weights + optimizer state
   from cfg.training.resume_from.
7. Class weight (optional): CrossEntropyLoss weight tensor from
   cfg.training.class_weight.
8. Mixed precision: torch.amp.GradScaler when mixed_precision + CUDA.
9. Train loop: for epoch in range(prior_epoch, epochs):
     - _train_one_epoch (with AMP)
     - _evaluate on val
     - Save best_model.pt when val_loss improves
     - Save last_ckpt.pt every epoch
     - Early stop after early_stopping_patience epochs of no improvement.
10. _save_test_results:
    - Load best_model.pt
    - Evaluate on test → training_metrics.json
    - mlflow.log_artifact(best_model.pt) IF cfg.mlflow.log_model.
11. Inspected pass (if inspected_ds present).
12. Eval pass (if eval_ds present) + per-country breakdown.
13. Generalization pass (if gen_ds present) + per-country breakdown.
```

## Model builder

Source:
[training/model.py](../../training/model.py).

- Loads the HF checkpoint via `AutoModelForImageClassification`.
- `_adapt_first_conv` — replicates the pretrained 3-channel RGB weights
  across the target input channels. E.g. 4-channel input: first 3 chans
  get the RGB weights, 4th gets the average of RGB.
- `_replace_head` — swaps the classifier head to `num_classes`.
- `freeze_backbone()` — sets `requires_grad = False` on every parameter
  except the classifier head.

Backbone is unfrozen once `epoch >= freeze_backbone_epochs`. Resume-from-
checkpoint sees whether prior training already crossed that threshold.

## Data augmentation

Configured under `training.augmentation` (see `AugmentationConfig` in
[training/config.py](../../training/config.py)).

Applied only in `train_ds`; val/test/inspected/eval/gen see raw patches.

Available knobs:

| Knob | Effect |
|---|---|
| `horizontal_flip` | Random left-right flip, per-image. |
| `vertical_flip` | Random top-bottom flip. |
| `random_rotation_90` | 90-degree rotation multiples (preserves grid). |
| `continuous_rotation` | Small arbitrary-angle rotation, `max_degrees` cap. |
| `random_resized_crop` | Zoom-in crop then resize back — mimics scale variance. |
| `brightness_jitter` | Multiplicative brightness. |
| `per_band_jitter` | Independent gain per spectral band — S2 sensor drift. |
| `gaussian_noise` | Additive noise, `sigma` in normalized-band units. |
| `channel_dropout` | Zero out a random channel, `max_channels` at a time. |
| `cutout` | Zero out a small square, `n_holes` × `hole_size` px. |
| `recompute_indices` | After spectral augs, re-derive NDVI/NDBI/NDWI. |

Every world_v* config runs the same "standard" recipe: flips + rotations +
resized crop + brightness/band jitter + light noise + channel dropout.

## Ablation knobs

Two lightweight training-time ablations (no re-extraction needed):

- `training.channel_subset: [B2, B3, B4, NDWI]` — the training loop feeds
  only these channels to the model. Must set
  `model.input_channels = len(channel_subset)`.
- `training.crop_center_px: 64` — center-crop the 128×128 patches down to
  64×64 (smaller receptive field, 640 m vs 1.28 km context).

All world_v* configs use `[B2, B3, B4, NDWI]` + 64-px crop.

## Scheduler + optimizer

- Optimizer: AdamW with `learning_rate` and `weight_decay` from config.
- Scheduler: `cosine` (default), `step`, or `plateau`.
- LR reset when unfreezing the backbone (halves the LR).

## Class weight

`cfg.training.class_weight: [w_0, w_1, w_2, ...]` becomes a
`torch.tensor(weight)` passed to CrossEntropyLoss. Per-sample gradient
scales by `w_true_class`. Used in v7 to try `[1.0, 0.7, 2.0]`; found
insufficient to force OtherFarm predictions.

## MLflow logging

Every metric — `train_loss`, `val_loss`, `val_accuracy`, `val_precision`,
`val_recall`, `val_f1`, `val_f1_class{0..k}`, `lr` — is logged per epoch
under `cfg.mlflow.experiment_name`.

`cfg.mlflow.log_model: true` also logs `best_model.pt` as an artifact
(~270 MB per run; matters for volume quota).

## Outputs

Under `data/output/{cfg._config_stem}/`:

| File | Written by | Content |
|---|---|---|
| `best_model.pt` | train loop (each val improvement) | ResNet-50 checkpoint |
| `last_ckpt.pt` | train loop (each epoch) | latest snapshot |
| `training_metrics.json` | `_save_test_results` | test-slice metrics |
| `inspected_metrics.json` | inspected block | inspected-slice metrics |
| `eval_metrics.json` | eval block | Rachel's representative sample |
| `eval_metrics_per_country.json` | eval block | per-country breakdown |
| `generalization_metrics.json` | gen block | OOD holdout |
| `generalization_metrics_per_country.json` | gen block | per-country |

The per-country files iterate every country present in the dataset, run
`_evaluate` on the subset, and dump `{country: {accuracy, precision,
recall, f1, f1_class0, ..., n}}`.

## Typical timing

- Full 50-epoch three_class on RTX 4090: ~4 h.
- Same on L4: ~5 h.
- Post-training evaluators: 2-5 min total.
- Inference (see step 07) is a separate ~5 min - 60 min depending on
  labeled-only vs full-world.

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA error: nll_loss_forward Assertion 't >= 0 && t < n_classes' failed` | unlabelled row (label=-1) leaked into train | Ensure `build_splits` strips unlabelled — see step 04. |
| val_loss climbs from epoch ~30 | Overfit; late cosine LR too low | Cap epochs at 40, or shorten `early_stopping_patience` to 5-7. |
| test F1 great but eval F1 collapses | Test slice contaminated by DMV | Set `dmv_force_to_train_only: true`. |
| Post-train JSONs missing but best_model.pt exists | Volume disk quota hit at `mlflow.log_artifact` | Free space; re-run via [scripts/post_hoc_evaluate.py](../../scripts/post_hoc_evaluate.py). |

## What the next stage needs

- `best_model.pt` in `data/output/{cfg._config_stem}/`.
- Splits CSV at `data/patches/splits/{cfg._config_stem}.csv` (written by
  build_splits).
- Candidate CSVs in `cfg.data.candidates_dir` (labels + diagnostic cols).
