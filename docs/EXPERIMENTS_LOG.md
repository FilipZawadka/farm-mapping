# Experiments log — world_v3 → world_v8

A running record of every world_v* experiment since `world_v3`. Each
section lists **what changed vs the previous flavor**, **why**, and the
actual **eval / generalization / test metrics** where a run finished.

For the codebase pieces (splits, patch cache, launcher…) see
[`pipeline/`](pipeline/README.md). For the current active matrix and the
research that shaped it see [`EXPERIMENTS_v8.md`](EXPERIMENTS_v8.md).
For the post-v8 diagnosis and the ranked v9 plan see
[`IMPROVEMENT_ROADMAP.md`](IMPROVEMENT_ROADMAP.md).

---

## Summary table (eval-set macro-F1, three_class)

| Config | eval F1 | eval f1_class0 (NotFarm) | eval f1_class1 (Poultry) | eval f1_class2 (OtherFarm) | gen F1 | gen f1_class2 | Notes |
|---|---|---|---|---|---|---|---|
| v3_three_class | 0.534 | 0.687 | 0.615 | 0.299 | — | — | Baseline; balancing ON, no country whitelist |
| v3_binary      | 0.900 | — | — | — | — | — | Farm vs NotFarm (not three-class) |
| v3_multiclass  | 0.205 | 0.593 | 0.097 | 0.080 | — | — | 7-class taxonomy; too little data per class |
| v4_three_class | 0.433 | 0.549 | 0.486 | 0.264 | 0.322 | 0.000 | Adds BGD/NGA generalization + `eval_set` framework |
| v5_three_class | 0.424 | 0.542 | 0.468 | 0.262 | 0.280 | 0.000 | Strict `training_countries`; DMV pinned to train |
| v6_three_class | 0.388 | 0.412 | 0.752 | **0.000** | 0.368 | **0.000** | Balancing OFF — Poultry recovers, OtherFarm collapses. **Invalid** (see §Postmortem) |
| v7_three_class | *n/a* | — | — | **0.000** | — | **0.000** | v6 + weight `[1, 0.7, 2.0]`. **Invalid** (same bug) |
| v8_three_class | 0.403 | 0.444 | 0.747 | 0.018 | 0.397 | 0.000 | Bugfix rerun; features fine on test (f1_c2=0.45) but eval undershoot |
| v8_ssl4eo      | **0.489** | 0.475 | 0.724 | **0.269** | **0.407** | 0.000 | S2-native pretrained backbone — wins on every headline |
| v8_logitadj    | 0.470 | 0.549 | 0.596 | 0.265 | 0.330 | 0.000 | Train-time logit-adjusted CE; matches ssl4eo on eval OtherFarm |
| v8_crt         | 0.494 | 0.495 | 0.723 | 0.264 | 0.426 | 0.000 | ssl4eo backbone frozen, head retrained balanced — best v8 overall (gen +0.019) |
| v8_cloudfree   | *blocked (disk)* | — | — | — | — | — | Partial extraction hit disk quota mid-run |
| v9_softcon     | **0.504** | 0.482 | 0.729 | **0.300** | 0.402 | 0.000 | SoftCon RN50 backbone — **best eval + best eval OtherFarm in the log** |
| v9_ctx128      | 0.484 | 0.445 | 0.728 | 0.278 | **0.440** | 0.000 | ssl4eo + 128px context crop — **best generalization in the log** |
| v9_softcon_ctx128 | 0.492 | 0.479 | 0.706 | 0.291 | 0.378 | 0.000 | Both levers combined — best *test* (0.751) but neither eval nor gen; **they don't stack** |
| v9_imagenet9ch | 0.393 | 0.379 | 0.731 | 0.068 | 0.382 | 0.000 | ImageNet RN50 + 9 bands — ablation: **pretraining, not band count, drives the SSL win** |
| v9_softcon_crt | 0.497 | 0.487 | 0.715 | 0.291 | 0.382 | 0.000 | cRT head-rebalance on softcon — **NEGATIVE**: eval −0.007, gen −0.020 vs softcon-plain (cRT helped ssl4eo, not softcon) |

Eval = Rachel's per-country representative sample (~100–150 rows / country
in the training-country set). Gen = BGD/NGA held-out OOD (present from v4
onwards). Both use the label routing enforced by
[`build_splits`](../training/dataset.py) — see
[`pipeline/04_splits.md`](pipeline/04_splits.md).

> **Why every `gen f1_class2` is 0.000**: BGD + NGA have only **4
> labelled OtherFarm rows** in total (58 NotFarm, 211 Poultry, 4
> OtherFarm — from the v8_ssl4eo `build_splits` log). With 4 positives,
> a model with even moderate OtherFarm precision on the training set
> still has ~90% probability of scoring 0/4 recall by chance — every
> v4-onwards run confirms this. Not diagnostic of model quality; needs
> more BGD/NGA OtherFarm labels to become a useful signal.

---

## world_v3 — Rachel's eval framework (2026-06-22)

**Commit**: `188e1bb Add world_v3 pipeline: Rachel's eval_set holdout + per-country eval metrics`

### What's new
- `all_clusters_v3.parquet` — Rachel's rebuilt per-country parquets for the
  five training countries with the column renames
  (`viz_label → visual_label`, `modified_label → final_label`) and two new
  columns: **`label_source`**, **`eval_set`** (bool).
- Split routing now recognises `eval_set == 1` and routes those rows to a
  new `eval` split. They **must never** enter train/val/test/inspected.
- Post-train evaluator: `eval_metrics.json` +
  `eval_metrics_per_country.json` written next to the existing inspected
  ones.
- Three config heads shipped in parallel: `world_v3_binary`,
  `world_v3_three_class`, `world_v3_multiclass`.

### Results

| Head | eval F1 | eval acc | test F1 | notes |
|---|---|---|---|---|
| binary | **0.900** | 0.842 | 0.941 | Strong baseline; Poultry ↔ NotFarm decision is easy |
| three_class | 0.534 | 0.554 | 0.699 | Poultry↔OtherFarm confusion visible (f1_class1=0.62, f1_class2=0.30) |
| multiclass (7-class) | 0.205 | 0.275 | 0.335 | Too little data per class; Poultry-Meat + Pigs are the two least-catastrophic classes |

### Lessons
- The binary head is production-grade; further improvement will need
  either more diverse data or explicit false-positive suppression.
- The three-class head is our real experiment surface — everything from
  v4 forward is a three-class variation.
- The 7-class head is dead-on-arrival with the current data volume.

Related earlier fixes bundled with v3:
- `33d135e fix(dataset): strip unlabeled rows (label=-1) from train/val/test/inspected` — CUDA assertion `t < n_classes` from unlabelled rows leaking into the labelled pool.
- `794351f fix(dataset): filter unlabeled rows BEFORE country-balanced split` — val=2/test=2 pathological split when unlabelled rows dominated a country.

---

## world_v4 — BGD/NGA OOD + auto-rebuild + self-heal (2026-06-30)

**Commits**: `d9b4b59 Add world_v4 three_class + eval framework + label propagation + audit script` · `47d618f Wire world_v4 launch end-to-end: seed BGD/NGA + auto-rebuild master + self-heal git`

### What's new
- `all_clusters_v4.parquet` — v3 master + **Bangladesh and Nigeria** as
  labelled countries, imported from Rachel's separate exports.
- `data.generalization_countries` config field. `build_splits` routes any
  labelled row in a listed country to a new `generalization` split, never
  entering train/val/test.
- `data_seed/` — small parquet copies committed to the repo for BGD/NGA;
  the launcher auto-rebuilds `all_clusters_v4.parquet` on the volume if
  the file is missing.
- **Self-heal git** in `runpod_launch.py`: on `git fetch` failure the
  startup script rebuilds `.git` via `/tmp` and moves it onto the volume.
- `scripts/audit_eval_splits.py` — per-country parity check across the
  master, candidate CSVs, splits CSV, and scored parquet.
- Label propagation: `original_label`, `standardized_label`,
  `visual_label`, `label_source`, `notes`, `eval_set`, `random_sample`,
  `viz_status` — all forwarded from candidates through to the scored
  parquet, so we can audit bad labels without joining back to Rachel's
  files.

Follow-up fix: `7dc7e6a fix(dataset): generalization takes priority over eval_set on conflict` — a BGD row marked `eval_set==True` was being put in `eval` instead of `generalization`.

### Results

| Slice | acc | F1 | f1_class1 (Poultry) | f1_class2 (OtherFarm) |
|---|---|---|---|---|
| test (headline) | 0.528 | 0.510 | 0.601 | **0.348** |
| eval (per-country sample) | 0.439 | 0.433 | 0.486 | 0.264 |
| generalization (BGD+NGA) | 0.436 | 0.322 | 0.538 | 0.000 |

### Lessons
- Adding BGD/NGA to the master **doesn't** improve headline test — same
  ballpark as v3 — but exposes the OOD gap: generalization F1 = 0.32.
- OtherFarm F1 = 0 on generalization is unsurprising (BGD/NGA have few
  labelled OtherFarm rows). Not the same failure mode as the later v6/v7
  collapse; here it's a data-coverage issue.

---

## world_v5 — Strict framework enforcement (2026-06-30)

**Commit**: `5309228 Add world_v5 three_class: strict training_countries + DMV-pinned-to-train`

### What's new
- **`data.training_countries: [USA, BRA, CHL, MEX, THA]`** — anything
  outside this whitelist is demoted from labelled splits to `unlabeled`.
  Fixes the previously-implicit "labelled countries" set.
- **`data.dmv_force_to_train_only: true`** — Rachel's Delmarva Peninsula
  poultry-barn dataset (`label_source contains "DMV"`) is pinned to
  train, never val/test/inspected. Reserved for her Isolation Forest.
- MLflow log_model re-enabled (`1ccb680`) once the network volume got
  a +50 GB quota bump. Best model snapshotted as an MLflow artifact
  again.

### Config recipe (unchanged from v4 otherwise)
```yaml
training:
  balanced_country_splits: true    # ON
  balanced_class_sampling: true    # ON
  channel_subset: [B2, B3, B4, NDWI]
  crop_center_px: 64
inference:
  labeled_only: false              # world map
```

### Results

| Slice | acc | F1 | f1_class1 (Poultry) | f1_class2 (OtherFarm) |
|---|---|---|---|---|
| test | 0.547 | 0.547 | 0.583 | **0.435** |
| eval | 0.425 | 0.424 | 0.468 | 0.262 |
| generalization | 0.344 | 0.280 | 0.420 | 0.000 |

### Lessons
- Enforcing the country whitelist **doesn't hurt headline test** — the
  demoted labelled rows outside the whitelist weren't buying much.
- **Poultry↔OtherFarm confusion is now clearly visible** in the eval
  matrix. Rachel's read of the meeting deck: OtherFarm is being drawn
  ~4.2× per row by `balanced_class_sampling`, and the model treats them
  interchangeably with Poultry.
- **Generalization F1 down** vs v4 (0.28 vs 0.32) — the country whitelist
  hurt OOD transfer slightly (less data variety at train time).

---

## world_v6 — Balancing off (2026-07-01)

**Commit**: `b46d615 Add world_v6 three_class: all balancing OFF (natural distributions)`

### What's new
- All three balancers turned **off**:
  ```yaml
  training:
    upsample_minority_regions: false
    balanced_country_splits: false
    balanced_class_sampling: false
  ```
  Goal: undo the Poultry↔OtherFarm confusion by letting the model see the
  natural class distribution.

Also: `20383ec Add --labeled-only inference flag` — inference on ~124k rest-of-world rows was a MooseFS I/O bottleneck (30–60 min); the flag drops those and cuts inference to ~2 min.

### Results

| Slice | acc | F1 | f1_class1 (Poultry) | f1_class2 (OtherFarm) |
|---|---|---|---|---|
| test (binary-format metric — see notes) | 0.842 | 0.896 | — | — |
| eval | 0.627 | 0.388 | **0.752** | **0.000** |
| generalization | 0.648 | 0.368 | 0.763 | **0.000** |

### Observations at the time
- Binary rollup jumps hugely (Poultry|OtherFarm) — model does bright/dark
  well.
- Poultry F1 hits 0.75 — best in the whole log for that class.
- OtherFarm F1 = 0 — the model puts every OtherFarm example into
  NotFarm or Poultry.

Initial reading: "balancing off is a win for Poultry, and OtherFarm
collapse is a class-imbalance side-effect — needs class_weight."

**Real reading (postmortem)**: see below — this was not a class-imbalance
symptom at all.

---

## world_v7 — Class-weight rescue attempt (2026-07-01)

**Commit**: `eb25fda Add world_v7 three_class: v6 recipe + light class weight to rescue OtherFarm`

### What's new
- `training.class_weight: [1.0, 0.7, 2.0]` — try to nudge the loss toward
  OtherFarm while leaving Poultry alone.

### Result
- **OtherFarm F1 still 0.** The class weight had nothing to act on.

---

## Postmortem: v6/v7 were invalid experiments

Root cause discovered while planning v8:
[`training/dataset.py`](../training/dataset.py) `_stratified_class_split`
stratified train/val/test **only over `label == 0` vs `label == 1`**. Any
row with `label == 2` (OtherFarm) fell into neither pool and was silently
dropped from train/val/test.

v3/v4/v5 dodged the bug because `balanced_country_splits: true` routed
through a different, class-agnostic splitter. v6 (and by inheritance v7)
turned that off and hit the buggy path.

**Consequences**:
- v6 and v7 models **never saw a single OtherFarm example during training**.
  F1_class2 = 0 was inevitable.
- v7's class_weight had zero training examples for class 2 to weight.
- Every conclusion in the earlier v6/v7 summaries about "natural
  distribution" or "OtherFarm needs stronger weight" is void.

**Fix** (see [`EXPERIMENTS_v8.md`](EXPERIMENTS_v8.md) §1.1): stratified
split now covers **all** classes; `build_splits` raises when any class
has 0 train rows.

While auditing we also cleaned up:

| Issue | Fix |
|---|---|
| No per-pixel cloud masking (only scene-level filter) | `patches.cloud_mask: scl` option (SCL 3/8/9/10 masked before median). Hashes only when enabled. |
| No input normalization (ImageNet backbone fed raw reflectances) | `training.normalization: per_channel` with train-split mean/std, persisted to a norm-stats JSON. |
| Checkpoint on weighted val **loss** | `training.checkpoint_metric: val_f1` (macro). |
| `f1_class{i}` index shifting when a class was absent from a slice | Fixed range 0..max; confusion matrix persisted per JSON. |
| `num_workers=0` everywhere | `training.dataloader_workers` with proper per-worker RNG reseeding. |
| Resume never unfroze the backbone if resumed before the unfreeze epoch | Transition keyed on `start_epoch`, not "was this a resume". |
| torchgeo pretrained weights adapted by first-k conv channels | Band-NAME-mapped adaptation (`model.pretrained_band_order`); indices mean-init. |

Also two launcher-side fixes surfaced during v8 rollout:
- `7f361ce Set github_branch: more_cnn_tests in v8 configs` — the launcher's
  fallback re-clone was hardcoded to `main`; concurrent pod launches raced
  on `.git/index.lock`, fallback fired, and the volume was reset to v7.
- `641c182 fix(patch_extraction): filter skip_ids by imagery_config_hash` —
  `patch_meta.csv` was consulted without hash filtering, so a fresh
  imagery config (e.g. `cloud_mask: scl` in `world_v8_cloudfree`) skipped
  every candidate that had *any* previous patch.

---

## world_v8 matrix (2026-07-02 → 2026-07-03)

Full spec in [`EXPERIMENTS_v8.md`](EXPERIMENTS_v8.md). Seven experiments
targeting the OtherFarm collapse + weak OOD transfer.

### Training runs — full metrics

Each three-class value is `(NotFarm / Poultry / OtherFarm)`.

| Run | test macroF1 | test per-class | eval macroF1 | eval per-class | gen macroF1 | gen per-class |
|---|---|---|---|---|---|---|
| **v8_three_class** | 0.626 | 0.619 / 0.812 / 0.448 | 0.403 | 0.444 / 0.747 / 0.018 | 0.397 | 0.360 / 0.831 / 0.000 |
| **v8_logitadj**    | 0.572 | 0.589 / 0.709 / 0.418 | 0.470 | 0.549 / 0.596 / 0.265 | 0.330 | 0.417 / 0.574 / 0.000 |
| **v8_ssl4eo**      | 0.723 | 0.711 / 0.842 / **0.617** | 0.489 | 0.475 / 0.724 / 0.269 | 0.407 | 0.427 / 0.793 / 0.000 |
| **v8_crt**         | **0.727** | 0.733 / 0.842 / 0.606 | **0.494** | 0.495 / 0.723 / 0.264 | **0.426** | 0.457 / 0.820 / 0.000 |
| **v8_cloudfree** *(blocked: disk quota exceeded mid-extraction)* | — | — | — | — | — | — |

### Post-hoc experiments (2026-07-03)

**#2 logit_adjust_sweep — τ swept on val macro-F1** (see
[`data/output/*/logit_adjust_report.json`](../data/output) on the volume):

| Base run | Selected τ | val macroF1 (best) | eval macroF1 baseline → adjusted | eval f1_c2 baseline → adjusted |
|---|---|---|---|---|
| v8_three_class | 0.25 | 0.634 → 0.644 | 0.403 → 0.404 | **0.018 → 0.029** |
| v8_logitadj    | 0.00 | 0.632 | 0.470 → 0.470 | 0.265 → 0.265 |
| v8_ssl4eo      | 0.00 | 0.754 | 0.489 → 0.489 | 0.269 → 0.269 |

Read: **logit adjustment does not fix v8_three_class's eval OtherFarm
collapse** (0.018 → 0.029 is arithmetic, not rescue). And ssl4eo /
logitadj are already at the τ=0 optimum — their decision rules are
well-calibrated. This means **the ceiling in v8_three_class is the
feature representation itself**, not the classifier — consistent with
Menon et al.'s "features first, decision second" framing.

**#6 adabn_adapt on ssl4eo** (report:
[`data/output/world_v8_ssl4eo/adabn_report.json`](../data/output/world_v8_ssl4eo/)):

| Slice | Before AdaBN | After AdaBN | Δ |
|---|---|---|---|
| generalization macroF1 | 0.407 | 0.383 | **−0.024** |
| generalization f1_class0 | 0.427 | 0.364 | −0.063 |
| generalization f1_class1 | 0.793 | 0.787 | −0.006 |
| generalization f1_class2 | 0.000 | 0.000 | 0 (mechanical) |
| training-country test macroF1 | 0.723 | 0.690 | −0.033 |
| training-country test f1_class2 | 0.617 | 0.612 | −0.005 |

**Negative result — but diagnostic.** AdaBN re-estimates BatchNorm
running stats on BGD/NGA target patches; the theory is that BN stats ARE
the source-domain radiometry. If the OOD gap were radiometric, AdaBN
would lift generalization macroF1 by several points. Here it moved
**backward** on both target (generalization) and source (test) —
indicating:

- Source BN stats already fit target radiometry well (SSL4EO's
  self-supervised pretraining on 250k S2 sites likely already saw
  BGD/NGA-like scenes).
- Real OOD gap is **morphological / label-distribution shift**, not
  spectral: building footprints, cluster sizes, and OtherFarm-vs-Poultry
  frequencies in BGD/NGA differ, and pixel statistics don't capture that.

The natural next lever for OOD is either (a) 50–200 labelled BGD/NGA
clusters (biggest evidence-backed lift for OOD in the literature),
(b) footprint-geometry fusion (see EXPERIMENTS_v8.md §4), or (c) MixStyle
+ heavier photometric augmentation on the source. AdaBN is off the
critical path.

### Key readings

- **Bugfix confirmed**. v8_three_class test f1_class2 = 0.448 vs v6/v7's
  0.000. The class-stratified split fix works end-to-end.
- **SSL4EO wins on every headline**: test macroF1 0.723 (+0.10 over
  three_class), eval 0.489 (+0.09), gen 0.407 (+0.01), test OtherFarm F1
  0.617 (+0.17). *S2-native pretraining is the single biggest lever* —
  the research prediction held.
- **cRT gives a small OOD lift** (gen macroF1 0.407 → **0.426**) but no
  eval OtherFarm rescue (0.269 → 0.264). Overall the winner overall, but
  only marginally — the ssl4eo backbone was already carrying almost all
  the signal.
- **v8_three_class's eval f1_class2 = 0.018 is the puzzle**. Test f1_c2 =
  0.448 says the model *can* learn OtherFarm. Eval f1_c2 = 0.018 says on
  Rachel's representative slice it barely predicts OtherFarm at all.
  Confusion matrix on eval: 98 true OtherFarm rows, 1 predicted as
  OtherFarm. Distribution shift between train and eval sets, not a class
  imbalance problem — a Rachel-side per-country audit of what
  distinguishes eval-OtherFarm from train-OtherFarm is the natural next
  step.
- **logitadj matches ssl4eo on eval OtherFarm F1** (0.265 vs 0.269). The
  training-time prior adjustment gets ~85% of the way to ssl4eo's
  OtherFarm precision on eval — but ssl4eo still wins overall because
  its features are better (test macroF1 gap: 0.15).

### Config/artefact recovery notes (learned the hard way)

- **Launcher branch fix** (`7f361ce`): staggered pod launches 90s apart to
  avoid `.git/index.lock` races that fire the fallback re-clone and
  clobber the volume back to `main` (which lags `more_cnn_tests` by two
  commits).
- **Patch-cache hash fix** (`641c182`): `skip_ids` was consulted without
  imagery-hash filtering, so `world_v8_cloudfree` skipped every candidate
  that had any previous patch under any hash.
- **v8_crt retargeted to ssl4eo** (`3779c28`): originally the crt config
  resumed from `three_class`'s backbone. Once ssl4eo won on test, we
  re-pointed to `world_v8_ssl4eo/best_model.pt` and matched architecture
  / band order / normalization / channel_subset — cRT only makes sense on
  top of the best features available.

---

## world_v9 matrix (2026-07-07)

Four single-variable runs off `world_v8_ssl4eo`, launched together on L4
pods. **`data:`/`patches:` byte-identical to v8** (same imagery hash +
seed 42 ⇒ identical split membership), so every number here is directly
comparable to the v8 table. Motivation + provenance in
[`IMPROVEMENT_ROADMAP.md`](IMPROVEMENT_ROADMAP.md). Judge on **eval**
(Rachel's deployment-representative slice) and **gen** (BGD/NGA OOD), not
test.

### Training runs — full metrics

Each three-class value is `(NotFarm / Poultry / OtherFarm)`.

| Run | test macroF1 | test per-class | eval macroF1 | eval per-class | gen macroF1 | gen per-class |
|---|---|---|---|---|---|---|
| *v8_ssl4eo (baseline)* | 0.723 | 0.711 / 0.842 / 0.617 | 0.489 | 0.475 / 0.724 / 0.269 | 0.407 | 0.427 / 0.793 / 0.000 |
| *v8_crt (prev. best)*  | 0.727 | 0.733 / 0.842 / 0.606 | 0.494 | 0.495 / 0.723 / 0.264 | 0.426 | 0.457 / 0.820 / 0.000 |
| **v9_softcon**        | 0.724 | 0.706 / 0.841 / 0.624 | **0.504** | 0.482 / 0.729 / **0.300** | 0.402 | 0.404 / 0.801 / 0.000 |
| **v9_ctx128**         | 0.729 | 0.715 / 0.847 / 0.625 | 0.484 | 0.445 / 0.728 / 0.278 | **0.440** | 0.469 / 0.852 / 0.000 |
| **v9_softcon_ctx128** | **0.751** | 0.734 / 0.856 / **0.663** | 0.492 | 0.479 / 0.706 / 0.291 | 0.378 | 0.367 / 0.766 / 0.000 |
| **v9_imagenet9ch**    | 0.617 | 0.586 / 0.810 / 0.456 | 0.393 | 0.379 / 0.731 / 0.068 | 0.382 | 0.306 / 0.839 / 0.000 |
| **v9_softcon_crt**    | 0.736 | 0.725 / 0.844 / 0.639 | 0.497 | 0.487 / 0.715 / 0.291 | 0.382 | 0.393 / 0.752 / 0.000 |

### Key readings

- **SoftCon is the new eval winner.** `v9_softcon` (MoCo → SoftCon RN50,
  per-band z-scored input) posts the **best eval macroF1 (0.504)** *and*
  the **best eval OtherFarm F1 (0.300)** in the whole log — over v8_crt
  (0.494 / 0.264) and v8_ssl4eo (0.489 / 0.269). Modest but real, and it
  lands on the two metrics that matter most (deployment slice + hardest
  class). Per-country the OtherFarm gain is Thailand 0.386, Mexico 0.364,
  Brazil 0.214, Chile 0.195; US OtherFarm stays weak (0.133, only 3 eval
  rows). test macroF1 is a dead heat with ssl4eo (0.724 vs 0.723).

- **128px context is the OOD winner.** `v9_ctx128` (ssl4eo backbone,
  `crop_center_px: 128` instead of 64) posts the **best generalization
  macroF1 (0.440)**, over v8_crt's 0.426. 1.28 km of context helps
  morphological transfer to BGD/NGA — consistent with the v8 AdaBN
  finding that the OOD gap is morphological, not spectral. It does *not*
  help the training-country eval slice (0.484 < softcon's 0.504).

- **The two levers do NOT stack.** `v9_softcon_ctx128` wins **test**
  (0.751, best in the log, OtherFarm 0.663) — but that's the wrong
  target. Its eval (0.492) is *below* softcon-alone (0.504) and its gen
  (0.378) is the **worst** of the four SSL-family runs, far under
  ctx128-alone (0.440). Stacking both levers overfits the
  training-country test distribution and *degrades* both deployment
  metrics. softcon helps eval, ctx128 helps gen, and combining them
  cancels on both.

- **imagenet9ch settles the pretraining-vs-band-count confound.** The
  v8_ssl4eo win over v8_three_class changed two things at once (S2-native
  pretraining AND 4→9 channels). `v9_imagenet9ch` holds bands at 9 and
  swaps only the backbone back to ImageNet: test **0.617** / eval
  **0.393** — statistically indistinguishable from v8_three_class
  (ImageNet, 4-band subset: 0.626 / 0.403) and *far* below every
  S2-native run (~0.72 / ~0.49–0.50). Adding five channels to ImageNet
  weights bought **nothing**; if anything test dipped. **The entire
  SSL4EO/SoftCon advantage is S2-native pretraining, not channel count.**
  This closes the open question from the v8 postmortem.

- **gen OtherFarm F1 = 0.000 across all four**, as in every run since v4
  — BGD/NGA have only 4 labelled OtherFarm rows total. Not diagnostic;
  see the summary-table note.

### cRT on softcon — negative (2026-07-07)

`world_v9_softcon_crt` freezes the softcon backbone and retrains the head
class-balanced (clone of `world_v8_crt`, `resume_from`
`world_v9_softcon/best_model.pt`, `normalization: per_channel`
recomputed identically from the frozen split). Resume loaded cleanly;
6 epochs, early-stopped, best val_f1 at epoch 1.

| Slice | v9_softcon (parent) | v9_softcon_crt | Δ |
|---|---|---|---|
| test macroF1 | 0.724 | 0.736 | +0.012 |
| eval macroF1 | **0.504** | 0.497 | **−0.007** |
| eval OtherFarm | **0.300** | 0.291 | **−0.009** |
| gen macroF1 | 0.402 | 0.382 | **−0.020** |

cRT did the **opposite** of what it did for ssl4eo (v8: gen +0.019). It
pushed the in-distribution **test up** but pulled **eval and gen down**.
Softcon's natural-sampling head already sat at the log's best eval
operating point (0.504); rebalancing shifted the boundary toward the
in-distribution test set and away from the shifted eval/gen
distributions. **cRT is off softcon's critical path.**

### Verdict + where the representation axis lands

Best models by slice, after the full v9 sweep:

- **eval / eval-OtherFarm → `world_v9_softcon`** (0.504 / 0.300) — the
  new deployment default.
- **generalization (OOD) → `world_v9_ctx128`** (0.440).
- The two don't compose (`softcon_ctx128`), and head-rebalancing
  (`softcon_crt`) doesn't lift softcon.

The backbone / crop / head-retrain axis is now **plateaued at eval
≈ 0.50** — softcon beats ssl4eo by only +0.015 eval, and every
head/context variant trades one deployment slice for another. This is
exactly what the [roadmap](IMPROVEMENT_ROADMAP.md) predicted: the eval
ceiling is a **label-source + geography shift** (train OtherFarm 98%
registry / 56% USA vs eval OtherFarm 80% visual / 97% non-US), not a
feature-representation problem — and AdaBN + logit-adjust + now cRT all
confirm the classifier/representation is not where the remaining points
are. The next real lever is the **data/label axis**, not another
backbone: (a) a Rachel-side eval-vs-train OtherFarm label audit, (b)
50–200 more labelled BGD/NGA clusters to make gen f1_class2 measurable,
or (c) self-training on the 125,812-row unlabeled pool. Ship
`world_v9_softcon` as the current best; stop tuning backbones.

### Validity caveat (see `notebooks/model_evaluation_analysis.ipynb`, 2026-07-07)

A full re-analysis from the raw `scored_candidates.parquet` (predictions
re-derived, splits + metrics audited) confirms the metric code is sound
(every eval/gen JSON reproduces exactly; splits are 100% stable across
all v8/v9 runs; DMV pinned; no cluster in two splits) — **but the eval
table above is optimistic and the fine ranking is not statistically
resolvable**:

- **Duplicate rows** (same `candidate_id`/coords/label survive de-dup)
  inflate every labelled split: deduplicating eval drops macro-F1 by
  ~0.03 (softcon 0.504 → 0.469). val is duplicated too → mild
  checkpoint-selection bias.
- **After dedup + 95% bootstrap CIs**, the top six S2-native models
  (softcon, ssl4eo, crt, ctx128, softcon_ctx128) sit at **0.466–0.470
  with CI width ~0.10 — a statistical tie.** The only robust eval signal
  is **S2-native pretraining vs ImageNet (+~0.10, ≈2× CI)**. Read the v9
  "winners" above as *ties*, not a pecking order.
- **~27% of eval clusters** lie within a patch-width (1.28 km) of a train
  cluster (no spatial-block splitting) → eval is mildly leakage-optimistic;
  generalization (cross-country) is the cleaner OOD number.

Fixes queued: de-dup the patch/candidate store; spatially-blocked splits;
label BGD/NGA OtherFarm.

---

## Cross-run inventory of scripts + configs added since v3

Configs (all under [`configs/rachel_clusters/`](../configs/rachel_clusters/)):
`world_v3_binary.yaml`, `world_v3_multiclass.yaml`, `world_v3_three_class.yaml`,
`world_v4_three_class.yaml`, `world_v5_three_class.yaml`,
`world_v6_three_class.yaml`, `world_v7_three_class.yaml`,
`world_v8_three_class.yaml`, `world_v8_crt.yaml`,
`world_v8_logitadj.yaml`, `world_v8_ssl4eo.yaml`,
`world_v8_cloudfree.yaml`,
`world_v9_softcon.yaml`, `world_v9_ctx128.yaml`,
`world_v9_softcon_ctx128.yaml`, `world_v9_imagenet9ch.yaml`.

Scripts (all under [`scripts/`](../scripts/)):
- `merge_clusters_v3.py`, `merge_clusters_v4.py` — build the master parquet
  from Rachel's per-country files + rest-of-world unlabelled.
- `audit_eval_splits.py` — per-country parity check (source → candidates
  → splits → scored).
- `post_hoc_evaluate.py` — re-run inspected / eval / generalization from a
  saved checkpoint (used when a pod dies between train and eval).
- `logit_adjust_sweep.py` — post-hoc τ sweep on saved probabilities.
- `adabn_adapt.py` — BatchNorm re-estimation on unlabelled target patches.

Core code deltas: see the diffs on each commit listed in the postmortem
section for the exact splits/model/training/normalization changes.

---

## v10 — Rachel's explicit splits + the OtherFarm experiment matrix (2026-07-14/15)

**Split regime change.** Rachel added `cnn_split_assigned`
(train/val/test/eval/generalization/predict) to the for_analysis files —
now the single source of truth in `build_splits()` (pass-through, gated on
column presence; no RNG). All runs below share identical splits:
train=10633 / val=2268 / test=2290 / eval=653 / gen=273 (patch rows).
`all_clusters_v5.parquet` + `scripts/merge_clusters_v5.py` carry the column;
DMV rows are now spread across train/val/test by Rachel (the
`dmv_force_to_train_only` flag is a no-op on this path). v3/v9_softcon/
v9_ctx128 were re-run on these splits; v3's old headline numbers turned out
to be a small-sample artifact of the old 66-per-country quota test set.

**Baselines on v10 splits** (dedup, macro-F1 — test / eval / gen):
v9_softcon **0.712 / 0.462 / 0.400**; v9_ctx128(SSL4EO) 0.710 / 0.469 / 0.415;
v3(ImageNet 4ch) 0.526 / 0.399 / 0.298.
Diagnosis motivating the matrix: dominant eval failure = OtherFarm→Poultry
(71% of eval Pigs predicted Poultry); train OtherFarm is a 10:1 pigs:cattle
merge; Poultry:Pigs+Cattle = 4:1 in train.

**Experiment matrix** (single levers off v9_softcon; 3-class-comparable
numbers, fourclass collapsed Cattle→OtherFarm):

| run | test | eval | gen | eval Pigs→Poultry | verdict |
|---|---|---|---|---|---|
| `world_v10_fourclass_softcon` | 0.716 | 0.460 | 0.373 | 74% | **No effect.** Splitting Pigs/Cattle didn't de-dilute the pig signature; Cattle (131 train rows) was never predicted once. Refutes the merge-dilution hypothesis. |
| `world_v10_softcon_balanced` | 0.690 | 0.440 | 0.330 | 44% | **Boundary moves, quality doesn't.** OtherFarm recall 0.23→0.39, but precision 0.21→0.15 (F1 flat 0.22) and Poultry recall 0.82→0.55; gen −0.07. Use only if pig-farm *recall* is the deployment goal. |
| `world_v10_softcon_ctx128` | 0.721 | 0.457 | 0.409 | 77% | Best test F1 of any run, but eval/gen tie the single-lever baselines — SoftCon+context not additive (matches the old-split finding). |

**Takeaway.** Neither taxonomy (4-class) nor imbalance (balanced sampling)
is the eval bottleneck. Eval pigs are missed at similar rates whether
registry-labelled (18% correct) or visually-labelled (24%) — the gap is
train-vs-eval *population* shift (registry CAFOs vs Rachel's representative
sample), consistent with AdaBN/logit-adjust/cRT all failing earlier. Next
lever should be data-centric: more representative-sample OtherFarm labels
in train (Rachel's call), or spatially/visually harder positives.

---

## 2026-07-20 — CRITICAL: stale ID-keyed patch store invalidates all runs after original world_v3 ("geofix")

**Found by Filip** via a single spot check: `MEX_cluster_16331` scored
farm=0.963 at (25.637, -101.831) — a real, obvious egg farm — but Rachel's
current file puts that cluster_id at (19.234, -96.186), a NotFarm. The model
was shown one location's image with another location's label.

**Mechanism.** Patches are keyed by `cluster_id` (+ imagery_config_hash) at
every join point — `{id}.npy` filenames, `patch_meta.csv`, label attach —
with no spatial check, and `patch_extraction.py` skips any id already in the
store. The store (hash `cc5a6ebb502a`, extracted ~03-18) predates the
**06-23 merge re-run that renumbered 93.5% of cluster_ids** (see 07-08
diagnosis; parquet geometry+label content was unchanged — only the
id↔cluster mapping churned). Every candidates build after 06-23 therefore
silently reused patches of different physical clusters.

**Blast radius** (patch coords vs current v5 centroid, >250 m = stale;
median displacement 100s of km; rest-of-world control 0.04%):
v1/v2 parquets 0.0% · v3_0618 snapshot 1.8% · v3_0623/v4/v5 **34.3–34.6%**.
Per-country (labeled rows): USA 37% / BRA 36% / CHL 48% / MEX 38% / THA 21%
/ BGD 21% / NGA 32% — across ALL splits incl. eval + generalization.
**Clean:** baseline_v2/world_v2 era + original world_v3 (trained 06-21 on
old ids — why "old model 3" genuinely looked better; the v3→v4 eval cliff
0.534→0.433 coincides exactly with misalignment onset). **Contaminated:**
world_v4→v9 and every v10 run above, incl. both Mexico runs. The MEX binary
run had 949/2495 (38%) wrong-location rows; among 384 with known
old-location labels, 84 (22%) were outright binary image/label
contradictions. All v10-matrix conclusions above are unreliable until
re-run; measured metrics since 06-23 are floors (many "false positives"
were the model correctly describing the image it was shown).

**Fix.**
1. `training/config.py validate_patch_locations()` (+ `haversine_m`,
   `MAX_PATCH_COORD_DRIFT_M=250`): dedup patch_meta keep-last per
   candidate_id and drop rows whose stored extraction coords are >250 m
   from the candidate's current coords. Wired into `dataset.py
   build_splits` and `inference.py score_candidates` (raise if >5% stale —
   store needs re-extraction) and into the `patch_extraction.py`
   skip-cache (stale ids re-extract instead of being skipped; the appended
   row supersedes via keep-last). Also retires the Thailand ×3 duplicate
   artifact (append-only meta now deduped at every consumer).
2. Store repaired by re-running candidates + patch_extraction against v5
   (~24.7k stale patches re-extracted at their current locations).
3. Re-runs (same config stems, `run_name: *_geofix`): world_v3_three_class,
   world_v9_softcon, world_v9_ctx128, world_v10_mex_binary_bal. Old
   contaminated metrics preserved in notebooks/results_cache/ + the tables
   above; website entries to be republished from the geofix runs.
4. Process note: treat Rachel's cluster_ids as **ephemeral** — any of her
   merge re-runs may renumber them (BGD/NGA show a second churn event).
   Never join her exports to our stores by id across pulls; the coordinate
   guard now enforces this mechanically.

---

## 2026-07-21 — geofix results in: all 7 runs republished; matrix verdict reverses

All 7 planned geofix re-runs completed (the 3 baselines, Mexico binary, and
the 3-experiment matrix). `world_v3_three_class`'s first relaunch stalled
12+ hours with zero checkpoints on an apparently-bad pod/GPU (every sibling
finished in <70 min); killed and retried cleanly on a fresh pod.

**Corrected numbers (test / eval / generalization macro-F1, geofix vs the
contaminated numbers above):**
world_v9_softcon 0.890/0.631/0.486 (was 0.712/0.462/0.400);
world_v9_ctx128(SSL4EO) 0.902/0.641/0.443 (was 0.710/0.469/0.415);
world_v3_three_class(ImageNet 4ch) 0.716/0.523/0.397 (was 0.526/0.399/0.298);
Mexico binary AUC 0.99 test / 0.93 eval (was 0.74 / 0.73). Every baseline
improved substantially, confirming the misaligned patches were suppressing
real model quality, not just adding noise.

**Experiment-matrix verdict reverses under corrected data** (vs the
world_v9_softcon geofix baseline; fourclass collapses Pigs+Cattle back to
OtherFarm for an apples-to-apples comparison):

| run | eval macro-F1 | eval OtherFarm→Poultry | verdict |
|---|---|---|---|
| fourclass_softcon | 0.657 (collapsed) | Pigs→Poultry 53% | **No effect**, confirming the original (contaminated-data) finding survives: splitting Pigs/Cattle doesn't touch the confusion (52%→53%, noise). |
| softcon_balanced | **0.658** | **38%** | **Real win** — reverses the old conclusion ("boundary moves, quality doesn't"). Confusion 52%→38%, eval F1 +0.027, test flat. The old verdict was measured on scrambled image/label pairs; balanced sampling's effect was there all along. |
| softcon_ctx128 | 0.648 | 42% | **Partial win**, also reversed from "not additive" — confusion 52%→42%, eval F1 +0.017. |

**Updated takeaway.** The imbalance lever (balanced_class_sampling) is a
real, meaningful fix for the OtherFarm→Poultry confusion — the opposite of
what the contaminated-data run showed. The taxonomy lever (4-class split)
still does nothing, on clean data as on dirty. Recommend folding
balanced_class_sampling into the next baseline by default and combining it
with the wider context crop (both independently helped; untested together).

**Website fixes alongside republishing:** `EvalBreakdown.tsx` hardcoded
"3-class" in its view toggle and footer caption regardless of the actual
release — wrong for the 4-class release, now derived from `classes`. Every
release's `model` field was a generic placeholder regardless of actual
backbone (SoftCon/SSL4EO/ImageNet) — now set per-release. The 3
experiment-matrix releases only stated their hypothesis, never a result —
now carry the verdicts above. Every release also got its exact archived
`config.yaml` attached (viewable/downloadable from the site) and generic
(not hand-allowlisted) column forwarding end to end — see
`training/rachel_to_candidates.py`, `training/inference.py`,
`web/scripts/export_dataset.py`, `web/src/lib/download.ts`.
