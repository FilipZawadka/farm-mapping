# Experiments log — world_v3 → world_v8

A running record of every world_v* experiment since `world_v3`. Each
section lists **what changed vs the previous flavor**, **why**, and the
actual **eval / generalization / test metrics** where a run finished.

For the codebase pieces (splits, patch cache, launcher…) see
[`pipeline/`](pipeline/README.md). For the current active matrix and the
research that shaped it see [`EXPERIMENTS_v8.md`](EXPERIMENTS_v8.md).

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
| v8_crt         | *in flight* | — | — | — | — | — | Freeze ssl4eo backbone, retrain head with balanced sampler |
| v8_cloudfree   | *blocked (disk)* | — | — | — | — | — | Partial extraction hit disk quota mid-run |

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
| **v8_ssl4eo**      | **0.723** | 0.711 / 0.842 / **0.617** | **0.489** | 0.475 / 0.724 / 0.269 | **0.407** | 0.427 / 0.793 / 0.000 |
| **v8_crt** *(in flight)* | — | — | — | — | — | — |
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

**#6 adabn_adapt on ssl4eo** — pending pod completion; results will land in
[`data/output/world_v8_ssl4eo/adabn_report.json`](../data/output/world_v8_ssl4eo/).

### Key readings

- **Bugfix confirmed**. v8_three_class test f1_class2 = 0.448 vs v6/v7's
  0.000. The class-stratified split fix works end-to-end.
- **SSL4EO wins on every headline**: test macroF1 0.723 (+0.10 over
  three_class), eval 0.489 (+0.09), gen 0.407 (+0.01), test OtherFarm F1
  0.617 (+0.17). *S2-native pretraining is the single biggest lever* —
  the research prediction held.
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

## Cross-run inventory of scripts + configs added since v3

Configs (all under [`configs/rachel_clusters/`](../configs/rachel_clusters/)):
`world_v3_binary.yaml`, `world_v3_multiclass.yaml`, `world_v3_three_class.yaml`,
`world_v4_three_class.yaml`, `world_v5_three_class.yaml`,
`world_v6_three_class.yaml`, `world_v7_three_class.yaml`,
`world_v8_three_class.yaml`, `world_v8_crt.yaml`,
`world_v8_logitadj.yaml`, `world_v8_ssl4eo.yaml`,
`world_v8_cloudfree.yaml`.

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
