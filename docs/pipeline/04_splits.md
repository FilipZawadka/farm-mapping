# 04 — Splits (`build_splits`)

The single most important function in the pipeline. Decides which patches
enter train / val / test / inspected / eval / generalization / unlabeled.

## Entry point

Called at the top of
[training/train.py](../../training/train.py) `train()`:

```python
train_ds, val_ds, test_ds, inspected_ds, eval_ds, gen_ds = \
    build_splits(cfg, patches_dir=patches_dir)
```

Source:
[training/dataset.py](../../training/dataset.py) `build_splits` (~line
522).

## Inputs

- `data/patches/patch_meta.csv` (or `.parquet`) — one row per patch.
- Candidate CSVs from `cfg.data.candidates_dir` (labels + eval_set flags).
- Config knobs (see [09_configuration.md](09_configuration.md)):
  - `data.training_countries`
  - `data.generalization_countries`
  - `data.dmv_force_to_train_only`
  - `data.inspected_as_test`
  - `training.upsample_minority_regions`
  - `training.balanced_country_splits`
  - `training.balanced_class_sampling`
  - `training.val_split`, `training.test_split`
  - `training.seed`

## Flow

```
1. Filter meta to current imagery_config_hash
2. Join labels: meta["_label"] = candidate.label  (int; -1 = unlabelled)
3. Compute holdouts (in priority order):
   a) eval_idx   = rows where candidate.eval_set == 1
   b) gen_idx    = rows in ADM0 ∈ generalization_countries AND labelled
   c) Reassign  eval ∩ gen  →  gen  (generalization wins)
4. Build labeled_pool_mask:
   - labelled (_label != -1)
   - NOT in eval_set
   - NOT in generalization
   - IF cfg.data.training_countries is set:  ADM0 ∈ training_countries
     (rows outside get demoted to _label = -1 → "unlabeled")
5. Optional: DMV pin — rows whose label_source contains "DMV" are
   memoized; after the split assignment, any DMV row that landed in
   val/test/inspected is moved to train.
6. Route the labeled_pool:
   - IF inspected_as_test → split off viz_status==inspected as
     inspected_idx, then random or country-balanced split of the rest
     into train/val/test.
   - ELIF cfg.data.train_regions → region-based split.
   - ELIF balanced_country_splits → country-balanced (equal val/test/
     country).
   - ELSE random stratified split by class.
7. Belt-and-braces strip: drop any eval / gen / unlabelled rows that
   somehow leaked into the labelled splits.
8. Build split_col (one string per meta row):
   - default "unassigned"
   - overwrite with "train"/"val"/"test"
   - overwrite unlabelled rows with "unlabeled"
   - overwrite inspected rows with "inspected"
   - overwrite eval rows with "eval"  (always wins over train/val/test)
   - overwrite gen rows with "generalization"  (always wins)
9. Persist splits CSV to  data/patches/splits/{config_stem}.csv
10. Build PatchDataset for each labelled split; return the six datasets.
```

## Priority order (top wins)

`generalization` > `eval` > `inspected` > `train`/`val`/`test` >
`unlabeled` > `unassigned`.

The order matters because Rachel marks BGD/NGA rows as `eval_set=True` AND
they're in `generalization_countries`. Without the priority override, they'd
end up in both eval and generalization, contaminating the metric that's
supposed to measure in-distribution training-country holdout.

## Country-balanced split (v3/v4 default)

`_country_balanced_split_indices` (line ~438).

Given the current `labeled_pool_mask`, compute:

```
min_country_size = smallest labelled country's row count
n_val  = max(1, int(min_country_size * val_split))    # e.g. 1
n_test = max(1, int(min_country_size * test_split))   # e.g. 1
```

Every country contributes exactly `n_val` + `n_test` rows to val/test. The
rest goes to train. Total val = `n_val × n_countries`, total test =
`n_test × n_countries`.

**Gotcha**: if the smallest labelled country has 1 row, val = test =
1×countries. Enable `balanced_country_splits: false` on any config where
one training country has very few labelled rows.

## Random-stratified split (v6/v7 default)

Standard: within the `remaining_idx` pool (after eval/gen/inspected/DMV/
whitelist filters), pos/neg rows are shuffled and cut by
`test_split` / `val_split`. Result: train/val/test naturally reflect
class ratios.

## DMV pin

Enabled by `data.dmv_force_to_train_only: true`.

`label_source` contains "DMV" → the row is Rachel's Delmarva clean poultry
set. Rachel uses these to fit her Isolation Forest. If they show up in our
CNN val/test, the metric is artificially easy.

Implementation:
- Compute `dmv_idx` = rows in the labelled pool whose `label_source`
  contains "DMV".
- After the normal split assignment, move any DMV row from val/test/
  inspected into train.

## Sampler weights (training-only)

Two knobs, both applied at the DataLoader level via `WeightedRandomSampler`
(handled in [training/train.py](../../training/train.py)):

- `upsample_minority_regions: true` — each sample weighted by
  `1 / country_size`; smaller countries get drawn proportionally more.
- `balanced_class_sampling: true` — each sample weighted by
  `1 / class_size`. When both are on, the weights multiply.

Both act only on the train split; they don't change what rows go where.
v6/v7 turn both OFF to test the effect on Poultry vs OtherFarm confusion
(see [00_overview.md](00_overview.md) run-flavor table).

## Persisted splits CSV

Written to `data/patches/splits/{cfg._config_stem}.csv`:

```
candidate_id,split
AFG_cluster_168,unlabeled
BRA_cluster_1039,train
BRA_cluster_2013,val
USA_cluster_9999,train
BGD_cluster_5,generalization
NGA_cluster_50,generalization
...
```

This file is the source of truth for the `split` column that inference
attaches to `scored_candidates.parquet` (see
[07_inference.md](07_inference.md)).

## Common auditing queries

Row counts per split:

```python
import pandas as pd
sp = pd.read_csv("data/patches/splits/world_v7_three_class.csv")
sp.split.value_counts().sort_index()
```

Verify no eval_set leak into train:

```python
cand = pd.concat([pd.read_csv(f) for f in Path(candidates_dir).glob("*.csv")])
eval_ids = set(cand[cand.eval_set==1].id.astype(str))
train_ids = set(sp[sp.split=="train"].candidate_id.astype(str))
assert not (eval_ids & train_ids)
```

Verify BGD/NGA never in train/val/test:

```python
sp["adm0"] = sp.candidate_id.str.split("_").str[0]
assert sp[sp.adm0.isin(["BGD","NGA"]) & sp.split.isin(["train","val","test"])].empty
```

The audit script
[scripts/audit_eval_splits.py](../../scripts/audit_eval_splits.py) runs
these checks + a per-country comparison against Rachel's source parquet.

## Log lines to sanity-check

When training kicks off, `build_splits` prints:

```
eval_set hold-out: 912 candidates (excluded from train/val/test/inspected)
generalization hold-out: 273 candidates from ['BGD', 'NGA'] ...
eval_set ∩ generalization: 260 rows reassigned to generalization
training-country whitelist: demoting 1072 labelled rows outside ['BRA', 'CHL', 'MEX', 'THA', 'USA'] to the unlabeled split
DMV protection: 1200 rows pinned to train
DMV protection: moved 14 rows from val/test/inspected to train
Inspected hold-out: 0 candidates (excluded from train/val/test)
Country-balanced splits: 5 countries, 61 val/country, 61 test/country ...
Splits (inspected_holdout): train=13297  val=308  test=307  inspected=0  eval=652  generalization=273  (pos ratio: -0.77)
Saved split assignments to data/patches/splits/world_v5_three_class.csv
```

The last line is the ground truth; anything else is diagnostic.
