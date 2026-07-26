# 11 — Audit + debug playbook

Common queries, sanity checks, and recovery paths. Refer here when a run
finishes but the numbers look wrong, or when a pod dies mid-pipeline.

## Sanity checks after any training run

Once `scored_candidates.parquet` lands, run these:

### 1. Unique-candidate counts per split

```python
import geopandas as gpd
df = gpd.read_parquet("data/output/world_v7_three_class/scored_candidates.parquet")
df.drop_duplicates("candidate_id").split.value_counts().sort_index()
```

Expected shape (v7, labeled_only):

```
eval                 652
generalization       273
inspected              0        # v5+ demotes non-training-country inspected
test                 306
train              13297
val                  308
```

### 2. Mutual exclusion

No candidate should appear in more than one labelled split.

```python
for a, b in [("train","test"),("train","eval"),("train","generalization"),
             ("eval","generalization"),("inspected","eval")]:
    sa = set(df[df.split==a].candidate_id.astype(str))
    sb = set(df[df.split==b].candidate_id.astype(str))
    assert not (sa & sb), f"OVERLAP {a} ∩ {b}: {len(sa & sb)}"
```

### 3. eval_set → eval or generalization

```python
c = df.drop_duplicates("candidate_id")
leak = c[(c.eval_set == 1) & ~c.split.isin(["eval","generalization"])]
assert leak.empty, f"eval_set rows outside eval/gen: {len(leak)}"
```

### 4. Generalization countries only in gen

```python
c["adm0"] = c.candidate_id.str.split("_").str[0]
gen_only = c[c.adm0.isin(["BGD","NGA"]) & c.split.isin(["train","val","test","inspected","eval"])]
assert gen_only.empty, f"BGD/NGA leaked into non-gen splits: {len(gen_only)}"
```

### 5. Training-country whitelist

```python
disp_to_iso = {"United States":"USA","Brazil":"BRA","Chile":"CHL",
               "Mexico":"MEX","Thailand":"THA"}
TRAIN = {"USA","BRA","CHL","MEX","THA"}
c["display_iso"] = c.country.map(disp_to_iso).fillna(c.adm0)
non_allowed = c[c.split.isin(["train","val","test","inspected"]) & ~c.display_iso.isin(TRAIN)]
assert non_allowed.empty, f"non-training countries in labelled splits: {len(non_allowed)}"
```

### 6. DMV never in val/test/inspected (if enabled)

```python
if "label_source" in c.columns:
    dmv = c[c.label_source.astype(str).str.contains("DMV", case=False, na=False)]
    escaped = dmv[dmv.split.isin(["val","test","inspected"])]
    assert escaped.empty, f"DMV escaped to holdout: {len(escaped)}"
```

### 7. Row-count parity — source parquet vs splits CSV

Run
[scripts/audit_eval_splits.py](../../scripts/audit_eval_splits.py):

```bash
python scripts/audit_eval_splits.py \
    --master data/rachel_geometry_candidates/all_countries/all_clusters_v4.parquet \
    --candidates-dir data/rachel_geometry_candidates/candidates_world_v7_three_class \
    --splits-csv data/patches/splits/world_v7_three_class.csv \
    --scored data/output/world_v7_three_class/scored_candidates.parquet
```

Prints a per-country table:

```
      source  candidate_csv  splits_csv  scored  match
ADM0
BGD      182            182         171     171   True
BRA      101            101          92      92   True
CHL      145            145         135     135   True
MEX      136            136         121     121   True
NGA      103            103         102     102   True
THA      106            106          84      84   True
USA      102            102          92      92   True
```

`match=True` per row means the counts agree across all four stages (up to
the "missing patch" gap explained below).

## Common count mismatches — explanations

- **Source > candidate CSV**: OSM-farm exclusion (`exclude_osm_farms`) or
  `exclude_labels` dropped rows.
- **Candidate CSV > splits CSV**: candidates without a patch in
  `patch_meta.csv`. Usually EE returned "no patch in previous runs" or
  "Image.select: no bands" — see [03_patch_extraction.md](03_patch_extraction.md).
- **Splits CSV > scored**: inference used a different
  `imagery_config_hash` (config changed). Re-run inference or fix the
  config.
- **Splits CSV = scored but per-country n differs from Rachel's notebook**:
  Rachel's notebook was run against a different snapshot of the master.
  Ours is authoritative — check by opening the source parquet directly.

## Confusion matrices

Three-class:

```python
NAMES = {0:"NotFarm", 1:"Poultry", 2:"OtherFarm"}
for split in ("test","eval","generalization"):
    s = df[df.split == split]
    print(pd.crosstab(s.true_label.map(NAMES), s.predicted_label.map(NAMES),
                      rownames=["TRUE"], colnames=["PRED"], margins=True))
```

Binary rollup:

```python
df["tb"] = df.true_label.isin([1, 2]).astype(int)
df["pb"] = df.predicted_label.isin([1, 2]).astype(int)
```

Per-country:

```python
for c, sub in df[df.split == "eval"].groupby("country"):
    tp = int(((sub.tb == 1) & (sub.pb == 1)).sum())
    fp = int(((sub.tb == 0) & (sub.pb == 1)).sum())
    fn = int(((sub.tb == 1) & (sub.pb == 0)).sum())
    tn = int(((sub.tb == 0) & (sub.pb == 0)).sum())
    ...
```

## Recovery playbook

### The pod died mid-training

Best model may or may not be on the volume — check
`data/output/{stem}/best_model.pt`.

- If present: run inspected / eval / generalization from it via
  [scripts/post_hoc_evaluate.py](../../scripts/post_hoc_evaluate.py),
  then `python -m training.run_pipeline --steps inference visualize
  --config <cfg>`.
- If missing: relaunch the pod; `run_pipeline` resumes from
  `patch_extraction` completed state (splits step re-runs; training
  restarts unless you set `resume_from` in the config).

### The training finished but the metrics JSONs are missing

Usually a volume-quota fault at `mlflow.log_artifact`. Sequence:

1. SSH to a relay pod that mounts the same volume.
2. Free space (delete `last_ckpt.pt`, redundant mlflow best_model.pt
   artifact copies).
3. Run `python -m scripts.post_hoc_evaluate --config <cfg>`.
4. Run `python -m training.run_pipeline --steps inference visualize
   --config <cfg>` for the map.

### Inference is stuck at 0% GPU / 100% wait

MooseFS read bottleneck. Options:

- Restart with `--labeled-only`: ~8× fewer patches, finishes in 2-5 min.
- Bump `num_workers` in the inference DataLoader (see
  [training/inference.py:180](../../training/inference.py#L180)).

### The pod won't boot (empty `_latest_startup.log`)

Volume quota is the usual culprit. Self-heal git tries to rebuild `.git`
in `/tmp` and move it back, but `mv` needs quota. Free 300+ MB by
deleting redundant checkpoints on the volume before relaunching.

## Live monitoring

### Tail the training log

```bash
ssh -p $POD_PORT root@$POD_HOST \
  'tail -F $(ls -td /workspace/farm-mapping/runs/<config>/pipeline/*/ | head -1)/train.log' \
  | grep --line-buffered "Epoch"
```

### MLflow UI

```bash
# On the relay pod
tmux new-session -d -s mlflow \
  '/workspace/farm-venv-cpu/bin/mlflow ui --port 5000 --host 0.0.0.0 --backend-store-uri ./mlruns'

# On your laptop
ssh -N -L 5000:localhost:5000 -p $POD_PORT root@$POD_HOST
# Open http://localhost:5000
```

### Attach to the pod tmux

```bash
ssh -t -p $POD_PORT root@$POD_HOST 'tmux attach -t prep'
```

Detach with `Ctrl-B D`. Do **not** Ctrl-C — that kills the training.

## Config sanity checks before launching

Before launching a new v* config:

- Config stem matches a fresh `candidates_dir` and unique output paths?
- `mlflow.log_model` matches the current volume quota?
- `cloud_type: SECURE` on `world_v*` configs?
- `parquet_source` points at an existing master (`all_clusters_v4`)?
- `training_countries` and `generalization_countries` non-empty for
  world_v5+?
- `inference.labeled_only` matches intent (`false` = world map,
  `true` = fast eval-only)?
