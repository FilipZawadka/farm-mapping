# Round_4 fleet — runtime observations

## 1. Throughput is I/O-bound, not compute-bound
Measured on `r4_a_s42` (RTX 4090, 4 pods sharing network volume `r8nyom4e4e`):

| Signal | Value |
|---|---|
| Epoch wall time | 180 s (1 epoch per 180 s sample, exact) |
| GPU utilisation | 0–1 % across 5 samples |
| GPU memory | 644 MiB |
| Load average | 4.6 |

GPU idle at 0 % while epochs advance means the bottleneck is patch reads from the
network volume, not the model. Two compounding causes:
- round_4 enlarged the training set **12,436 → 21,478 rows (+73 %)**, since Rachel set
  everything not reserved for generalization to train/val;
- four pods read the same network volume concurrently.

Wave 1 (smaller train set) ran ~60 s/epoch, so this is ~3× slower per epoch.

**Not fixed mid-fleet, deliberately** — see §2: dataloader worker count changes results,
so tuning it now would make arms incomparable. Candidate speedups for a future campaign:
stage the ~29k needed patches to the pod's local SSD before training, or raise
`dataloader_workers` (latency-bound reads scale well with concurrency) — but only if
applied uniformly to every arm.

## 2. Augmentation RNG is duplicated across DataLoader workers
`training/dataset.py:111` builds `self.rng = np.random.default_rng(rng_seed)` in the
parent process. With `dataloader_workers: 8`, every worker inherits a **fork-copy of the
same generator state**, so all 8 workers draw the *same* sequence of augmentation
parameters. This is the classic PyTorch numpy-RNG-in-workers pitfall (the documented fix
is a `worker_init_fn` that reseeds per worker).

Consequences:
- **Not a validity threat for this campaign** — every arm and every seed shares the
  behaviour, so A/B/C/D/E remain directly comparable.
- It does reduce *effective augmentation diversity* roughly 8-fold relative to intent.
- It means `dataloader_workers` is silently a **model hyperparameter**, not just a
  throughput knob: changing it changes which worker augments which sample, hence the
  augmentation stream, hence the trained weights. That is why it was left untouched
  mid-fleet.

**Follow-up experiment worth running** (not part of the current 15): proper per-worker
seeding via `worker_init_fn`. Given the paper's finding that the full augmentation stack
is doing real work, restoring intended augmentation diversity is a plausible free win —
and it must be judged against the seed band, like every other lever.

## 3. `freeze0` is a COMPOUND lever, not a single factor (found 2026-08-21, mid-fleet)

Triggered by arm A sitting at val F1 **0.640** while arm B sat at **0.824** — a gap far too
large to be the 5-epoch frozen head start, since A unfroze 8 epochs earlier.

`training/train.py:357-359`, the unfreeze transition:

```python
ctx.model.unfreeze_backbone()
optimizer = _make_optimizer(ctx.model, ctx.cfg, lr_scale=0.1)   # <-- 10x LR drop
scheduler = _build_scheduler(optimizer, ctx.cfg)                # <-- fresh cosine
```

The branch requires `freeze_backbone_epochs > 0`, so **it never executes for `freeze0`**:

| Arm | Epochs 1-5 | Epoch 6+ backbone LR |
|---|---|---|
| A (`freeze=5`, the v6/v9/production recipe) | head only @ 1e-4 | **1e-5** (0.1x, fresh cosine) |
| B (`freeze0`) | everything @ 1e-4 | **1e-4** (10x higher) |

Confirmed in the MLflow `lr` metric: `ep5=9.755e-05 -> ep6=9.990e-06`, an exact 10x drop at
the unfreeze epoch. A sweep of historical runs in `mlruns` shows almost all carry the ep6
drop; freeze0 runs are the rare exception.

**Consequence for the record.** The campaign's headline — "freeze0 is the only validated win
in 20 runs" (+0.0045 AUC, +5.9 sigma) — remains true as a statement about *recipes*, but its
stated *mechanism* is misattributed. `freeze0` changes two things at once: it removes the
frozen warm-up **and** trains the backbone at 10x the learning rate for the entire remaining
run. The second is very likely the dominant term. Any text claiming "freezing hurts" should
instead say "the freeze schedule's coupled 0.1x LR drop hurts".

**Untested and plausibly the actual best recipe:** freeze for 5 epochs, then unfreeze at
**full** LR (`lr_scale=1.0`) — warm-up benefits without the LR penalty. Nothing in the
20-run record tests this, because `lr_scale=0.1` is hard-coded rather than configurable.
Proposed as arm **F** (3 seeds, ~$7, ~3 h).

## 4. Arm E (DenseNet) is a three-way confound — verified buildable, interpret narrowly

Pre-flight smoke test on a live pod (CPU-only, container python: torch 2.4.1+cu124,
torchvision 0.19.1+cu124) — run BEFORE arm E reached the queue head so a broken builder
would not burn 3 x ~3 h of GPU time:

```
arch: densenet121 | hub: densenet121 | in_ch: 6
RESULT densenet121 in=6 -> (2, 4) params=7.0M  OK
d_s42  resnet50_softcon in=6 -> (2, 4) params=23.5M  OK
```

Both build and forward correctly. But E vs D differs in **three** ways at once, not one:

| Factor | D | E |
|---|---|---|
| Architecture | ResNet-50 | DenseNet-121 |
| Pre-training | SoftCon (Sentinel-2 native) | ImageNet (RGB) |
| Capacity | 23.5M params | **7.0M params (3.4x smaller)** |

The pretraining handicap was known and accepted when the run was requested (no S2-native
DenseNet weights exist). The 3.4x capacity gap was not, and it pushes the same direction.
So a loss for E is close to uninterpretable as an architecture verdict — it cannot separate
"DenseNet is worse" from "ImageNet init is worse" (the record's most robust result, ~+0.10
for S2-native) from "7M params is not enough". A *win* for E, by contrast, would be strongly
informative, since it would arrive despite two handicaps.

Report E accordingly: as an architecture-family datapoint under an explicit handicap, never
as "ResNet beats DenseNet".

## 5. The v6 -> r4 series comparison IS leakage-free (verified, not assumed)

The frozen blind benchmark was retired for round_4 because the split restructure
contaminated it. That retirement does **not** extend to the test/eval/generalization
slices, and this was checked rather than presumed.

**Provenance of every round_4 evaluation row, traced back through v9:**

| v10 slice | n | came from (v9) |
|---|---|---|
| generalization | 662 | 462 v9-generalization + 200 v9-`predict` (unlabelled then, labelled now) |
| test | 2164 | 2164 v9-test |
| eval | 590 | 590 v9-eval |

**Overlap with each archived model's train+val set:**

| | generalization | test | eval |
|---|---|---|---|
| v6 / v7 / v8 / v9 | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |

And the archived models really were trained on those splits — every archived run log
records `Splits (explicit)`, so `cnn_split_assigned` describes their actual training
data rather than a column added later:

```
v6  Splits (explicit): train=9728   val=2081  test=2094  eval=523  generalization=273
v9  Splits (explicit): train=12062  val=2666  test=2094  eval=523  generalization=426
r4  Splits (explicit): train=21478  val=5022  test=2094  eval=523  generalization=617
```

**Conclusion:** v6, v7, v8, v9 and every round_4 arm can be compared directly on all three
slices. The label-round series table is valid.

**Where round_4's +73% training data came from.** `qual_eval` went 16,663 (v6) -> 11,772 (v9)
-> **0** (r4): Rachel folded her qualitative-eval hold-out into train/val. Two consequences:
- `qual_eval` is no longer a hold-out and must never be used to compare round_4 models
  against archived ones — it would be 100% contaminated. `evaluate_r4.py` never reads it
  (the slice is empty and is dropped).
- This is also why `qual_eval_metrics.json` is never written for round_4, which silently
  broke the collector's completion marker (section 4 of the collector fix).

## 6. Pods cannot self-terminate — auto_terminate has never worked

Found when `r4_b_s42` sat idle and billing ~10 min after its pipeline logged
`Pipeline completed successfully`, still holding one of the four slots.

`training/auto_terminate.py` needs `RUNPOD_API_KEY` and `RUNPOD_POD_ID`:

```python
if runpod.api_key and pod_id:  runpod.terminate_pod(pod_id)
else:                          print("... not set, skipping auto-terminate")
```

The startup script *does* append it (`; python3 -m training.auto_terminate`), and the
`runpod` package *is* present (1.12.0). But `_RUNPOD_SECRETS_ENV` — the only env injected
into pods — carries just `GEE_SERVICE_ACCOUNT`, `GEE_PRIVATE_KEY_JSON` and
`GOOGLE_MAPS_API_KEY`. **`RUNPOD_API_KEY` is never passed**, so the branch always falls
through to "skipping". This is long-standing, not a round_4 regression, and it plausibly
explains the previously "stranded billing pod" incident.

Cost of the bug: a finished pod idles until the 30-minute no-progress watchdog fires
`runpodctl remove` — about **$0.37 and 30 minutes of slot latency per run**, i.e. ~$7 and
several hours of wall clock across 18 runs. Worse, if the watchdog path also fails, the
fleet stalls permanently at 4 pods and the remaining runs never launch.

**Fix: reap from the launcher, not the pod.** `launch_fleet.reap_finished()` terminates any
pod whose `scored_candidates.parquet` the collector has already pulled (120 s grace for the
trailing archive step). This keeps the API key on the laptop rather than shipping it to
every pod, and it also covers the case where the on-pod watchdog fails. Reaping is wrapped
in try/except so a failure can never take down the fleet.

## 7. Live demonstration that one seed is not a result (arm B, 2026-08-21)

Arm B's contrast against A was computed twice, once when B had a single collected
seed and again when all three had landed:

| B seeds | b - a | p_holm | verdict |
|---|---|---|---|
| 1 (0.8840) | +0.0499 | 0.003 | "b BETTER than a" |
| 3 (0.8840, 0.8699, **0.8169**) | +0.0229 | 0.287 | **not distinguishable** |

B's third seed landed 0.067 below its first. Pooled sigma_seed on generalization went
0.0094 -> **0.0224**, and the apparently decisive effect fell inside the noise band.
Nothing about the data or the code changed; only the number of seeds did.

This is the campaign's founding premise reproduced in real time, and it is worth
keeping as the canonical example: the historical record's verdicts were built on
exactly this kind of single-run comparison.

**Second finding, from the same numbers:** arm B's seed sd is **3.8x arm A's**
(0.0354 vs 0.0094). That is mechanistically coherent -- B trains the backbone at
1e-4 from epoch 1, where A trains it at 1e-5 after a warm-up (section 3). The
higher learning rate buys instability. It also raises the value of arm F, which
keeps the warm-up *and* the full rate: if the warm-up is what stabilises training,
F should show A-like variance with B-like or better mean.

**Correction issued at the same time:** an earlier note that arm A sits "~4 sigma"
below archived v9 on generalization used arm A's own within-arm sd (0.0094) as the
yardstick. Against the pooled seed sigma (0.0224) the -0.040 gap is ~1.5 sigma --
suggestive, not established. Judge cross-model gaps against the POOLED sigma, never
against one arm's internal spread, which understates it whenever arms differ in
stability.
