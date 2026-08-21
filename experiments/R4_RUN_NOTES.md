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
