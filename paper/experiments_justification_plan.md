# Experiment plan: justifying every design decision

Companion to `paper/main.tex`. Every choice marked `†(Ex.y)` in the paper maps to an
entry here.

> **Execution status (2026-08-13).** A first pass was run; results are in
> `experiments/RESULTS.md` with raw output in `experiments/results/*.json`.
> No GPU was available (pod refused connections; no local CUDA device), so every
> experiment requiring training is **blocked**. All experiments answerable by
> re-analysing archived model outputs were **completed**:
>
> | | Status | Outcome |
> |---|---|---|
> | E0.1 | done (4-class) | benchmark frozen, n=11,365, CI half-width **0.003** |
> | E0.2 | diagnostic done | eval AUC inflated **+0.056** by proximity; blocked splits emitted |
> | E2.1 | **done** | **ship threshold 0.4 OOD**; per-country fitting fails |
> | E2.2 | **done** | reject — real but +0.0018 AUC, below practical floor |
> | E2.3 | partial | zero in-domain value; OOD claim untestable (10% coverage) |
> | E2.5 | diagnostic done | **hypothesis refuted** — trade is not leakage |
> | E2.6 | diagnostic done | shift confirmed and quantified per split |
> | E0.3, E1.1–E1.9 | **blocked** | needs GPU (~46 runs) |
> | E0.4, E2.4, E2.7 | **blocked** | needs annotation (~110 h) |
>
> Two findings change the plan itself: the frozen benchmark is powerful enough
> that trivial effects reach significance, so **every remaining decision rule
> needs a practical-effect floor, not just a significance test**; and E0.1b
> removed the main evidence for 128 px context, so **E1.2 is now open rather than
> nearly-decided** and rises in priority.

The organising claim: **at the current evaluation-set size (n=523, 95% CI ≈ ±0.05
macro-F1), most of our design decisions cannot be justified even in principle.**
Required sample size scales as `n ≈ 524·(0.05/Δ)²`, so detecting Δ=0.02 needs
n≈3,275. Nearly every intervention we have tried moves the metric by 0.01–0.05.
This is why Tier 0 comes first: running Tier 1 ablations on today's benchmark
would produce a table of ties, not a justification.

Each entry gives: **hypothesis · config delta · decision rule · cost · what it upgrades.**
Costs are in A40-GPU-hours for training runs (one production-config run ≈ 3–4 h)
plus annotation hours where relevant.

---

## Tier 0 — Measurement (blocking prerequisite)

Nothing in Tier 1 or Tier 2 is worth running before E0.1–E0.3 land. Together they
take the minimum detectable effect from ≈0.05 to ≈0.02 and, for the first time,
give every future Δ an error bar.

### E0.1 — Adopt the blind out-of-domain benchmark
- **Hypothesis.** A ~6× larger held-out pool separates comparisons that read as
  ties on the current eval set.
- **Delta.** Freeze the 3,297 labelled rows carrying `split=unlabeled` in
  `web/public/data/world_v9_softcon/points.csv` as a named, versioned benchmark
  slice; no model has trained on them. Additionally extract patches for the
  14,138 v5-labelled clusters that currently have none.
- **Caveat — do not adopt the raw slice as-is.** Its composition is not
  deployment-like (RUS contributes 438 near-uniformly NotFarm rows; POL 726 and
  AUS 696 dominate the positives), so pooled macro-F1 on it is prior-skewed.
  Rebalance or report per-country before treating it as the standing benchmark.
  This is also why the ctx128 +0.047 result (E1.2) is directionally trusted but
  its magnitude is not.
- **Decision rule.** Adopt permanently if the frozen slice reproduces the known
  ranking on the one comparison that already separates (S2-native vs ImageNet,
  ≈+0.10). Report all future results on it alongside `eval` and `generalization`.
- **Cost.** ~0 GPU-h for the freeze; ~1 day of Earth Engine extraction for the
  14k clusters; one scoring pass per archived model (~6 × 0.5 h).
- **Upgrades.** Every "statistical tie" in §5.1 becomes decidable. Direct
  evidence already exists that this matters: on this slice `ctx128` beats the
  production model by **+0.047**, a difference invisible on the current eval set.

### E0.2 — Spatially blocked splits + de-duplication, in one rebuild
- **Hypothesis.** In-domain metrics are inflated by spatial leakage; ~27% of eval
  clusters lie within one patch width (1.28 km) of a training cluster.
- **Delta.** Rebuild split assignment with (a) `candidate_id` de-duplication and
  (b) spatial blocking that forbids any held-out cluster within 1.28 km of a
  training cluster (cite Roberts et al. 2017; Ploton et al. 2020). Rachel's
  explicit `cnn_split_assigned` becomes the input to a blocking pass rather than
  the final word — coordinate this with her, it changes the shared split contract.
- **Decision rule.** Adopt unconditionally (this is a correctness fix, not a
  comparison). Publish the before/after delta as the leakage estimate — expected
  magnitude ~0.03 from de-dup alone, unknown from blocking.
- **Cost.** ~2 h engineering + one retrain of the production config (4 h).
- **Upgrades.** §6.3, and every in-domain number in the paper. **Also re-opens
  E2.5**: balanced sampling's eval-vs-gen trade may be partly a leakage artefact.

### E0.3 — Seed variance
- **Hypothesis.** Some fraction of the 0.01–0.05 effects in the record are seed noise.
- **Delta.** Production config, seeds {42, 43, 44, 45, 46}. Nothing else changes.
- **Decision rule.** Report σ per slice. Retroactively annotate every past verdict
  with Δ/σ; any past conclusion with |Δ| < 2σ is downgraded to "unmeasured".
- **Cost.** 5 runs ≈ 20 GPU-h. **Highest value-per-hour experiment in this document.**
- **Upgrades.** Every comparison in §5. We currently have *no* variance estimate
  anywhere in the experimental record.

### E0.4 — Double-annotation to bound the label ceiling
- **Hypothesis.** Part of the eval-set error is annotator disagreement, not model error.
- **Delta.** Two independent annotators re-label ~300 eval rows blind; compute
  Cohen's κ and inter-annotator F1.
- **Decision rule.** Inter-annotator F1 is the ceiling. Any model within noise of
  it on a class is done improving; stop optimising that class.
- **Cost.** ~10 annotation hours, 0 GPU-h.
- **Upgrades.** §6.2, §7. Likely resolves whether the Pigs→Poultry confusion is
  representational or annotational.

---

## Tier 1 — Un-ablated core choices

All of these are currently inherited defaults. Configs marked *(exists)* are
already written but were never run, or were run only on the ImageNet-era recipe
and need regeneration on the production recipe.

### E1.1 — Band set (justifies 9 channels + the three indices)
- **Hypothesis.** The 6-band + 3-index set beats cheaper subsets; the indices
  contribute beyond the bands they are derived from.
- **Delta.** SoftCon backbone, production recipe, varying `channel_subset`:
  `rgb` / `rgb+nir` / `rgb+ndwi` / `6bands` / `9ch` *(configs exist as
  `configs/rachel_clusters/ablation_*.yaml`, but ImageNet-era — regenerate)*.
  Plus one run with `recompute_indices: true`.
- **Decision rule.** Keep 9ch only if it beats `6bands` by >2σ (E0.3) on the blind
  slice. Note the a-priori concern: indices are deterministic functions of the
  bands, so a CNN could in principle derive them — but only `6bands` has
  pretrained counterparts, so the indices may be *harmful* dead channels.
  `recompute_indices` fixes a known inconsistency (per-band jitter perturbs bands
  without updating indices in ~30% of training samples) and should be adopted if
  neutral-or-better.
- **Cost.** 6 runs ≈ 24 GPU-h.
- **Upgrades.** §3.4, §4.2.

### E1.2 — Spatial context (may change the production model)
- **Hypothesis.** 128 px context beats the production 64 px crop.
- **Delta.** SoftCon, crop ∈ {48, 64, 128}. *(48/64/128 ablation configs exist.)*
- **Decision rule.** Adopt the winner on the blind slice + generalization.
- **Cost.** 3 runs ≈ 12 GPU-h.
- **Priority: highest in Tier 1.** Existing evidence already favours 128 px
  (gen +0.033 for SSL4EO; +0.047 on the blind slice — direction corroborated by
  gen disagreements resolving 2:1 to ctx128, magnitude unsettled per E0.1), and the only
  counter-evidence — that SoftCon×ctx128 fails to stack — came from the
  contaminated patch store. The production crop is 64 px for historical reasons,
  not measured ones.

### E1.3 — Cloud handling and temporal window
- **Hypothesis.** Per-pixel SCL masking with a relaxed scene threshold beats
  scene-level filtering at 15%; a multi-year window beats single-year.
- **Delta.** (a) `cloud_mask: scl` + `max_cloud_cover: 60` *(this is the blocked
  `world_v8_cloudfree` run — it died on a disk quota and was never retried)*;
  (b) 2022–2024 median vs 2023-only at cloud<15.
- **Decision rule.** Adopt (a) if ≥neutral, since it also eliminates the
  2,197-candidate coverage hole and would retire the `relaxed_cloud60` tier
  entirely. Adopt (b) if ≥neutral — it makes the recovery tiers unnecessary.
- **Cost.** Full re-extraction (~2 days EE) + 2 runs ≈ 8 GPU-h. Expensive because
  it changes the imagery hash and invalidates the patch cache.
- **Upgrades.** §3.4 and Table 3 (tiers). Note `composite: least_cloudy` is a
  dead knob in code (`.median()` is hardcoded) — either implement or delete it.

### E1.4 — Optimiser and schedule
- **Hypothesis.** LR 1e-4 / AdamW / cosine / 5-epoch freeze / 0.1× unfreeze scale
  are not at an optimum; they were never tuned.
- **Delta.** LR ∈ {3e-5, 1e-4, 3e-4} × freeze ∈ {0, 5} × unfreeze-scale ∈ {0.1, 1.0}.
  Run as a fractional design (6–8 runs), not full factorial.
- **Decision rule.** Adopt any setting beating the default by >2σ on the blind
  slice; otherwise declare the default justified-by-insensitivity, which is a
  legitimate and publishable outcome.
- **Cost.** 8 runs ≈ 32 GPU-h.

### E1.5 — Augmentation, term by term
- **Hypothesis.** The 9-term recipe contains terms that do nothing or hurt.
- **Delta.** Leave-one-out over the recipe (9 runs) + a geometric-only control
  (flips/rot90 alone) + single-lever cutout.
- **Decision rule.** Drop any term whose removal is neutral-or-better; simpler is
  better at equal performance.
- **Cost.** 11 runs ≈ 44 GPU-h. Consider running only the 4 photometric terms
  (brightness, per-band jitter, noise, channel dropout) if budget-constrained —
  the geometric terms are near-certainly beneficial for an orientation-arbitrary
  overhead task.
- **Upgrades.** Table 5. Also isolates cutout, which v10 only tested inside a
  three-way bundle.

### E1.6 — Architecture and backbone re-test
- **Hypothesis.** ResNet-50 depth and SoftCon-over-SSL4EO are justified.
- **Delta.** {ResNet-18, ResNet-50} × {SoftCon, SSL4EO} on the blind slice; add
  ViT-S if matched weights are available.
- **Decision rule.** Prefer the smaller/cheaper model at equal performance.
- **Cost.** 4 runs ≈ 14 GPU-h.
- **Note.** The production SoftCon-over-SSL4EO choice currently rests on external
  GEO-Bench evidence, not on our own measurements (our own Δ=0.015 is well inside
  CI). This experiment either confirms it on our task or reverses it.

### E1.7 — Checkpoint metric
- **Hypothesis.** Selecting on val macro-F1 beats selecting on val loss under 50:1 imbalance.
- **Delta.** `checkpoint_metric: val_loss` vs `val_f1`, same run otherwise.
- **Decision rule.** Keep val_f1 unless val_loss wins by >2σ.
- **Cost.** 2 runs ≈ 8 GPU-h (cheap; the change was made in v8 with a stated
  rationale but no A/B).

### E1.8 — Test-time augmentation
- **Hypothesis.** 8-fold dihedral TTA improves out-of-domain accuracy enough to
  justify 8× inference cost.
- **Delta.** `inference.tta: true` on existing checkpoints — **no retraining**.
- **Decision rule.** Adopt for the eval/gen slices only if Δ > 2σ; full-world
  deployment at 8× cost needs a larger margin.
- **Cost.** ~2 GPU-h. Implemented but never once enabled — cheapest untested lever
  in the codebase.

### E1.9 — Taxonomy (3-class vs 4-class)
- **Hypothesis.** Splitting OtherFarm into Pigs/Cattle neither helps nor hurts.
- **Delta.** Re-run both taxonomies on clean data + blind slice.
- **Decision rule.** If 4-class is neutral and Cattle stays unmeasurable (E2.4),
  merge back to 3-class — a nominal class with placeholder metrics is worse than
  an honest coarser one.
- **Cost.** 2 runs ≈ 8 GPU-h. Twice previously found "no effect", but both tests
  predate the blind benchmark and one predates the geofix.

---

## Tier 2 — Deployment and data

### E2.1 — Domain-conditional thresholds
- **Hypothesis.** A single global threshold is wrong; miscalibration is domain-conditional.
- **Delta.** No training. Fit per-country / per-domain thresholds on held-out
  data; publish a threshold table with the release.
- **Decision rule.** Ship if OOD recall improves ≥5 points at ≤5 points FPR.
  Existing evidence says yes: 0.5→0.3 buys +9 recall for +5 FPR, while a *global*
  retune is a no-op (val-optimal 0.481 vs default 0.5).
- **Cost.** ~0. **Best cost/benefit item in this document.**

### E2.2 — Ensembling
- **Hypothesis.** The v8+v9 probability ensemble is worth 2× inference cost.
- **Delta.** Mean-probability ensemble, evaluated on the blind slice.
- **Decision rule.** Ship only if Δ > 2σ. Measured gain so far is +0.009 AUC on
  one OOD pair — likely inside noise, and the v9+v10 ensemble was identical to v9
  alone.
- **Cost.** ~1 GPU-h (scoring only).

### E2.3 — Geometry fusion
- **Hypothesis.** The Isolation Forest's morphometric features add OOD value the
  CNN lacks.
- **Delta.** Late fusion of the 7 geometric features + `template_score_if` with
  CNN logits (small MLP or logistic blend). Note footprint features are currently
  dropped at `training/inference.py:118-127`.
- **Decision rule.** Ship if OOD gain > 2σ. In-domain a probe showed only +0.004,
  but geometry transfers where imagery does not (on Afghanistan, n=23, the
  geometric model beat the CNN 0.825 vs 0.667) — so evaluate on OOD only.
- **Cost.** ~4 h engineering, ~2 GPU-h.

### E2.4 — Cattle: acquire or merge
- **Hypothesis.** Cattle is unlearnable at 153 training rows.
- **Delta.** Either (a) targeted acquisition of ~500 Cattle labels in feedlot
  regions (AUS, ARG, BRA, USA), or (b) merge Cattle into a coarser class.
- **Decision rule.** Pre-register: if after (a) the OOD Cattle OvR-AUC stays
  below 0.7, adopt (b). Current OOD AUC is 0.485 — chance.
- **Cost.** ~40 annotation hours + 1 run, or ~0 for the merge.
- **Upgrades.** §5.7, §7. Blocks E1.9.

### E2.5 — Balanced sampling, re-tested on blocked splits
- **Hypothesis.** The replicated eval+/gen− trade is partly a spatial-leakage artefact.
- **Delta.** `balanced_class_sampling: true` vs `false` on E0.2 splits.
- **Decision rule.** If the eval gain vanishes under blocking, the trade was
  leakage and the "keep off" verdict is confirmed for a *different* reason than
  we currently state in §5.2.
- **Cost.** 2 runs ≈ 8 GPU-h. **Depends on E0.2.**

### E2.6 — Label-source shift
- **Hypothesis.** Train/eval degradation is a measurement-instrument change
  (98% registry / 56% USA in train vs 80% visual / 3% USA in eval), not overfitting.
- **Delta.** Train on registry-only vs visual-only vs mixed; evaluate each on both
  label-source strata.
- **Decision rule.** If the cross-source drop is symmetric, it is a genuine
  distribution shift and the fix is more visual labels in training. If asymmetric,
  the registry labels are noisier and should be down-weighted.
- **Cost.** 3 runs ≈ 12 GPU-h.

### E2.7 — Candidate-stage recall (the unmeasured quantity)
- **Hypothesis.** Morphometric filters calibrated on US broiler complexes miss
  facilities elsewhere — and this failure is invisible to every metric we report,
  because missed sites never become candidates.
- **Delta.** In 3–5 countries spanning construction practices (e.g. NGA, IDN, POL,
  BRA), exhaustively annotate a sample of grid cells for livestock facilities and
  measure what fraction the candidate stage produced.
- **Decision rule.** Report per-country candidate recall alongside classifier
  metrics. If recall < 0.7 anywhere, loosen the filters for that region and re-run.
- **Cost.** ~60 annotation hours, ~0 GPU-h.
- **Note.** Not in the original roadmap; added because §7 identifies it as the
  most important unmeasured quantity in the system. Every headline number is
  conditional on a recall we have never estimated outside Delmarva.

---

## Closed axes — do not re-run

Documented so the plan shows what is settled, and why.

| Axis | Verdict | Evidence |
|---|---|---|
| S2-native vs ImageNet pretraining | **Closed — S2-native wins** | +0.10 macro-F1 ≈ 2× CI; the 9-channel ImageNet control rules out band count as the cause |
| Post-hoc logit adjustment | Closed — no gain | τ sweep puts the optimum at τ=0 for the best models; the ceiling is the representation, not the decision rule |
| AdaBN / radiometric domain adaptation | Closed — harmful | gen macro-F1 −0.024; implies the OOD gap is morphological, not radiometric |
| Country-balanced sampling | Closed — structurally impossible | 26 train countries have ≤2 rows; inverse-frequency weights would draw a 1-row country ~100×/epoch |
| Split-rebalancing volume (cap 50→70) | Closed — saturated | +1,963 rows moved AUC by ≤0.004 on every slice |
| Negatives-only label campaigns | Closed — prior shift, not learning | AUC flat, matched-FPR recall identical, mean P(farm) fell for *both* true classes |
| Regularizer bundle (smoothing + cutout + longer schedule) | Closed as a bundle | ≤ production model everywhere that matters; ECE worsened 0.027→0.052. *Cutout alone is NOT closed — see E1.5* |

---

## Suggested sequencing

1. **E0.3** (seed variance, 20 GPU-h) and **E2.1** (thresholds, free) — run first;
   E0.3 is what makes every later number interpretable, E2.1 is a free deployment win.
2. **E0.1** + **E0.2** — the benchmark rebuild. Everything below is underpowered until these land.
3. **E1.2** (context) and **E1.8** (TTA) — cheapest paths to a real model improvement.
4. **E0.4**, **E2.4**, **E2.7** — the annotation campaigns; long lead time, so start
   them in parallel with step 2 rather than after it.
5. **E1.1**, **E1.5**, **E1.4**, **E1.6**, **E1.7** — the ablation sweep, once
   effects are detectable.
6. **E2.5**, **E2.6**, **E2.3**, **E2.2**, **E1.9**, **E1.3** — re-tests and
   deployment decisions.

Total: ≈200 GPU-hours and ≈110 annotation hours, excluding the E1.3 re-extraction.
