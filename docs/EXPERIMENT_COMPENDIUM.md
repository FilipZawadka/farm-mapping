# Experiment compendium: every experiment, what it tested, what it found

*Companion to `paper/main.tex` §5–6 and `paper/experiments_justification_plan.md`.
Narrative results: `experiments/RESULTS.md`. Raw: `experiments/results/*.json`.
Notebook: `notebooks/experiment_results_analysis.ipynb`.*

This document exists because the project's experimental record accumulated across
two differently-numbered series over five months, and because the most recent
work — the parameter-validation campaign — **retroactively changed how the older
results should be read**. It is organised by what each experiment was actually
testing, not by chronology.

---

## Part 0 — The result that reframes everything: seed variance (E0.3)

**Question.** Every verdict in this project's history is a single-seed point
estimate. How much does a metric move if *nothing* changes but the random seed?

**Design.** Five runs of the identical production configuration
(`world_v10_fourclass_v9`), varying only `training.seed` ∈ {42, 43, 44, 45, 46}.
Same data, same recipe, byte-identical configs otherwise.

**Result.**

| Slice | σ (4-class macro-F1) | Decision band for two single runs (2√2·σ) |
|---|---|---|
| test | 0.0057 | ±0.016 |
| **eval** | **0.0334** | **±0.095** |
| generalization | 0.0075 | ±0.021 |
| qual_eval | 0.0163 | ±0.046 |
| frozen blind benchmark (farm AUC) | **0.00076** | ±0.002 |

**Why this matters.** `eval` was the project's primary comparison metric. Its
decision band is ±0.095 macro-F1 — wider than nearly every intervention ever
ranked on it:

| Historical verdict | Reported Δ (eval) | Δ/σ | Status |
|---|---|---|---|
| Class-balanced sampling helps in-domain | +0.046 | 1.4σ | **within noise** |
| v10 regularizer bundle helps eval | +0.057 | 1.7σ | **within noise** |
| SoftCon beats SSL4EO | +0.015 | 0.4σ | **within noise** |
| Round-3 label expansion | ~0 | ~0 | confirmed null |

Two conclusions survive on other evidence: balanced sampling's *out-of-domain*
loss (−0.054 where σ=0.0075, ≈7σ) and the v8-vs-v9 null (independently confirmed
at p=0.715 on the blind benchmark).

**The mechanism — it is a metric artefact, not model instability.** Decomposing
the seed spread by class:

| Slice | macro-F1 spread | NotFarm | Poultry | Pigs | **Cattle** | Cattle n |
|---|---|---|---|---|---|---|
| eval | 0.081 | 0.029 | 0.022 | 0.075 | **0.200** | 7 |
| qual_eval | 0.046 | 0.003 | 0.009 | 0.040 | **0.150** | 6 |
| test | 0.014 | 0.008 | 0.003 | 0.026 | 0.038 | 25 |

On `qual_eval`, NotFarm (9,676 rows) varies by 0.003 across seeds while Cattle
(6 rows) varies by 0.150. Macro-F1 weights classes equally, so **one near-empty
class supplies ~80% of the variance in the headline number.** `test` is the
control: give Cattle 25 rows and the spread collapses to 0.014.

Excluding Cattle halves eval σ (0.033 → 0.016) and cuts qual_eval σ threefold
(0.016 → 0.0056). Every lever below is therefore reported both ways.

**Cautionary example.** Raw 4-class macro-F1 made the SSL4EO run look 0.143
(−25σ) worse than SoftCon on `test`. Per class the two differ by 0.001 / 0.006 /
0.010 on NotFarm / Poultry / Pigs — and **0.541 on Cattle**, which collapsed to
zero in that single run. The real backbone effect is ~0.005 AUC. Always
decompose a macro-F1 gap by class before believing it.

---

## Part 1 — Building an instrument you can trust

### E0.1 — Frozen blind benchmark
**Question.** Can we build a held-out slice powerful enough to decide Δ≥0.02?

**Design.** 11,365 adjudicated clusters across 131 countries, drawn from the
`qual_eval` pool and filtered to those ≥1.28 km from any training cluster — so
blind in both senses: no model trained on them, and none is a spatial
near-duplicate of a training row.

**Result.** 95% bootstrap CI half-width **0.0030** on farm ROC-AUC, 0.0061 on
macro-F1 — roughly **16× tighter** than the ±0.05 of the 523-row eval slice.
Δ≥0.01 is comfortably detectable.

Re-ranking the label rounds on it (paired bootstrap vs v9):

| Model | AUC | Δ vs v9 | 95% CI | p | Separates |
|---|---|---|---|---|---|
| v6 | 0.9690 | −0.0143 | [−0.0175, −0.0110] | 0.000 | yes |
| v7 | 0.9519 | −0.0314 | [−0.0365, −0.0265] | 0.000 | yes |
| v8 | 0.9829 | −0.0003 | [−0.0019, +0.0012] | 0.715 | **no** |
| v9 | 0.9832 | — | — | — | reference |

**v8 and v9 are statistically indistinguishable even at n=11,365** — the
round-3 label expansion (1,963 promoted rows) bought nothing measurable. This
turns "the per-country cap is not a sensitive knob" from an observation about
small deltas into a properly powered null.

**Caveat this introduced.** The benchmark is powerful enough that trivial effects
reach significance (see E2.2). **Every decision rule now needs a practical
effect floor, not just a p-value.** Its own limitation: dominated by NotFarm
rows with only 6 Cattle examples, so it resolves farm-vs-not well and
minority-type questions not at all.

### E0.2 — Spatial leakage
**Question.** Random splits over spatially autocorrelated data inflate scores.
How exposed are we?

**Result.** Distance from each held-out cluster to the nearest of 12,062
training clusters:

| Split | n | median | <1.28 km | AUC near | AUC far | inflation |
|---|---|---|---|---|---|---|
| val | 2,666 | 2.65 km | 32.5% | 0.993 | 0.964 | **+0.028** |
| test | 2,094 | 1.96 km | 38.0% | 0.997 | 0.992 | +0.006 |
| eval | 523 | 8.00 km | 16.6% | 0.961 | 0.905 | **+0.056** |
| generalization | 426 | 115.59 km | **0.5%** | — | — | — |
| qual_eval | 11,772 | 33.85 km | 3.5% | 0.981 | 0.983 | −0.002 |

**Corrected a paper claim:** the "~27% of eval clusters within one patch width"
figure was wrong — measured 16.6%. `generalization` has essentially zero leakage
(median 116 km), independently validating it as the deployment-relevant slice.

This is a *lower bound* on leakage cost — the model still trained on those
neighbours. The blocked-split artifact (`results/e02_blocked_splits.csv`, 1,751
rows demoted) is ready for a retrain that has not yet been run.

---

## Part 2 — The parameter-validation campaign (20 GPU runs)

**The most recent work, and the reason this document exists.** Every design
choice in the production recipe was, until this campaign, an inherited default.
Twenty runs, each the production configuration with **exactly one** change,
judged on the frozen benchmark against measured seed noise (σ = 0.00076).

Cost: **$23.60**, RTX 4090s in EU-RO-1, ~55 min per run.

### Master table — farm ROC-AUC on the frozen blind benchmark (n=11,365)

| Change | ΔAUC | Δ/σ | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| no freeze phase | +0.0045 | +5.9σ | [+0.0025, +0.0065] | 0.000 | BETTER |
| no photometric augs | +0.0010 | +1.3σ | [-0.0004, +0.0024] | 0.164 | within noise |
| recompute indices | +0.0009 | +1.2σ | [-0.0000, +0.0018] | 0.058 | within noise |
| lr 3e-4 | +0.0007 | +0.9σ | [-0.0010, +0.0026] | 0.466 | within noise |
| cutout only | +0.0004 | +0.5σ | [-0.0009, +0.0017] | 0.542 | within noise |
| checkpoint on val_loss | +0.0000 | +0.0σ | [-0.0009, +0.0010] | 0.905 | within noise |
| 3-class taxonomy | +0.0000 | +0.0σ | [-0.0014, +0.0014] | 0.957 | within noise |
| lr 3e-5 | -0.0004 | -0.5σ | [-0.0024, +0.0015] | 0.712 | within noise |
| 6 bands, no indices | -0.0010 | -1.3σ | [-0.0023, +0.0004] | 0.171 | within noise |
| geometric augs only | -0.0022 | -2.8σ | [-0.0041, -0.0001] | 0.042 | WORSE |
| context 48 px | -0.0026 | -3.4σ | [-0.0050, -0.0001] | 0.036 | WORSE |
| context 128 px | -0.0044 | -5.7σ | [-0.0068, -0.0019] | 0.002 | WORSE |
| SSL4EO backbone | -0.0053 | -6.9σ | [-0.0084, -0.0022] | 0.000 | WORSE |
| RGB+NIR | -0.0102 | -13.4σ | [-0.0127, -0.0079] | 0.000 | WORSE |
| RGB+NDWI | -0.0115 | -15.2σ | [-0.0147, -0.0085] | 0.000 | WORSE |
| RGB only | -0.0170 | -22.3σ | [-0.0203, -0.0138] | 0.000 | WORSE |

**Exactly one of twenty changes improved the model.**

### Per-slice view, Cattle excluded

Because Cattle is noise (Part 0), the per-slice comparison uses mean F1 over
NotFarm/Poultry/Pigs. Seed σ for this metric: test 0.0032, eval 0.0159, gen
0.0099, qual_eval 0.0056. `*` = beyond the 2√2·σ band.

| Run | test | eval | gen | qual_eval |
|---|---|---|---|---|
| e12_crop128 | +0.015* | +0.019 | -0.085* | +0.003 |
| e12_crop48 | -0.004 | -0.024 | +0.014 | -0.035* |
| e11_rgb | -0.031* | -0.092* | -0.016 | -0.068* |
| e11_rgb_nir | -0.005 | -0.007 | -0.002 | -0.016* |
| e11_rgb_ndwi | -0.028* | -0.035 | -0.038* | -0.059* |
| e11_6bands | -0.008 | +0.021 | +0.012 | -0.012 |
| e11_recompute_idx | -0.006 | +0.009 | -0.002 | +0.000 |
| e16_ssl4eo | -0.006 | -0.043 | -0.008 | -0.037* |
| e17_val_loss | -0.009 | +0.014 | -0.004 | -0.003 |
| e19_three_class | -0.003 | +0.039 | +0.000 | -0.009 |
| e15_cutout_only | +0.004 | +0.020 | -0.000 | +0.003 |
| e15_no_photometric | +0.004 | +0.015 | -0.021 | +0.006 |
| e15_geometric_only | -0.007 | -0.002 | -0.036* | -0.013 |
| e14_lr3e-5 | -0.026* | -0.044 | -0.019 | -0.014 |
| e14_lr3e-4 | +0.012* | -0.005 | +0.003 | -0.021* |
| e14_freeze0 | +0.017* | +0.012 | -0.005 | +0.020* |

### E1.4 — Optimiser and schedule *(the one win)*
**Tested:** freeze phase (5 epochs vs 0), learning rate (3e-5 / 1e-4 / 3e-4).

**Result.** Removing the 5-epoch frozen-backbone warmup improves the frozen
benchmark by **+0.0045 AUC (5.9σ, CI [+0.0025, +0.0065])**, test macro-F1 by
+0.017 and qual_eval by +0.020 — and shortens training. Learning rate is already
at a sensible optimum: 3e-5 is clearly worse (test −0.026, −8.3σ) and 3e-4 trades
test (+0.012) against qual_eval (−0.021).

**Action: set `freeze_backbone_epochs: 0`. Keep lr 1e-4.**

### E1.1 — Band composition
**Tested:** RGB / RGB+NIR / RGB+NDWI / 6 bands / 9 channels / recomputed indices.

**Result.** Six raw bands are **indistinguishable** from the full nine-channel
input on every slice (−0.0010, 1.3σ), while dropping to RGB+NIR costs 13.4σ.
So the SWIR bands (B11/B12) carry real signal and **the three derived indices
(NDVI/NDBI/NDWI) carry none** — expected for quantities that are deterministic
functions of channels the network already sees. `recompute_indices` is likewise
a no-op, which retires the known index/jitter inconsistency as a concern.

**Action: drop the indices — a third fewer input channels for no measured loss.**

### E1.2 — Spatial context
**Tested:** centre crop 48 / 64 / 128 px.

**Result.** 128 px is *worse* overall (−0.0044, −5.7σ) and the per-slice split is
the informative part: **+0.015 macro-F1 on `test` but −0.085 on
`generalization` (−8.6σ)**. More context helps the model memorise training
regions and hurts transfer — the opposite of what a global screening deployment
needs. 48 px is worse on qual_eval (−0.035, −6.2σ).

This also **refuted a standing project belief.** The roadmap's "+0.047 macro-F1
for ctx128 on a 3,297-row blind slice" failed to replicate: on the 1,072 blind
rows pairable across both models the difference is +0.002 (95% CI [−0.044,
+0.048], p=0.93). Two independent lines now say larger context is not an upgrade.

**Action: keep the 64 px crop.**

### E1.6 — Backbone
**Tested:** SSL4EO-S12 MoCo vs SoftCon (production).

**Result.** SoftCon confirmed, but by far less than raw macro-F1 implies: SSL4EO
is −0.0053 AUC (−6.9σ) and −0.037 macro-F1 on qual_eval, and indistinguishable
on test / eval / generalization once Cattle is excluded. The production choice
had previously rested only on external GEO-Bench evidence; it now has our own,
modest, support.

### E1.5 — Augmentation
**Tested:** cutout alone; drop the 4 photometric terms; geometric-only
(flips + rot90).

**Result.** Reducing to flips+rot90 costs the frozen benchmark (−0.0022, −2.8σ)
and generalization (−0.036, −3.6σ). Dropping only the photometric terms is a tie
on the frozen benchmark and −0.021 on generalization (2.1σ — inside the band).
**The augmentations buy transfer, not in-domain accuracy.** Cutout in isolation
is neutral everywhere, which retrospectively clears it of responsibility for the
failure of the v10 regularisation bundle.

**Attribution caveat:** these test *groups*, not individual terms, and **no
augmentation hyperparameter (probability, magnitude) has been ablated.** A
per-term leave-one-out remains outstanding.

### E1.7 / E1.9 — Two non-issues
Checkpoint-selection metric (`val_f1` vs `val_loss`) and class taxonomy (4-class
vs 3-class) are both **exactly 0.0000** on the frozen benchmark. The taxonomy
result is consequential: **dropping to three classes costs nothing measurable**,
so Cattle can be merged away — removing the dominant source of metric noise.

---

## Part 3 — Deployment decisions (no training required)

### E2.1 — Decision thresholds *(shipped)*
**Global retune is a no-op:** val-optimal threshold is 0.481 vs the 0.5 default,
worth ≤2.5 points of OOD recall. But miscalibration is **domain-conditional**:

| Threshold | Recall | FPR | Precision | Δrecall | ΔFPR | Pre-registered rule |
|---|---|---|---|---|---|---|
| 0.2 | 0.894 | 0.211 | 0.864 | +0.129 | +0.094 | fail |
| 0.3 | 0.855 | 0.170 | 0.883 | +0.090 | +0.053 | fail (marginal) |
| **0.4** | **0.820** | **0.135** | **0.901** | **+0.055** | **+0.018** | **PASS** |
| 0.5 | 0.765 | 0.117 | 0.907 | — | — | default |

Rule: *adopt the lowest threshold buying ≥5 points of recall for ≤5 points of
FPR*. This selects **0.4**, narrowing the paper's earlier "0.3–0.4".

**Per-country thresholds fail.** Fitted leave-one-country-out, they buy +0.114
recall for **+0.170 FPR** — far worse than the fixed constant. With only four
OOD countries each fit rests on three others and is unstable (one collapses to
0.05 and triples its FPR). Not viable until many more labelled OOD countries exist.

### E2.2 — Ensembling *(rejected)*
v8+v9 mean-probability ensemble: **+0.0018 AUC on the blind benchmark, p<0.001
— statistically real** — and +0.0024 on generalization (p=0.515, a tie).

This is precisely the failure mode a powerful benchmark introduces. Against an
explicit practical floor (0.005 AUC), a gain roughly three times too small does
not justify doubling inference cost across 157k candidates. **Do not ship.**

### E2.3 — Geometry fusion *(rejected in-domain; OOD untested)*
On blind-benchmark rows carrying an Isolation Forest score (3,136 rows, 27.6%
coverage): IF alone AUC **0.5384** — near chance — against CNN 0.9986.
Leave-one-country-out blending selected weight **0.00 for every country**.

The actual hypothesis — that morphometry transfers where imagery does not — is
**untestable here**: only 44 of 426 OOD rows (10.3%) carry an IF score. Note the
IF-covered subset is unusually easy, so it is a biased place to measure marginal
value.

### E2.5 — Class-balanced sampling *(hypothesis refuted)*
**Hypothesis:** the replicated eval-gain / gen-loss trade is a spatial-leakage
artefact. **It is not — the opposite holds.**

| Slice | subset | macro-F1 v9 | v9_bal | Δ |
|---|---|---|---|---|
| eval | all | 0.6023 | 0.6484 | +0.046 |
| eval | near (<1.28 km) | 0.6222 | 0.5299 | **−0.092** |
| eval | far (leakage-free) | 0.5721 | 0.6566 | **+0.084** |
| generalization | far | 0.3959 | 0.3420 | −0.054 (ΔAUC −0.019, p=0.038) |

The gain is *larger* on leakage-free rows and the OOD loss is undiminished. The
trade is a genuine property of the intervention, so "keep balancing off" stands
— now for the right reason.

### E2.6 — Label-source shift *(confirmed and quantified)*
Holding the split fixed and varying only the labelling instrument:

| Split | Registry AUC (n) | Visual AUC (n) | ΔAUC |
|---|---|---|---|
| train | 0.9994 (9,650) | 0.9976 (1,001) | −0.0018 |
| test | 0.9933 (1,756) | 0.9500 (74) | −0.0433 |
| eval | 1.0000 (73) | 0.9091 (449) | **−0.0909** |
| qual_eval | 0.9792 (5,856) | 0.9605 (356) | −0.0187 |

Train is 80% registry; eval is 86% visual. **The headline eval AUC of 0.911 is
essentially the visual-label figure**, while test (84% registry) is essentially
the registry figure. The gap is near-zero within train (−0.002) but large on
held-out slices — the model fits both sources equally well and generalises worse
only on visual ones. That points to visually-adjudicated rows being intrinsically
harder cases, not to registry labels being noisy: **acquire more visual labels
for training rather than down-weighting registry ones.**

---

## Part 4 — Historical series, reread

Two differently-numbered series both run v3–v10 and must not be conflated:
**Series A** (`world_v2..v9`, 3-class, varying model) and **Series B**
(`world_v10_fourclass_v6..v10`, 4-class, architecture fixed, varying labels).

### The one robust result in the whole record
Sentinel-2–native pretraining vs ImageNet: **+0.10 macro-F1, ≈2× the CI**. The
`v9_imagenet9ch` control settles the confound — holding nine channels fixed and
reverting only the initialisation gives eval 0.393, statistically identical to
4-channel ImageNet (0.403) and far below every S2-native run. The extra bands are
near-worthless to an ImageNet network and valuable to an S2-pretrained one.

After de-duplication and CIs, the six leading S2-native configurations sit at
0.466–0.470 — **a statistical tie**.

### Label acquisition (Series B)
**Round 2 — negatives alone shift the prior without improving the model.** 962
NotFarm corrections dropped the hardest-OOD false-positive rate from 66.2% to
5.0%, but recall collapsed 92.5% → 42.5%. Three diagnostics prove no ranking
gain: AUC flat (0.777→0.772); recall *identical* at matched FPR (95.0% both); and
mean predicted farm probability fell for true negatives (0.091→0.021) **and**
true positives (0.846→0.568) — a uniform shift, not sharper separation. The model
learned "unfamiliar country ⇒ not a farm".

**Lesson:** any label campaign adding only one class must be evaluated with
threshold-free and matched-operating-point metrics.

**Round 3 — balanced promotion is what helped.** 1,740 farm positives improved
ranking on every slice, and against round 1 the qual_eval FP rate fell 4.9% →
3.6% *with* recall rising 86.8% → 88.9% — a genuine Pareto improvement.

**The cap is not a sensitive knob** (50 → 70 per country): confirmed as a
powered null on the blind benchmark (p=0.715, E0.1).

### Interventions that failed
- **cRT** (classifier retraining): +0.019 gen on SSL4EO, −0.020 on SoftCon. A
  sign flip on n=273 is noise; the effect is unmeasured, not backbone-dependent.
- **Logit adjustment:** train-time cost 0.15 test macro-F1; post-hoc τ sweep put
  the optimum at τ=0 — the ceiling is the representation, not the decision rule.
- **AdaBN:** gen macro-F1 0.407 → 0.383. A useful negative — the OOD gap is
  morphological and label-distributional, not radiometric.
- **v10 regularisation bundle** (label smoothing + cutout + longer schedule):
  rejected. Label smoothing *worsened* calibration (qual_eval ECE 0.027 → 0.052),
  the opposite of its usual justification; cutout did not rescue Cattle. Run as a
  bundle, so unattributable — a design mistake, now corrected by E1.5.

### The data-integrity incident
Patches were keyed by cluster identifier with no spatial check; an upstream merge
renumbered 93.5% of identifiers, so **34.3–34.6% of labelled rows were paired
with another candidate's imagery** for months, across *all* splits. Correcting it
moved v9_softcon from 0.712/0.462/0.400 to **0.890/0.631/0.486** (test/eval/gen)
— larger than any modelling intervention in this record — and **reversed two
verdicts**. Invariants now enforced: 250 m haversine validation with a hard abort
above 5% stale, identifier-stability asserts at merge, and the rule that upstream
cluster ids are ephemeral (location, not identity, is the durable key).

---

## Part 5 — Where this leaves the production recipe

**Change these**

| Change | Evidence |
|---|---|
| `freeze_backbone_epochs: 0` | +0.0045 AUC, +5.9σ — the only win in 20 runs |
| Drop NDVI/NDBI/NDWI (9→6 channels) | indistinguishable from 9ch; RGB+NIR is −13σ |
| Merge or drop Cattle | 3-class is exactly 0.0000; removes ~80% of metric noise |
| OOD screening threshold 0.4 | +0.055 recall for +0.018 FPR (pre-registered rule) |
| Stop reporting 4-class macro-F1 on `eval` | σ=0.033; use the frozen benchmark (σ=0.0008) |

**Confirmed already correct:** 64 px crop, lr 1e-4, SoftCon backbone, `val_f1`
checkpointing, the full augmentation stack, balancing off, no ensemble, no
geometry fusion.

**Important caveat:** every lever was measured *individually*. Composition is not
assumed — SoftCon × ctx128 famously failed to stack — and the combination runs
(freeze0 + 6 bands + 3-class) were launched but lost to an infrastructure failure
before producing results.

## Part 6 — What is still unmeasured

1. **Candidate-stage recall (E2.7)** — the binding constraint. A facility the
   morphometric filter never proposes is invisible to every number above. The
   shortlist review found two complexes 930–1,300 m from any candidate in a
   single Vietnamese hotspot box (`docs/SHORTLIST_REVIEW_IDN_PHL_VNM.md`).
2. **Augmentation hyperparameters** — ~22 probability and magnitude values,
   none ablated; only three coarse group ablations exist.
3. **Imagery parameters (E1.3)** — cloud threshold 15%, 2023-only window, median
   compositing, 128 px @ 10 m extraction. Never tested; needs re-extraction.
4. **Weight decay, batch size, scheduler** — launched in the second wave, lost
   with it.
5. **The spatially blocked retrain (E0.2)** — split artifact ready, run not done.
6. **Label-noise ceiling (E0.4)** — no double-annotation, so we cannot say how
   much residual error is annotator disagreement.
