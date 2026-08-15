# Experiment results — run 2026-08-13

Execution of `paper/experiments_justification_plan.md`.

**Compute reality at run time:** no GPU was available. The RunPod pod refused
connections (`213.173.105.101:41075`), and the local machine has no CUDA device
(12 CPU cores, 31 GB RAM). The full patch store lives on the pod volume; only
1.5 GB is local. **No training run was therefore possible.**

What *was* possible turned out to be a large fraction of the scientific content:
the archived scored parquets carry full per-class probabilities, true labels,
explicit split assignments, coordinates, label sources, and the Isolation Forest
score for 154,908 candidates. Every experiment whose question can be answered by
re-analysing fixed model outputs was run to completion.

| Experiment | Status | Verdict |
|---|---|---|
| E0.1 blind benchmark | **Complete** (4-class); partial (3-class) | Frozen, n=11,365, CI half-width **0.003** |
| E0.2 spatial leakage | **Complete** (diagnostic); retrain blocked | Leakage real: eval AUC inflated **+0.056** |
| E2.1 thresholds | **Complete** | **SHIP threshold 0.4 OOD**; per-country fitting fails |
| E2.2 ensembling | **Complete** | **DO NOT SHIP** — real but +0.0018 AUC |
| E2.3 geometry fusion | Partial — OOD untestable | Zero in-domain value; blend weight 0.00 |
| E2.5 balancing | **Complete** (diagnostic) | **Hypothesis REFUTED** — trade is not leakage |
| E2.6 label source | **Complete** (diagnostic) | Shift confirmed, quantified per split |
| E0.3, E1.1–E1.9, training halves | **Blocked** | Needs GPU (~46 runs, ~200 GPU-h) |
| E0.4, E2.4, E2.7 | **Blocked** | Needs human annotation (~110 h) |

Artifacts: `experiments/results/*.json`, `e01_blind_benchmark_frozen.csv`,
`e02_blocked_splits.csv`. Harness: `experiments/lib.py`.

---

## E0.1 — Blind benchmark frozen, and it is powerful

A benchmark of **11,365 rows across 131 countries** was frozen from `qual_eval`,
further filtered to clusters ≥1.28 km from any training cluster, so it is blind
in both senses: no model trained on these rows, and none is a spatial
near-duplicate of a training row.

**Achieved detection power far exceeds the plan's target.** The plan asked for
Δ≥0.02 to become detectable. Measured 95% bootstrap CI half-widths:

| Metric | Value | 95% CI | Half-width |
|---|---|---|---|
| farm ROC-AUC | 0.9832 | [0.9802, 0.9862] | **0.0030** |
| binary macro-F1 | 0.9287 | [0.9228, 0.9349] | **0.0061** |

That is roughly **16× tighter than the ±0.05 macro-F1 of the 523-row eval set**.
Δ≥0.01 is now comfortably detectable; the measurement bottleneck described in the
paper is, for this class of comparison, solved.

**Re-ranking the label rounds** (paired bootstrap against v9, shared resamples):

| Model | AUC | Country-macro AUC | Δ vs v9 | 95% CI | p | Separates? |
|---|---|---|---|---|---|---|
| v6 | 0.9690 | 0.9713 | −0.0143 | [−0.0175, −0.0110] | 0.000 | **yes** |
| v7 | 0.9519 | 0.9539 | −0.0314 | [−0.0365, −0.0265] | 0.000 | **yes** |
| v8 | 0.9829 | 0.9831 | −0.0003 | [−0.0019, +0.0012] | 0.715 | no |
| v9 | 0.9832 | 0.9830 | — | — | — | reference |

Two findings. The v7 negatives-only regression is confirmed and is the largest
effect in the series (−0.031, decisively separating). And **v8 and v9 are
statistically indistinguishable even at n=11,365** (p=0.715, CI excludes anything
larger than ±0.002). The round-3 label expansion — 1,963 additional promoted rows
— bought nothing measurable. This upgrades the paper's "the per-country cap is
not a sensitive knob" from an observation about small deltas into a properly
powered null result.

### E0.1b — the ctx128 advantage does not replicate

The roadmap's headline "ctx128 beats softcon by +0.047 macro-F1 on 3,297 blind
rows" could not be reproduced: the 133k-row 3-class release it refers to is not
in the local cache (all local 3-class releases are the 15,770-row labeled-only
exports). The closest available substitute is the **1,072 labeled `unassigned`
rows**, perfectly paired across both models.

| Measure | softcon | ctx128 | Δ | 95% CI | p |
|---|---|---|---|---|---|
| pooled macro-F1 | 0.6179 | 0.6200 | **+0.0021** | [−0.0442, +0.0484] | 0.932 |
| country-macro macro-F1 | 0.4059 | 0.4229 | +0.0170 | — | — |

The direction favours ctx128 in both pooled and country-macro terms, consistent
with the roadmap, but the magnitude is **an order of magnitude smaller than
+0.047** and nowhere near separating. This slice is itself skewed (70% Australia),
so it does not settle the question — but it does mean the +0.047 figure should not
be quoted as evidence, and E1.2 (context ablation) remains genuinely open rather
than nearly-decided.

---

## E0.2 — Spatial leakage is real, and the paper's exposure figure was wrong

Distance from every held-out cluster to the nearest of 12,062 training clusters:

| Split | n | median dist | <500 m | **<1.28 km** | <2.56 km | <5 km |
|---|---|---|---|---|---|---|
| val | 2,666 | 2.65 km | 0.149 | **0.325** | 0.491 | 0.632 |
| test | 2,094 | 1.96 km | 0.173 | **0.380** | 0.573 | 0.730 |
| eval | 523 | 8.00 km | 0.078 | **0.166** | 0.270 | 0.398 |
| generalization | 426 | 115.59 km | 0.000 | **0.005** | 0.009 | 0.021 |
| qual_eval | 11,772 | 33.85 km | 0.014 | **0.035** | 0.063 | 0.110 |

**Correction to the paper:** the claimed "~27% of eval clusters within one patch
width" is wrong for `eval` — the measured figure is **16.6%**. The 27% figure
appears to describe a different slice; `test` (38.0%) and `val` (32.5%) are both
considerably more exposed than eval. The paper has been corrected.

**Inflation attributable to proximity** (production model, near vs far):

| Split | n near | AUC near | n far | AUC far | ΔAUC | Δmacro-F1 |
|---|---|---|---|---|---|---|
| val | 866 | 0.993 | 1,800 | 0.964 | **+0.028** | +0.025 |
| test | 796 | 0.997 | 1,298 | 0.992 | +0.006 | +0.023 |
| eval | 87 | 0.961 | 436 | 0.905 | **+0.056** | +0.050 |
| qual_eval | 407 | 0.981 | 11,365 | 0.983 | −0.002 | −0.028 |

Leakage inflation is substantial on `val` and `eval` and negligible on
`qual_eval`. **`generalization` has essentially zero leakage (0.5% within
1.28 km, median distance 116 km)**, which independently validates it as the
deployment-relevant slice.

This is a *lower bound* on true leakage cost: the model still trained on those
nearby clusters, so this measures evaluation contamination only, not the full
effect a blocked retrain would show. A blocked-split artifact
(`results/e02_blocked_splits.csv`, 1,751 rows demoted) is ready for that retrain.

---

## E2.1 — Ship a 0.4 out-of-domain threshold; do not fit per-country

**Global retune is confirmed a no-op.** The val-optimal threshold is 0.48 (vs the
0.50 default) and applying it changes recall by at most +0.012 on any slice.

**A fixed lower threshold out of domain passes the pre-registered rule.** Scanning
the sweep against the rule (≥+5 pts recall for ≤+5 pts FPR versus the default):

| Threshold | Recall | FPR | Precision | Δrecall | ΔFPR | Rule |
|---|---|---|---|---|---|---|
| 0.2 | 0.894 | 0.211 | 0.864 | +0.129 | +0.094 | fail |
| 0.3 | 0.855 | 0.170 | 0.883 | +0.090 | +0.053 | fail (marginal) |
| **0.4** | **0.820** | **0.135** | **0.901** | **+0.055** | **+0.018** | **PASS** |
| 0.5 | 0.765 | 0.117 | 0.907 | — | — | default |

**→ Ship 0.4 for out-of-domain screening.** Note this narrows the paper's earlier
"0.3–0.4" recommendation: 0.3 misses the rule (5.3 points of FPR against a 5-point
budget), 0.4 passes comfortably.

**Per-country threshold fitting fails and should not be shipped.** Fitted
leave-one-country-out (so no country's threshold sees its own rows):

| Country | n | Fitted thr | Recall 0.5 → tuned | FPR 0.5 → tuned |
|---|---|---|---|---|
| ALB | 135 | 0.05 | 0.765 → 1.000 | 0.093 → 0.314 |
| BGD | 162 | 0.24 | 0.769 → 0.878 | 0.067 → 0.067 |
| COD | 44 | 0.22 | 0.783 → 0.870 | 0.381 → 0.429 |
| NGA | 85 | 0.22 | 0.750 → 0.853 | 0.000 → 0.118 |
| **Pooled** | 426 | — | **0.765 → 0.878** | **0.117 → 0.287** |

Pooled, this buys +11.4 points of recall for **+17.0 points of FPR** — far worse
than the fixed 0.4 threshold. With only four out-of-domain countries each fit is
based on three others and is wildly unstable (ALB draws 0.05, tripling its FPR).
Per-country calibration needs many more labelled OOD countries before it is viable.

---

## E2.2 — Ensemble rejected: statistically real, operationally negligible

| Slice | v9 alone | v8+v9 mean | Δ | 95% CI | p | Separates |
|---|---|---|---|---|---|---|
| blind benchmark (n=11,365) | 0.9832 | 0.9851 | +0.0018 | [+0.0010, +0.0028] | 0.000 | **yes** |
| generalization (n=426) | 0.9153 | 0.9177 | +0.0024 | [−0.0048, +0.0103] | 0.515 | no |

The frozen benchmark is powerful enough that a **+0.0018 AUC** gain reaches
p<0.001. This is exactly the failure mode a large benchmark introduces, so the
decision rule was given an explicit practical floor (0.005 AUC) alongside the
significance test. The gain is real and roughly three times too small to justify
doubling inference cost across 157k candidates. **Do not ship.**

---

## E2.3 — Geometry fusion adds nothing in domain; the OOD claim remains untested

On blind-benchmark rows carrying an Isolation Forest score (3,136 rows, 27.6%
coverage): **IF alone AUC 0.5384** — near chance — against **CNN alone 0.9986**.
Leave-one-country-out blending selected weight **0.00 for every country**: the
optimiser declines to use the geometric score at all.

The actual hypothesis — that morphometry transfers where imagery does not — is
**untestable here**: only 44 of 426 out-of-domain rows (10.3%) carry an IF score.
Two caveats worth carrying forward: the IF-covered subset is unusually easy (CNN
AUC 0.9986 there vs 0.9832 overall), so it is a biased place to measure marginal
value; and this IF AUC is not comparable to the 0.862 reported in
`docs/selection_pipeline_analysis.md`, which used a different pool and base rate.
Verdict: no in-domain value; extend IF-score coverage to OOD rows before revisiting.

---

## E2.5 — Hypothesis refuted: the balancing trade is not a leakage artefact

The plan hypothesised that class-balanced sampling's eval gain would shrink once
leakage-exposed rows were removed. **The opposite is true.**

| Slice | Subset | n | macro-F1 v9 | macro-F1 v9_bal | Δ |
|---|---|---|---|---|---|
| eval | all | 523 | 0.6023 | 0.6484 | +0.0461 |
| eval | near (<1.28 km) | 87 | 0.6222 | 0.5299 | **−0.0923** |
| eval | far (≥1.28 km) | 436 | 0.5721 | 0.6566 | **+0.0844** |
| generalization | all | 426 | 0.3940 | 0.3404 | −0.0537 |
| generalization | far | 424 | 0.3959 | 0.3420 | −0.0539 (ΔAUC −0.019, p=0.038) |

The eval gain is **larger** on leakage-free rows (+0.084) than overall (+0.046),
and is actually *negative* on the leaked rows. The generalization loss is
undiminished by filtering and is statistically significant (p=0.038).

So the in-domain/out-of-domain trade is a genuine property of the intervention,
not an artefact of spatial contamination. The production decision (keep balancing
off) is unchanged, and its stated justification in the paper is now
experimentally supported rather than merely plausible.

---

## E2.6 — Label-source shift confirmed and quantified

Within every split — domain held constant, only the labelling instrument varying
— visually-labelled rows score worse than registry-labelled rows:

| Split | Registry AUC (n) | Visual AUC (n) | ΔAUC | Δmacro-F1 |
|---|---|---|---|---|
| train | 0.9994 (9,650) | 0.9976 (1,001) | −0.0018 | −0.093 |
| test | 0.9933 (1,756) | 0.9500 (74) | −0.0433 | −0.255 |
| eval | 1.0000 (73) | 0.9091 (449) | **−0.0909** | −0.110 |
| qual_eval | 0.9792 (5,856) | 0.9605 (356) | −0.0187 | −0.323 |

The composition flip is stark: **train is 80% registry / 8% visual, eval is 86%
visual / 14% registry.** Decomposing eval's headline AUC of 0.911 by source gives
**1.000 on its 73 registry rows and 0.909 on its 449 visual rows** — the reported
eval figure is essentially the visual-label figure, while test is essentially the
registry figure. A large part of the test→eval "generalisation gap" is therefore a
change of measurement instrument, exactly as the paper argues.

The direction is consistent across all four splits, and the gap is near-zero on
train (−0.002 AUC) but large on held-out slices. Per the plan's decision rule this
is the **asymmetric** case; but the asymmetry favours the reading that visual rows
are intrinsically harder cases (adjudicated hard negatives and ambiguous sites)
rather than that registry labels are noisy — the model fits both sources equally
well in training and generalises worse only on visual ones. The recommended action
is therefore more visual labels in training, not down-weighting registry labels.

---

## Blocked experiments and what unblocks them

### Needs a GPU (~46 training runs, ~200 GPU-hours)
E0.3 (seed variance, 5 runs), E1.1 (bands, 6), E1.2 (context, 3), E1.3 (cloud, 2
+ re-extraction), E1.4 (optimiser, 8), E1.5 (augmentation, 11), E1.6
(architecture, 4), E1.7 (checkpoint metric, 2), E1.9 (taxonomy, 2), plus the
retrain halves of E0.2, E2.5, E2.6.

Two blockers, both outside what this session could resolve without spending money
or user credentials:
1. **The pod is down** — SSH connection refused. It needs restarting from the
   RunPod console, or a new GPU pod rented.
2. **Even when up, the pod was CPU-only with a broken CUDA torch install**
   (`/workspace/farm-venv` raises `No module named 'torch._C'`). A GPU pod plus a
   working `torch`+`torchgeo` environment is required.

E1.8 (TTA) is the cheapest of these — inference-only on existing checkpoints, no
retraining — but the checkpoints and the full patch store are both on the
unreachable volume.

### Needs human annotation (~110 hours)
E0.4 (double-annotate ~300 eval rows), E2.4 (~500 Cattle labels), E2.7
(candidate-stage recall in 3–5 countries). No model or compute substitute exists;
these are measurements of the labels themselves.

### Recommended order once a GPU is available
1. **E0.3 seed variance (20 GPU-h).** Still the highest value per hour: several
   results above have deltas in the 0.002–0.05 range and no σ to compare against.
2. **E1.2 context (12 GPU-h).** E0.1b removed the main evidence for 128 px, so
   this is now genuinely open rather than nearly-settled.
3. **E0.2 blocked retrain.** The split artifact is ready; this converts the
   leakage lower bound into a real estimate.
4. Then E1.1, E1.5, E1.4, E1.6, E1.7 on the frozen benchmark, which now has the
   power to resolve them.

---

## Changes these results imply for `paper/main.tex`

1. **§6.3** — the 27% eval-proximity figure is wrong; measured 16.6% (eval), with
   test 38.0% and val 32.5%. Replace with the measured table. *(applied)*
2. **§5.5 / §7** — recommend a 0.4 OOD threshold specifically, and record that
   per-country threshold fitting fails at four OOD countries. *(applied)*
3. **§5.1 / §6.3** — the +0.047 blind-slice ctx128 result does not replicate
   locally (+0.002 pooled, p=0.93); stop citing it as near-decisive. *(applied)*
4. **§5.2** — the balancing trade survives leakage filtering, so state it as
   experimentally confirmed rather than mechanistically plausible. *(applied)*
5. **§5.3** — v8 vs v9 is a properly powered null (p=0.715 at n=11,365), not just
   a small delta. *(applied)*
6. **§6.2** — add the per-source AUC decomposition of the eval figure. *(applied)*

---
---

# Part 2 — GPU experiments (run 2026-08-13/14)

The Tier-1 experiments blocked in Part 1 were run after credits were added to
RunPod. **20 single-lever configs**, each the production `world_v10_fourclass_v9`
recipe with exactly one change, on RTX 4090s in EU-RO-1 (the only datacentre the
network volume attaches to — it has no A40/A5000/A6000, so the configs' entire
GPU fallback list was unavailable except L4).

**Cost: $23.60** — ~$0.80/run at $0.74/hr secure-cloud, ~55 min each.
Configs: `configs/experiments/*.yaml` (generated by `experiments/gen_configs.py`).
Analysis: `experiments/analyze_gpu_runs.py` → `experiments/results/gpu_runs_analysis.json`.

## E0.3 — Seed variance: the result that reframes the project

Five runs of the identical config differing only in seed:

| Seed | frozen-benchmark AUC | test | eval | gen | qual_eval |
|---|---|---|---|---|---|
| 42 (production) | 0.9832 | 0.8030 | 0.6023 | 0.3940 | 0.6330 |
| 43 | 0.9836 | 0.7992 | 0.6830 | 0.4126 | 0.6785 |
| 44 | 0.9832 | 0.7999 | 0.6266 | 0.3966 | 0.6519 |
| 45 | 0.9816 | 0.8130 | 0.6746 | 0.3961 | 0.6517 |
| 46 | 0.9832 | 0.8005 | 0.6472 | 0.4008 | 0.6507 |
| **σ** | **0.00076** | **0.0057** | **0.0334** | **0.0075** | **0.0163** |

**eval macro-F1 has σ = 0.033, i.e. a decision band of ±0.095.** Every headline
verdict in the project's history was read off this metric:

| Historical verdict | Reported Δ (eval) | Δ/σ | Status |
|---|---|---|---|
| Class-balanced sampling helps in-domain | +0.046 | 1.4σ | **within noise** |
| v10 regularizer bundle helps eval | +0.057 | 1.7σ | **within noise** |
| SoftCon beats SSL4EO | +0.015 | 0.4σ | **within noise** |
| Round-3 label expansion | ~0 | ~0 | confirmed null |

None of these were measurable on the instrument used to measure them. Two
survive for other reasons: the balanced-sampling *out-of-domain* loss (−0.054 on
gen, where σ=0.0075, so ~7σ) is real and independently confirmed by the Part-1
leakage test; and the v8-vs-v9 null was independently confirmed on the frozen
benchmark (p=0.715).

### Why eval macro-F1 is so unstable: it is mostly Cattle

Per-class F1 spread across the five seeds:

| Slice | macro-F1 | NotFarm | Poultry | Pigs | **Cattle** | Cattle n |
|---|---|---|---|---|---|---|
| eval | 0.081 | 0.029 | 0.022 | 0.075 | **0.200** | 7 |
| qual_eval | 0.046 | 0.003 | 0.009 | 0.040 | **0.150** | 6 |
| test | 0.014 | 0.008 | 0.003 | 0.026 | 0.038 | 25 |

Macro-F1 weights all four classes equally, so a class with **six held-out rows**
contributes ~80% of the variance while NotFarm, with 9,676 rows, contributes
0.003. The `test` row is the control: give Cattle 25 rows instead of 6 and the
spread collapses.

Excluding Cattle tightens every slice — eval σ 0.033→0.016, qual_eval σ
0.016→0.0056 — so all lever verdicts below are reported both ways.

**This is a measurement bug, not a model property.** Reporting 4-class macro-F1
on slices where Cattle has single-digit support is reporting a coin flip.

## Lever verdicts — frozen blind benchmark (farm-vs-not AUC, n=11,365)

Judged against seed σ=0.00076, paired bootstrap vs the production baseline:

| Lever | ΔAUC | Δ/σ | Verdict |
|---|---|---|---|
| **no backbone freeze phase** | **+0.0045** | **+5.9σ** | **BETTER** |
| drop photometric augs | +0.0010 | +1.3σ | within noise |
| recompute indices | +0.0009 | +1.2σ | within noise |
| lr 3e-4 | +0.0007 | +0.9σ | within noise |
| cutout only | +0.0004 | +0.5σ | within noise |
| checkpoint on val_loss | +0.0000 | +0.0σ | within noise |
| **3-class taxonomy** | +0.0000 | +0.0σ | within noise |
| lr 3e-5 | −0.0004 | −0.5σ | within noise |
| **6 bands, no indices** | −0.0010 | −1.3σ | **within noise** |
| geometric augs only | −0.0022 | −2.8σ | worse |
| context 48 px | −0.0026 | −3.4σ | worse |
| context 128 px | −0.0044 | −5.7σ | worse |
| SSL4EO backbone | −0.0053 | −6.9σ | worse |
| RGB+NIR | −0.0102 | −13.4σ | worse |
| RGB+NDWI | −0.0115 | −15.2σ | worse |
| RGB only | −0.0170 | −22.3σ | worse |

**Exactly one of twenty changes improved the model.**

## Per experiment

### E1.4 — Optimiser: drop the freeze phase (the one win)

Removing the 5-epoch frozen-backbone warmup improves the frozen benchmark by
+0.0045 AUC (+5.9σ, CI [+0.0025, +0.0065]), test macro-F1 by +0.017 (+5.3σ) and
qual_eval by +0.020 (+3.6σ), with eval and generalization unchanged. It also
shortens training. **Recommend adopting `freeze_backbone_epochs: 0`.**

Learning rate is at a sensible optimum: 3e-5 is clearly worse (test −0.026,
−8.3σ), and 3e-4 is mixed (test +0.012 but qual_eval −0.021). **Keep 1e-4.**

### E1.1 — Bands: the three indices are dead weight, the SWIR bands are not

| Config | frozen AUC Δ | verdict |
|---|---|---|
| 6 bands (no NDVI/NDBI/NDWI) | −0.0010 (−1.3σ) | **indistinguishable from 9 channels** |
| RGB+NIR | −0.0102 (−13.4σ) | clearly worse |
| RGB+NDWI | −0.0115 (−15.2σ) | clearly worse |
| RGB | −0.0170 (−22.3σ) | clearly worse |

The six raw bands match the full nine channels on every slice, but dropping to
RGB+NIR costs 13σ — so **B11/B12 (SWIR) carry real signal and the three derived
indices carry none.** `recompute_indices` is likewise a no-op, consistent with
the indices simply not mattering.

**Recommend dropping the indices**: a third fewer input channels for no measured
loss, and it retires the known index/jitter inconsistency as a concern.

### E1.2 — Context: 64 px confirmed, and the ctx128 belief is refuted

128 px is worse on the frozen benchmark (−0.0044, −5.7σ) and much worse on
generalization (−0.085 macro-F1, −8.6σ), though better on test (+0.015). 48 px is
worse on qual_eval (−0.035, −6.2σ). **Keep 64 px.**

This closes a standing project belief from three directions: the roadmap's
"+0.047 blind-slice ctx128 win" failed to replicate in Part 1 (+0.002, p=0.93),
and now larger context measurably *hurts* out-of-domain performance. The
test-vs-generalization split is the tell — more context helps memorise training
regions and hurts transfer.

### E1.6 — Backbone: SoftCon justified, but by far less than raw macro-F1 implies

SSL4EO is worse on the frozen benchmark (−0.0053, −6.9σ) and on qual_eval
(−0.037, −6.7σ), and indistinguishable on test, eval and generalization once
Cattle is excluded.

**This one nearly produced a false headline.** Raw 4-class macro-F1 showed SSL4EO
at −0.143 on test — a −25σ "collapse". Per class: NotFarm −0.001, Poultry −0.006,
Pigs −0.010, **Cattle −0.541** (0.541 → 0.000). The entire gap was one run failing
to learn a 25-example class that flips between seeds anyway. The real backbone
effect is ~0.005 AUC, consistent with the historical near-tie.

### E1.5 — Augmentation: the photometric terms matter for transfer only

Stripping to flips+rot90 costs generalization (−0.036, −3.6σ) and the frozen
benchmark (−0.0022, −2.8σ). Dropping only the four photometric terms is within
noise on the frozen benchmark but −0.021 on generalization. Cutout as a single
lever is neutral everywhere.

**Keep the augmentation stack.** And this cleanly isolates the rejected v10
bundle: cutout was not the culprit — it does nothing either way.

### E1.7 / E1.9 — Two non-issues

`checkpoint_metric` (val_loss vs val_f1) is +0.0000 on the frozen benchmark and
within noise on every slice. The 3-class taxonomy is likewise +0.0000 on the
frozen benchmark; its eval +0.039 is 2.5σ, inside the ±2√2σ band.

The taxonomy null is now measured rather than asserted, and it matters for E2.4:
**if Cattle is dropped or merged, nothing measurable is lost** — and the dominant
source of metric noise disappears with it.

## Recommended production changes

1. **`freeze_backbone_epochs: 0`** — the only measured improvement (+5.9σ).
2. **Drop NDVI/NDBI/NDWI**, keep the six bands — no loss, a third fewer channels.
3. **Merge or drop Cattle** — unlearnable at 153 train rows, contributes ~80% of
   metric noise, and the 3-class taxonomy costs nothing.
4. **Stop reporting 4-class macro-F1 on eval.** Use the frozen benchmark
   (σ=0.00076) as the primary metric and generalization for transfer.
5. Keep: 64 px crop, lr 1e-4, SoftCon, the full augmentation stack, val_f1
   checkpointing.

## Infrastructure defects found and fixed

Four, all pushed. Three would have corrupted or silently degraded a parallel fleet:

- **Shared startup-log path.** Every pod teed into one file on the network volume;
  `tee` truncates on open, so 20 concurrent pods would have left one usable log.
- **Block-buffered step subprocesses.** Steps are spawned with stdout to a file
  and no `-u`, so a 50-epoch run showed no progress for its entire duration and
  looked hung.
- **Unhandled transient RunPod API errors.** A GraphQL response with no `data`
  key raised `KeyError`, killing the launcher mid-fleet and stranding a created
  pod that billed with no work on it — the startup script is uploaded *after* the
  SSH wait, which is exactly where the crash landed. Now retried with backoff.
- **`farm-venv` health check.** Presence of the directory was treated as proof it
  worked; a venv built on a CPU-only pod leaves an import-broken torch.

One experiment-design bug of my own: `label_mode` is baked into the candidate CSVs
by the `candidates` step, so changing it in config while running `--steps train`
gives a 3-class head four-class labels and a CUDA device-side assert
(`t >= 0 && t < n_classes`). E1.9 needed `--steps candidates train inference` and
its own `candidates_dir`.

## What remains blocked

Only the annotation work: **E0.4** (double-annotate ~300 eval rows), **E2.4**
(Cattle labels — though the E1.9 result now supports merging instead), and
**E2.7** (candidate-stage recall, still the most important unmeasured quantity in
the system). No compute substitute exists for these.

E0.2's blocked retrain is also still open: the spatially-blocked split artifact
(`results/e02_blocked_splits.csv`) is ready, but a retrain on it was not part of
this fleet.
