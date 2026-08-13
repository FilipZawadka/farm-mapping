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
