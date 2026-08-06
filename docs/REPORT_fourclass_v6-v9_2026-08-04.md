# Four-class farm model — results report and roadmap
**2026-08-04 · covers the v6 → v7 → v8 → v9 line (round_2 / round_3 labeling), the class-balancing verdict, and ranked next steps**

---

## 1. Executive summary

Over two weeks we ran a controlled four-step experiment line on the 4-class SoftCon model
(NotFarm / Poultry / Pigs / Cattle), holding the architecture, optimiser, schedule and
augmentation **byte-identical** throughout and changing only the data/split assignment at
each step. All four models were scored over the full 157,102-candidate world and three of
them are published side by side on the website.

**The five results that matter:**

1. **One-sided labels teach a prior, not a feature.** Adding 962 false-positive
   corrections (all NotFarm) in round_2 cut farm false positives dramatically at the
   default threshold — but *only* by making the model reluctant to say "farm" anywhere
   unfamiliar. ROC-AUC did not improve, and at a matched FP rate recall was identical.
2. **Balanced promotion fixes it.** Adding 1,740 farm positives from the same pool (v8)
   raised AUC on every held-out slice (Albania+DR Congo 0.772 → 0.912) and, against the
   pre-round_2 baseline, cut FPs *and* raised recall simultaneously.
3. **The per-country cap is not a sensitive knob.** Rachel's round_3 70-cap is
   statistically indistinguishable from the 50-cap (AUC flat to three decimals) because
   most countries have fewer than 50 eligible farms anyway.
4. **Class balancing is a trade, not a win: in-domain up, out-of-domain down.** It
   replicated across two independent experiments (eval +0.046 / +0.027, generalization
   −0.054 / −0.016). Since the inference countries are the target, it stays off.
5. **The split lever is exhausted.** Three successive split interventions have left
   qual_eval AUC pinned at ~0.98. What remains: threshold/operating-point work,
   Pigs/Cattle labels, and modest model-side gains (ensembling, wider context).

**Recommended model today: v8 or v9 (equivalent).** The published v9 map predicts farm on
~68% of unlabeled candidates versus v6's 77% and v7's over-suppressed 58%.

---

## 2. What was run

| step | data change (single lever) | outcome in one line |
|---|---|---|
| **v6** | first model trained on all 167 countries' labels (32,046 — double v5, after fixing the 7-country ingestion bug) | baseline; strong in training countries, Poultry over-prediction elsewhere |
| **v7** | round_2: +962 NotFarm FP-corrections across 106 countries; ALB+COD added as generalization | FP collapse *and* recall collapse — a threshold shift, not learning |
| **v8** | + 1,740 farm positives from qual_eval (≤50/country, 80:20) | AUC up everywhere; FPs down and recall up vs v6 — the real fix |
| **v9** | Rachel's round_3 (≤70/country, 3 model classes only, LMIC-favouring) | == v8 within noise; cap size doesn't matter |
| **v9_bal** | + `balanced_class_sampling` (not published) | in-domain +0.046, out-of-domain −0.054; rejected for this target |

Integrity work underpinning the line (details in `EXPERIMENTS_LOG.md`):
the patch-store/cluster-id misalignment was repaired (34.6% → 0.31% stale) and a
coordinate guard now re-validates every patch at train/score time; every round's files
were verified for cluster-id + geometry stability before use (round_2/round_3: 100%
stable, so the patch store was reused with zero re-extraction); Rachel's
`cnn_split_assigned` was verified to flow **unmodified** from her files into the actual
training splits at every hop.

---

## 3. Results

### 3.1 Held-out metrics across the line

Farm-vs-NotFarm ROC-AUC (threshold-free — the metric that distinguishes real learning
from threshold shifts):

| slice | n | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|
| ALB+COD (plastic-cover test) | 179 | 0.777 | 0.772 | 0.912 | **0.916** |
| BGD+NGA | 247 | 0.979¹ | 0.910 | 0.936 | **0.937** |
| qual_eval (inference countries) | ~11.9k | 0.963¹ | 0.952 | 0.983 | **0.983** |
| eval (training countries) | 523 | 0.893 | 0.903 | 0.919 | 0.910 |
| test (training countries) | 2,094 | 0.994 | 0.993 | 0.993 | **0.994** |

¹ v6 numbers computed on the v6-era row sets; later columns on the common row set.

Farm FP-rate / recall at the default (argmax) threshold:

| slice | v6 | v7 | v8 |
|---|---|---|---|
| ALB+COD | 66.2% / 92.5% | 5.0% / 42.5% | **12.9% / 80.0%** |
| BGD+NGA | 3.1% / 92.6% | 0.0% / 59.5% | 6.2% / 78.6% |
| qual_eval | 4.9% / 86.8% | 0.6% / 58.9% | **3.6% / 90.4%** |
| eval | 39.3% / 95.4% | 26.7% / 91.8% | 31.1% / 95.9% |
| test | 6.8% / 98.5% | 7.5% / 97.8% | 7.5% / 98.1% |

v8 is the only model that improves both axes at once relative to v6 on the inference
countries — the original goal of the round_2/round_3 effort.

### 3.2 Why v7 failed: the prior-shift mechanism

The diagnostic triple that exposed it, and that should be standard for every future
comparison: (a) AUC flat (0.777 → 0.772 on ALB+COD); (b) identical recall at a matched FP
rate (95.0% for both models); (c) mean P(farm) fell for true NotFarms (0.091 → 0.021)
**and** for true farms (0.846 → 0.568). The model didn't learn to distinguish greenhouses
from farms — it learned "unfamiliar country ⇒ not a farm", because negatives were the
only lesson available in 106 of the countries.

### 3.3 Class balancing: a replicated in/out-of-domain trade

| experiment | test | eval | generalization |
|---|---|---|---|
| geofix matrix, 3-class (July) | −0.002 | **+0.027** | −0.016 |
| round_3 pair, 4-class (Aug) | +0.007 | **+0.046** | **−0.054** |

Mechanism: `eval` is drawn from the training countries the sampler re-weights within;
`generalization` has no training rows, so re-weighting only moves the boundary — and with
NotFarm at 22.7% of train vs Poultry's 63%, balancing hands NotFarm ~2.75× relative
weight, making the model cautious exactly where it is least informed. The memorisation
signature showed exactly where predicted: Cattle (153 train images drawn ~20×/epoch)
improved on in-distribution test (0.556 → 0.636) and collapsed on held-out qual_eval
(0.250 → 0.000). **Decision: balancing off while inference countries are the target;
revisit only if the deployment target returns to the five training countries.**

### 3.4 Map-level effects (the published full-world releases)

Farm predictions over the 154,905 common points: v6 **113,137 (73.0%)** → v7 **87,265
(56.3%)** → v8 **107,013 (69.1%)**; v9 lands at ~68%. v7 had effectively switched off
detection in whole countries (Saudi Arabia 69% → 11% of sites; restored to 65% by v8).
v8/v9 are net more conservative than v6 (median −4.3%/country on unlabeled rows) while
*better* on held-out recall — i.e. they reject things v6 got wrong rather than being
timid. Countries where v8 remains far below v6 and we cannot adjudicate without labels:
Nigeria (85.8% → 53.7%), Pakistan (90.2% → 65.4%), India (86.4% → 71.2%), Egypt.

### 3.5 Structural ceilings (unchanged across the entire line)

- **Pigs → Poultry confusion:** true pigs are predicted Poultry 42% / 27%² / 45% under
  v6 / v7 / v8. No split or sampling intervention has moved it. (² v7's "improvement"
  came from calling pigs NotFarm instead.)
- **Cattle is unmeasurable:** 153 train rows, 6–25 rows per held-out slice; F1 swings
  0.0–0.64 on noise. No modelling change substitutes for labels.
- **BGD+NGA operating point:** AUC improved (0.910 → 0.937) yet default-threshold recall
  regressed — a pure calibration problem, visible on the map as Nigeria's farm-rate drop.

---

## 4. New analyses for this report

**Ensembling v8+v9 is a small free win.** Averaging the two models' farm probabilities
(both already trained and scored): AUC +0.005 on ALB+COD (0.921), **+0.009 on BGD+NGA
(0.945)**, +0.002 on qual_eval (0.985). No training cost; scoring cost only.

**Global threshold calibration is a no-op — the miscalibration is domain-conditional.**
The val-optimal farm threshold is 0.481 vs the default ~0.5, changing OOD recall by ≤2.5
points. Val is dominated by training-country rows, so tuning on it cannot fix the OOD
operating point. What *does* have room is an OOD-specific threshold:

| threshold | BGD+NGA FP / recall | qual_eval FP / recall |
|---|---|---|
| 0.50 (default) | 3.1% / 76.3% | 3.1% / 90.3% |
| 0.40 | 3.1% / **81.4%** | 4.3% / 91.8% |
| 0.30 | 9.4% / **85.6%** | 5.5% / 93.5% |

A single lowered "screening" threshold (~0.3–0.4) for out-of-training-domain countries
recovers most of the lost recall at single-digit FP rates.

---

## 5. Recommended next steps, ranked

### A. Labeling (highest value — model changes cannot substitute)
1. **Round_4 spec: balanced promotion per country.** The v7/v8 pair is a controlled
   demonstration that FP-only correction backfires; every future round should promote
   confirmed positives alongside corrected FPs. (Rachel's round_3 already converged to
   this; keep it.)
2. **Pigs and Cattle labels are the binding constraint.** Pigs: 0.48–0.55 F1; Cattle:
   unmeasurable at 153 rows. Feedlots and lagoon complexes are visually distinctive —
   even +200–300 labels each would make these classes real.
3. **Adjudication samples where the map changed most:** ~50 labeled clusters each in
   Nigeria, Pakistan, India would settle whether the v8/v9 farm-rate drops there are
   correct rejections or lost farms — and double as round_4 training data.
4. **Restore the 26 BGD/NGA rows** that round_2 moved from generalization into
   train/val (or formally accept the impurity); ALB/COD remain the clean OOD reference.

### B. Deployment (cheap, immediate)
5. **Ship the v8+v9 ensemble as the map model** (+0.009 AUC on BGD+NGA for scoring cost
   only), or keep single-model v9 if simplicity wins.
6. **Two thresholds on the map, not one:** keep the precision-oriented default for
   confirmations, and expose a recall-oriented "screening" view at ~0.3–0.4 for
   out-of-training-domain countries. This directly fixes the BGD/NGA/Nigeria regression
   without any retraining.
7. **Two-stage presentation: farm gate → type.** NotFarm-vs-Farm is strong everywhere
   (F1 0.91–0.97); farm-type is the weak layer. Presenting P(farm) as the primary signal
   with type as a secondary attribute matches the model's real competence.

### C. Model experiments (moderate cost, moderate expected gain)
8. **ctx128 crop on v9 data** — wider context helped independently before (+0.017 eval)
   and was never combined with the corrected data; greenhouse/solar rejection plausibly
   benefits from context. One training run.
9. **TTA at inference** (8-flip averaging, already implemented behind `inference.tta`) —
   typically 1–2 points on minority-class F1 for 8× scoring cost; affordable sharded.
10. **Hard-negative mining round 2**: highest-confidence farm predictions on unlabeled
    rows in the worst-calibrated countries → Rachel verifies → round_4. This is the
    labeled-data flywheel the round_2/round_3 loop already proved out.
11. **Source-weighted loss** (down-weight noisy label sources) once Rachel confirms
    relative trust in OSM vs registry vs visual labels in the new countries.

### D. Evaluation protocol (adopt as standard)
12. **AUC-first comparisons** with matched-FP-rate recall as the tiebreaker; a
    fixed-threshold FP rate alone cannot distinguish better ranking from a moved
    threshold (v7 is the cautionary example).
13. Keep test / eval / generalization membership **frozen** across rounds — this is what
    made the v6→v9 line interpretable; qual_eval shrinks as rounds promote from it, so
    always compare on the newest round's held-out subset.

### E. Infrastructure hygiene (low cost, prevents money leaks)
14. `runpod_launch` must never be killed mid-hand-off: a truncated launch leaves a bare
    rented GPU that looks RUNNING but holds no work (burned money twice this round), and
    a duplicate launch can clobber a completed shard's non-atomic output. Launch
    detached (`setsid`) and verify the startup script landed.
15. The diagnostic CPU pod bills whenever left up (~$0.30/hr); kill between work
    sessions.
16. Website: 16 releases now total ~500 MB of tracked data; PMTiles remains the upgrade
    path if repo size or first-load latency becomes a problem.

---

## 6. Current published state

| release | model | points | role |
|---|---|---|---|
| `world_v10_fourclass_v7` | round_2 (negatives only) | 154,908 | cautionary baseline; high-precision/low-recall |
| `world_v10_fourclass_v8` | + positives rebalance | 154,908 | **recommended map** (tied with v9) |
| `world_v10_fourclass_v9` | round_3 (Rachel's 70-cap) | 154,908 | confirms cap insensitivity; equivalent to v8 |

All three carry the full 40-column audit trail (`points.csv.gz`), the exact training
config, and release notes stating the caveats (BGD/NGA operating point, Pigs→Poultry
confusion, Cattle sample size). The class-balanced variant is deliberately unpublished.
