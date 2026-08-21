# Evaluation methods — round_4 campaign

*Produced by a 21-agent literature survey (six angles, every finding adversarially
re-checked), then **verified against our own data** before adoption. This file is a
pre-registration: it is fixed before any round_4 scores are inspected.*

## Verification log — what we checked before trusting the survey

| Survey claim | Our check | Outcome |
|---|---|---|
| Row bootstrap is blind to seed variance and manufactures significance | Ran our own `paired_bootstrap_delta` on the 5 identical-recipe seed runs, aligned on 426 common generalization rows | **CONFIRMED — 3 of 10 identical-recipe pairs declared "significant"** (p=0.006 / 0.036 / 0.037) |
| σ_seed on generalization ≈ 0.0080 | Measured from the same 5 runs | **CONFIRMED — 0.0078**, two-run band ±0.0220 AUC vs effects of ~0.0045 |
| The 2,000-row blind slice may lack NotFarm rows | Read round_4 `qual_eval` label mix | **CONFIRMED, worse than stated** — zero NotFarm; it is entirely Farm:Unknown / Ambiguous / PigsOrPoultry. Binary AUC is undefined. Slice **retired**. |
| Archived models cover only ~247/426 of the generalization slice | Joined archived scored parquets against round_4 gen ids | **REFUTED — coverage is 661–662 / 662 (100%)**. Cross-version comparison on the primary slice is sound. |

## Design change this forced

The survey was written for the original **one-run-per-config** design, and correctly
concluded that such a design can only support Dietterich's **Q3** ("which trained
artifact ranks these rows better"), never **Q8** ("which recipe is better").

Because σ_seed turned out to be ~5× the effects under test, we changed the design
rather than accept the weaker claim:

- **5 single-factor arms × 3 seeds = 15 runs.** Arms: A baseline · B freeze0 only ·
  C 6-bands only · D freeze0+6bands (composition test) · E DenseNet on top of D.
- The original "best" arm bundled freeze0 **and** 6-bands — the same unattributable
  -bundle error as the rejected v10 regularizer experiment. Arms are now
  single-factor so any effect is attributable.
- With 3 seeds per arm we can report **per-arm mean ± seed SE** and compare arms as
  two small samples, which moves the recipe comparisons from Q3 toward Q8. It does
  not fully reach Q8 — 3 seeds is a small sample and training-set variability is
  still not resampled — so claims stay hedged accordingly.

Where the note below says "one run per config", read "three seeds per arm"; its
mandatory seed-variance term (`SE_total = sqrt(SE_boot² + 2σ_seed²)`) still applies,
and is now partly *measured* per arm rather than imported from the v9 config.

---

# Evaluation Methods — round_4 seven-model comparison

**Status:** pre-registration. Everything below is fixed before any round_4 scores are inspected.

## 1. What we compare, and what each comparison can claim

We adopt Dietterich's (1998) taxonomy as a scoping rule, in two registers: **Q3** = "which trained artifact ranks these rows better" (answerable from one run per config); **Q8** = "which recipe is better" (not answerable — one run per config gives zero degrees of freedom for seed variance). Note the register split is justified on df=0, *not* on our 0.033 macro-F1 seed sigma, which is a 6-row Cattle artefact.

| Contrast | Nature | Licensed claim | Barred |
|---|---|---|---|
| **A vs B** | same round_4 data/splits; B = no freeze-warmup **and** 6 bands | "checkpoint B ranks better/worse than A" | any attribution to freeze-warmup *or* band count — two factors move together, unattributable at any n |
| **B vs C** | backbone family + init + pretraining domain all swap (SoftCon RN50 → ImageNet DenseNet-121) | "checkpoint C vs checkpoint B" | "ImageNet beats SoftCon", "DenseNet beats ResNet" |
| **v6–v9 vs A/B/C** | release comparison: recipe + train-set size + label-release quality + known patch-store ID misalignment all move together | Q3 ship claim ("which checkpoint do we deploy") | any recipe or architecture claim |

Archived arms stay as *tested* Q3 arms (Q3 is defined over artifacts, regardless of provenance) with the confound as a footnote — **conditional on two data-integrity gates run first**: (a) verify v6–v9 held-out membership under the pre-2026-06-23 ID scheme and confirm join rows agree on lat/lng within ~100 m; (b) confirm the 2,000-row blind slice actually contains NotFarm rows (in v10 qual_eval it did not, making binary AUC undefined). Report the inner-joined complete-case n, not the nominal 662/2,164/590/2,000 — archived coverage was only 247/426 on the old generalization slice.

## 2. Primary metric and endpoint

**Binary farm-vs-not ROC-AUC (1 − P(NotFarm))**: threshold-free, rank-based, prevalence-invariant, and unchanged by any recalibration — so it survives our slices' differing farm base rates (0.60–0.80 vs 0.16). Primary slice: **generalization** (662 rows, 6 zero-training countries) — a leave-country-out block design, the strongest methodological asset we have. Primary contrast: **B vs A**. Secondary confirmatory: C vs B; best-new vs v9.

Not eligible as decision metrics: 4-class macro-F1 (~80% of its variance is a 6-row Cattle class; merge to 3-class, which we measured as free), and PR-AUC across slices (precision rescales deterministically with prevalence). Report per-class F1 with support printed, and precision/NPV at the shipped 0.4 threshold.

## 3. Tests and their validity conditions

**Estimator (all ΔAUC contrasts):** paired shared-index bootstrap (`experiments/lib.py:206`), with `BOOT_N` raised from 2,000 (line 56) to ≥10,000 so p-values stop being floored at 2/2000. *Valid when:* all models scored on identical rows (they are, after the inner join). *Assumes:* i.i.d. rows — which is false here.

**Mandatory seed term.** Report `SE_total = sqrt(SE_boot² + 2·σ_seed²)`, with per-slice σ_seed measured from the five archived identical-config runs: **0.0080 (generalization), 0.0053 (eval), 0.0006 (test), 0.00076 (11,365-row frozen benchmark)**. *Why it is mandatory:* on those identical-recipe runs the row bootstrap alone declares 3 of 6 pairs significantly different at p<0.05 on generalization. *Limit:* σ_seed is measured on the SoftCon v9 config only; extrapolation to DenseNet C is an assumption, stated as such. Re-measure σ_seed on the 2,000-row blind slice — 0.00076 does not transfer to it.

**Spatial term (sensitivity column, not headline).** Report a hex3/10-km block-bootstrap SE alongside. Measured design effects on generalization span 0.92–1.74 (hex3) up to 2.19 (country); country ICC on the farm label is 0.57 on the frozen benchmark. With 6 countries (df=5) no cluster-robust variance is credible, so **no number is labelled "cluster-adjusted"**; every SE is declared a lower bound on true uncertainty.

**Thresholded decisions at 0.4:** exact McNemar on the discordant counts, reporting n01/n10 alongside p. Never applied to AUC (category error) and never to macro-F1. Dietterich's own caveat carries: McNemar cannot see training-set or seed variability.

## 4. Multiplicity

Confirmatory family = the **3 pre-registered contrasts** on generalization farm-AUC, Holm-adjusted (α=0.0167 on the smallest p) — valid under arbitrary dependence, which we need since all seven models share rows. All other cells (21 pairs × 4 slices) are exploratory, reported unadjusted and labelled as such. Slice hierarchy is fixed here — generalization primary; test/eval/blind supporting — so that reporting the friendliest slice is not available to us. We report the full ranking with intervals, including every model whose interval overlaps the leader's, because selecting the best of seven and quoting its interval is winner's-curse biased.

## 5. Practical-significance floor and MDE

Floor: **0.005 AUC**, carried over unchanged from the E2.2 ensemble decision (a +0.0018 gain at p<0.001 that we declined). A win requires point estimate > floor **and** Holm-adjusted p < 0.05.

MDE is computed **empirically and per pair**: `MDE_80 = 2.80 × SE_total`. We do **not** use the Obuchowski–McClish / Hanley–McNeil closed forms: on our own frozen benchmark Hanley–McNeil returns a 95% half-width of 0.0040 where we measured 0.0030 (32% off), because AUC 0.983 at 18% positives sits far outside the balanced, moderate-AUC regime those tables assume. MDE is per *pair* because it is driven by ρ, which is not constant (on generalization, MDE_80 = 0.032 at ρ=0.80 vs 0.0071 at ρ=0.99); in particular the 0.0075 half-width from v8-vs-v9 (ρ≈0.99) must **not** be carried to the C-vs-B backbone swap.

Stated up front, before results: seed variance alone puts a two-run decision band of ~±0.023 AUC on generalization, against a largest-ever-measured recipe effect of +0.0045 (freeze0) and an expected A-vs-B effect of ~+0.0035. **Generalization is therefore a direction and OOD check, not the instrument that decides.** Nulls are written "underpowered below Δ = X", never "no difference".

## 6. Calibration

Diagnostic only; the seven models are **not ranked** on it. Temperature scaling cannot change rank order and therefore cannot move the headline AUC (Guo et al. 2017), so calibration is orthogonal evidence, never a tiebreak. Power forbids more: below ~500 rows only >10 pp miscalibration is reliably detectable, and 2% needs >10,000 (Roelofs et al. 2022) — every slice here is under. Actions: consolidate the two incompatible ECE implementations (`experiments/lib.py:102`, n_bins=15, vs `scripts/full_metrics_report.py:66`, bins=10 — these are not the same statistic and must never share a table), move to equal-mass bins, and report Brier decomposed (CORP MCB/DSC/UNC, Dimitriadis et al. 2021) **within a slice only**, since the uncertainty term is a pure function of base rate. Exclude Cattle.

## 7. What we will not claim

No recipe-, architecture-, initialization- or band-level claim from any single-run contrast. No decomposition of the A→B bundle. No "ImageNet beats SoftCon". No "no difference" from an unrejected null. No map-accuracy or absolute-precision statement — our slices are not probability samples of any region, so under Olofsson et al. (2014) this is a *relative model comparison on a frozen benchmark*, and an absolute claim needs a fresh stratified sample of the scored 157k output. No significance test on 4-class macro-F1. No cross-slice metric comparison for Brier, ECE or PR-AUC.

## 8. Rejected methods (and why)

- **DeLong / fast DeLong / DeLong omnibus χ² (DeLong 1988; Sun & Xu 2014):** conditions on fitted scores, so it is blind to seed variance — on our identical-config runs it calls 3/10 pairs significant at p<0.05 on the primary slice. Its SE also matches our existing paired bootstrap to ~1%, so it adds no information, only the most optimistic p-value on the page. Fast DeLong is additionally *slower* than textbook DeLong at n=426.
- **Cluster bootstrap as the primary interval:** `cluster_id` is our primary key (1 row = 1 id), so resampling it *is* the row bootstrap; at the real grouping level (hex3) the measured design effect on labeled slices is ~1.0.
- **Unpaired AUC comparison (generalization vs test):** country is perfectly confounded with training exposure by construction; within-generalization per-country AUC spread (0.134) exceeds the gap being tested.
- **Stratified paired bootstrap:** measured no-op — P(a resample loses a class) is 1e-68 to 1e-207 for binary AUC; variance *reduction* is the wrong direction for a procedure that already understates uncertainty.
- **Bandos–Rockette–Gur permutation (2005); Venkatraman (1996/2000):** the permutation null destroys exactly the between-country dependence at issue; Venkatraman is rank-invariant and so provably blind to the 0.4 operating point it was proposed to protect (p=1.000 under monotone rescaling that moves sens@0.4 by 20 pts).
- **max-t / pairs bootstrap (Westphal & Zapf 2024):** measured contrast correlations are 0.12–0.90 with one negative; max-t buys 0.0004 AUC of MDE over free Holm.
- **TOST / equivalence verdicts (Liu et al. 2006):** at our margin the rule is degenerate — 10/10 known-null seed pairs return INCONCLUSIVE on generalization, and a real +0.072 gap also returns INCONCLUSIVE.
- **P(A>B) with γ=0.75 (Bouthillier et al. 2021):** degenerate at k=1 run (CI collapses, rule auto-passes); inapplicable to archived checkpoints by the authors' own stated boundary.
- **Demler nested-model protocol (2012):** our models are non-nested and evaluated out-of-sample, so the degeneracy does not arise; its gating first step has no referent for end-to-end CNNs; and it would push our intervals in the anti-conservative direction.
- **5×2cv (Dietterich 1998, §3.5); Friedman/Nemenyi (Demšar 2006):** 5×2cv resamples the training set, destroying the fixed round_4 split and the country-disjoint construction of the generalization slice; Friedman over N=4 non-independent slices is theatre.

## 9. Sources

Dietterich, *Neural Computation* 10(7):1895–1923 (1998) — Q3/Q8 taxonomy, §3.1 McNemar caveat, §3.5. Hoenig & Heisey, *Am. Stat.* 55(1) (2001) — MDE over observed power. Reimers & Gurevych, arXiv:1803.09578 — single-score comparison. Bouthillier et al., MLSys (2021) — variance accounting. Holm, *Scand. J. Stat.* 6:65–70 (1979). DeLong et al., *Biometrics* 44:837–845 (1988); Sun & Xu, *IEEE SPL* 21(11) (2014); Liu et al., *Stat. Med.* 25(7) (2006); Bandos, Rockette & Gur, *Stat. Med.* 24:2873–2893 (2005); Venkatraman & Begg, *Biometrika* 83:835–848 (1996); Westphal & Zapf, *SMMR* 33(4) (2024); Demler, Pencina & D'Agostino, *Stat. Med.* 31(23) (2012) — all rejected, §8. Guo et al., ICML (2017); Roelofs et al., AISTATS (2022); Dimitriadis, Gneiting & Jordan, *PNAS* 118(8) (2021); Gneiting, Balabdaoui & Raftery, *JRSS-B* 69(2) (2007) — calibration. Roberts et al., *Ecography* 40:913–929 (2017); Kattenborn et al., *ISPRS OJPRS* 5:100018 (2022) — block/leave-region-out design. Olofsson et al., *RSE* 148:42–57 (2014) — why this is a benchmark, not a map-accuracy statement. McDermott et al., NeurIPS (2024); Williams, arXiv:2007.01905 — AUROC over AUPRC, prevalence rescaling. Opitz & Burst, arXiv:1911.03347 — macro-F1.

*In-repo:* `experiments/lib.py:56,102,206`; `scripts/full_metrics_report.py:66,77`; `docs/EXPERIMENT_COMPENDIUM.md` (Part 0 seed study, E2.2 floor); `experiments/gpu_results/e03_seed43–46/`.

## Estimands: recipe vs artifact (added 2026-08-21, pre-results)

Caught by a synthetic-data smoke test of `evaluate_r4.py` run before any round_4 result
landed (`scratchpad/test_eval_r4.py`, injected ground truth b,d > a > e).

Two different quantities were being printed side by side and were easy to conflate:

| Estimand | Definition | Answers |
|---|---|---|
| **`dAUC_rec`** (PRIMARY) | difference in **mean per-seed AUC** | "is recipe B better than recipe A?" — Dietterich's Q8, the question every arm was built to answer |
| `dAUC_ens` (secondary) | AUC of the **seed-averaged probability** | "is this 3-seed ensemble artifact better?" — closer to Q3 |

`dAUC_ens` systematically **compresses** recipe differences, because averaging seeds
removes independent noise and pushes both arms toward the ceiling. In the smoke test a
true +0.0162 recipe effect appeared as +0.0004 at ensemble level — a ~40x shrink that
would have been reported as "no difference". The report leads with `dAUC_rec`; `dAUC_ens`
is retained as a column because a deployed model *is* an artifact, but it is not the test.

`SE_total` keeps the mandatory seed term `sigma_seed^2 (1/n1 + 1/n2)`. Its row-noise
component `se_boot` is taken from the ensemble bootstrap as a proxy for the
mean-per-seed statistic; the seed term dominates it by roughly 5x, so the approximation
does not move any verdict.

### Bug found and fixed by the same test
`CONFIRMATORY` was written as `("b","a")` while contrast rows are keyed by
`itertools.combinations` order (`"a_vs_b"`). No key ever matched, so `conf` was empty and
**the entire Holm-corrected confirmatory section silently printed nothing**. The pre-registered
headline analysis would have been missing with no error raised. Keys are now resolved in
either order with an explicit sign flip, verdicts name their direction
("b BETTER than a"), and an explicit warning fires if the family ever resolves empty.

## Amendments to the pre-registration (2026-08-21, mid-campaign)

**What changed.** Arm **F** (`freeze=5` + `unfreeze_lr_scale=1.0`) was added, and the
confirmatory family grew from 3 comparisons to 5:

| Contrast | Isolates |
|---|---|
| `b > a` | freeze0 recipe vs baseline recipe (original) |
| `d > a` | freeze0 + 6 bands vs baseline (original) |
| `e > d` | DenseNet-121 vs SoftCon (original) |
| **`f > a`** | **pure LR effect** — warm-up held fixed in both arms |
| **`b > f`** | **pure warm-up effect** — backbone LR held at 1e-4 in both arms |

Holm-Bonferroni now corrects across m=5 rather than m=3, which is *more* conservative for
the three original contrasts. No original contrast was removed or redefined.

**Why, and what I had seen.** Full disclosure, because this is an amendment made after data
started arriving. The trigger was a **code-inspection** finding, not an outcome: `train.py`
hard-coded `lr_scale=0.1` at the unfreeze transition, so `freeze_backbone_epochs > 0` silently
coupled a frozen warm-up to a permanent 10x backbone LR cut (R4_RUN_NOTES section 3). What
prompted the inspection *was* partial data — arm A at val F1 0.640 vs arm B at 0.824, mid-training
on the val split. So this amendment is **not blind**.

Mitigating factors, stated so a reader can discount appropriately:
- No arm-F result existed, or could exist, at the time of the amendment — the config it needs was
  unrepresentable until `unfreeze_lr_scale` was added.
- The evidence was the *validation* curve used for checkpointing, not any test/eval/generalization
  slice; all reported contrasts are computed on the held-out slices.
- The amendment adds a mechanism decomposition; it does not alter the original three hypotheses.

**Effect on the historical record.** The prior headline "freeze0 is the only validated win in 20
runs" stands as a claim about recipes but its stated mechanism was wrong: freeze0 removes the
warm-up *and* raises the backbone LR 10x. Arms A/B/F now decompose that; until F reports, no
mechanistic claim about freezing should be made.
