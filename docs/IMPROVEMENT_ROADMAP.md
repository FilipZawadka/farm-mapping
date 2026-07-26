# Improvement roadmap — post-v8 analysis & every lever we have (2026-07-06)

This document does three things:

1. **Analyzes the v8 experiment matrix results** (what we learned, what's
   still broken).
2. **Diagnoses the remaining failure modes** with new data analysis of the
   split composition (label sources, geography) — this changes the reading
   of the eval-set numbers substantially.
3. **Catalogs every improvement lever we have**, each with a verdict,
   expected impact, and effort — then distills a ranked v9 shortlist.

Companion docs: [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) (full v3→v8
results), [`EXPERIMENTS_v8.md`](EXPERIMENTS_v8.md) (v8 plan + audit
findings), [`pipeline/`](pipeline/README.md) (codebase reference).

---

## 1. Where we stand after v8

| Run | test macroF1 | eval macroF1 | eval f1 (NF/P/OF) | gen macroF1 |
|---|---|---|---|---|
| v8_three_class (ImageNet, 4ch) | 0.626 | 0.403 | 0.444 / 0.747 / 0.018 | 0.397 |
| v8_logitadj | 0.572 | 0.470 | 0.549 / 0.596 / 0.265 | 0.330 |
| v8_ssl4eo (S2-native, 9ch) | 0.723 | 0.489 | 0.475 / 0.724 / 0.269 | 0.407 |
| **v8_crt (ssl4eo + balanced head)** | **0.727** | **0.494** | 0.495 / 0.723 / 0.264 | **0.426** |
| v8_cloudfree | *blocked (disk quota mid-extraction)* | | | |

**What v8 established:**

- **The split bug is fixed and confirmed** — OtherFarm went from
  literally-unseen (v6/v7) to test F1 0.45–0.62.
- **S2-native pretraining (SSL4EO) is the biggest single lever we've
  pulled** — +0.10 test macroF1, +0.09 eval, +0.17 test OtherFarm over the
  ImageNet baseline. The research prediction held exactly.
- **cRT adds a small OOD bump** (+0.019 gen) on top; current best overall.
- **Post-hoc logit adjustment is exhausted** — ssl4eo/logitadj already sit
  at the τ=0 optimum. The remaining ceiling is **representation +
  labels**, not the decision rule.
- **AdaBN was negative → the BGD/NGA gap is NOT radiometric.** SSL4EO's
  pretraining already covers target-country pixel statistics; what differs
  is morphology and label distribution. Pixel-statistics tricks are off
  the table for OOD.

**What is still unsatisfactory:**

- eval macroF1 0.494 vs test 0.727 — a 0.23 gap concentrated in
  **NotFarm (0.73 → 0.50)** and **OtherFarm (0.61 → 0.26)**. Poultry
  transfers fine (0.84 → 0.72).
- gen macroF1 0.426, and gen OtherFarm is **unmeasurable** (4 labeled rows
  in BGD, 0 in NGA).

---

## 2. New diagnosis: the eval gap is a *label-source and geography shift*, not a modeling failure

Composition analysis of the master parquet (via the v5 scored parquet —
routing identical from v4 onward) shows the train and eval slices measure
**different populations**:

### 2.1 OtherFarm: train is registry-US, eval is visual-non-US

| | train OtherFarm (n=2,218) | eval OtherFarm (n=98) |
|---|---|---|
| Label source | **98% registry** (IA_official 833, Mexico_official 452, Thailand_FT 273, MN/NC/IN/CA officials, Brazil_MfA 124, Chile_Liebsch 71) — only 43 visual | **80% visual inspection** (78 rows); 20 registry |
| Geography | **USA 56%** (1,252), MEX 466, THA 299, BRA 130, CHL 71 | **USA 3 rows (3%)** — THA 45, CHL 19, MEX 16, BRA 15 |
| Animal type | registry pig CAFOs dominate (Iowa hogs) | visual sublabels: 73 "Farm: Pigs", 5 "Farm: Cattle", 20 blank |

The natural-distribution test split inherits the train composition
(~56% US, ~98% registry), so **test OtherFarm F1 = 0.62 is a measurement
of "US registry pig-CAFO lookalikes" — the population the model trained
on**. Eval OtherFarm F1 = 0.26 is a measurement of "visually identifiable
pig/cattle farms in TH/CL/MX/BR". These are different tasks. The model
isn't failing to learn; **it's learning exactly what we fed it.**

Per-country eval OtherFarm F1 (v5, the last run with local per-country
artifacts): THA 0.34, CHL 0.28, BRA 0.28, MEX 0.21. Even the best country
sits at ~0.34 — the gap is systematic across countries, proportional to
how little non-US OtherFarm the train set has.

### 2.2 NotFarm: train negatives are auto-sourced, eval negatives are hard

Train NotFarm: **2,807 of 3,063 rows have no label_source** (pipeline
negatives), 128 visual. Eval NotFarm: 175/177 visually confirmed — and
they are *candidates*, i.e. building clusters that looked farm-like enough
to enter the candidate pool. The eval negatives are **hard negatives by
construction**; most train negatives are not. This is why NotFarm F1
craters from 0.73 (test) to 0.50 (eval) — same mechanism as OtherFarm,
different class.

### 2.3 Generalization OtherFarm is statistically unmeasurable

BGD: 24 NotFarm / 143 Poultry / **4 OtherFarm**. NGA: 34 / 68 / **0**.
`gen f1_class2 = 0.000` will stay 0.000 regardless of model quality.
Poultry gen F1 is actually *good* (0.79–0.83 for ssl4eo/crt) — the OOD
story is mostly a NotFarm-precision story + an unmeasurable-OtherFarm
story.

### 2.4 What this means

> The single highest-leverage direction is **not architecture — it's
> aligning the training distribution with the deployment distribution**:
> more non-US OtherFarm labels, hard-negative mining, and using the 125k
> unlabeled candidates. The second is keeping the representation gains
> coming (SoftCon next). Architecture/loss tinkering below these two is
> rearranging deck chairs.

The 125,812-row unlabeled candidate pool is the largest untapped asset in
the project.

---

## 3. The catalog: every lever, with verdicts

Verdicts: **DO** (evidence-backed, high impact-to-effort) · **TRY**
(plausible, cheap enough to test) · **LATER** (real ceiling, real cost) ·
**SKIP** (evidence says no).

### A. Data & labels (highest expected impact)

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| A1 | **Non-US OtherFarm label enrichment** — ask Rachel to prioritize labeling OtherFarm candidates in THA/MEX/BRA/CHL (model-assisted: send her the top-K `prob_class2` unlabeled candidates per country) | **DO** | eval OtherFarm F1 is capped by having only ~950 non-US OtherFarm train rows vs 98 eval targets from a different source | Labeling session(s) | The composition table above is the argument |
| A2 | **BGD/NGA labeled set (50–200/country)**, stratified over candidate types; include OtherFarm hunting | **DO** | Only way to make gen OtherFarm measurable; literature says ~100 target labels beat every unsupervised DA trick | Labeling budget | Also enables few-shot fine-tuning (D4) |
| A3 | **Label-noise audit** — cleanlab confident-learning on out-of-fold probabilities, rank-fused with AUM margins; review top ~500 stratified by country × label_source | **DO** | Registry labels can be wrong at the *candidate matching* step (registry point → nearest building cluster); visual pig/poultry labels are inherently noisy. ~51% of confident-learning flags are human-confirmed errors in benchmark studies — an efficient review queue. If flags concentrate in one country/source, that *is* the eval-gap mechanism | 2–3 days incl. review | Script sketch in §5; strongest-evidence single action per the research sweep |
| A7 | **Double-annotate ~300 eval rows** (Rachel + a second rater, blind) | **DO** | Inter-rater F1 on eval OtherFarm is an *upper bound* on measurable model F1 — the decisive test between "label ceiling" and "feature ceiling". If humans agree at ~0.5, our 0.27 is near the ceiling; if 0.9, spend on features/fusion | One labeling session | Bayes-error framing: Ishida et al. ICLR 2023 |
| A4 | **Hard-negative mining for NotFarm** — add visually-confirmed non-farm building clusters (industrial, greenhouses, warehouses) to train; source candidates from high-`prob_class1/2` unlabeled rows Rachel rejects | **DO** | Directly attacks the 0.73→0.50 NotFarm eval gap; cheap because rejects are a by-product of A1 review | Piggyback on A1 | |
| A5 | **Split OtherFarm into Pigs / Cattle** (4-class or hierarchical head) | TRY | Pig barns (long, narrow, lagoons) and cattle feedlots (large bare pens) look *nothing* alike at 10 m; forcing one class may blur both prototypes. Registry sources already encode animal type for most rows | Small config + label-map change | Guard: cattle rows may be scarce — check counts first |
| A6 | **Audit the registry→cluster matching** — for each registry-labeled candidate, check distance from registry point to cluster centroid; flag >500 m matches | TRY | Silent mislabels poison both train and test | Half-day script | |

### B. Representation / backbone

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| B1 | **SoftCon ResNet-50** (HF `wangyi111/softcon`, repo `zhu-xlab/softcon`; 13-band S2) | **DO** | Successor to our SSL4EO weights from the same lab; tops the independent Panopticon-paper GEO-Bench table on m-EuroSAT 64 px linear probe (**80.0** vs ~70 for peers); the model behind Earth Genome's Earth Index poultry-CAFO success | ~1 day (same band-mapping machinery as ssl4eo) | Highest evidence-to-effort backbone option; decision point from EXPERIMENTS_v8 §3 triggered |
| B2 | **Frozen-embedding + gradient-boosted head probe** — pooled SoftCon/SSL4EO embeddings for all labeled rows → LightGBM | TRY | PANGAEA finding: fine-tuning still wins at 15k labels — so frozen+GBM is *not* a replacement, but it is the right tool for cheap iteration, the natural fusion point for footprint features (F1), and a decorrelated ensemble member | ~1 day, CPU only | Earth Index's production recipe is this shape (frozen SoftCon + light head) |
| B3 | Controlled ablation: ImageNet ResNet with the same 9 bands as ssl4eo | TRY | Separates "S2 pretraining" from "more bands" in the ssl4eo win — informs the next backbone choice | 1 config, 1 run | v8_three_class used 4ch; ssl4eo used 9ch — confounded |
| B4 | **AlphaEarth / Google Satellite Embedding V1** (GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 10 m, 64-dim, annual, global, 2017–2024) — pull per-site vectors for all 15k labeled + 125k unlabeled, train LightGBM | TRY | Zero training cost; encodes a full year of multi-sensor time series (captures most of the temporal signal C3 would buy); trivially covers BGD/NGA → doubles as a global candidate ranker and a decorrelated ensemble member | ~1 day (GEE export + GBM) | Evidence is Google's self-eval — treat as promising, verify on our eval slice |
| B5 | Panopticon ViT-B (wavelength-conditioned channels; CVPR'25 EarthVision best paper) | LATER (after B1) | 78.4 m-EuroSAT LP — close to SoftCon; native flexible band sets | 2–3 days (ViT switch) | |
| B6 | Clay, Prithvi-EO-2.0, SatMAE++, AnySat, Galileo (single-timestep) | SKIP | Clay: no credible small-patch classification evidence; Prithvi: 30 m/6-band mismatch; SatMAE++/AnySat: mid-table everywhere; Galileo's wins are time-series-pixel tasks (trails SoftCon by ~10 pts here) | — | Re-open Galileo only if we move to real time-series input |

### C. Inputs (what the model sees)

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| C1 | **Wider context crop** — `crop_center_px: 96` or full 128 (1.28 km) instead of 64 (640 m) | **DO** (cheapest experiment we have) | Pig lagoons, feedlot pens, feed mills sit *around* the barns; 640 m may clip the discriminative context. Zero re-extraction: patches are already 128 px | 1-line config change ×1 run | Interacts with B1: SSL4EO pretrained at 264 px |
| C2 | **Finish `world_v8_cloudfree`** (SCL masking) — blocked on disk quota, not on merit | DO (unblock) | Cleaner composites for THA + both OOD countries | Free disk / re-run extraction | Quota bump or prune old patch stores |
| C3 | **Seasonal / temporal channels** — 2 seasonal (wet/dry) medians + per-band temporal std | TRY (demoted) | Research check: multi-temporal gain evidence is concentrated in *crop phenology*, thin for static man-made structures — the signal here is contextual (surrounding field cycles, lagoon dynamics). Expect +0–2, kill fast if flat. Note B4 (AlphaEarth) already encodes the full annual time series for ~zero cost — run that first | Re-extraction + provider change | Do after C2 unblocks extraction, if B4 shows temporal signal matters |
| C4 | **Sentinel-1 VV/VH fusion** | LATER | Structure-sensitive, weather-independent; but needs the per-source scaling fix first (dB channels would be mangled by /10000 + clip) | Provider exists; scaling scheme ~1 day + re-extraction | |
| C5 | Higher-res basemap imagery | SKIP | ToS-prohibited (Google/Esri/Bing); NICFI discontinued | — | Unchanged from v8 research |

### D. Training recipe

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| D1 | **Noisy-Student self-training on the 125k unlabeled pool** — teacher = best model (soup/crt); *per-class quota* pseudo-label selection (NOT a global threshold); student = fresh SSL4EO/SoftCon init + strong augmentation | **DO** | The pool is 8× the labeled set and covers the exact deployment distribution incl. BGD/NGA candidate sites. Direct evidence in our regime: on FMoW-WILDS (satellite, geographic OOD) Noisy Student gained **+3.0** worst-region pts while **FixMatch actively hurt** (32.1 vs ERM 34.8) — do NOT use FixMatch-style consistency training | ~2–4 days (pseudo-label script + config) | Details & guardrails in §5 |
| D2 | **Country-balanced OtherFarm sampling** — oversample non-US OtherFarm rows only (not the blanket class-balanced sampler that broke v5) | TRY | Points the existing 950 non-US OtherFarm rows at the eval distribution | Small sampler change | Cheap, targeted version of what A1 does properly |
| D3 | MixStyle / stronger photometric augmentation | TRY (demoted) | AdaBN's negative result says radiometric shift isn't the OOD gap — so style augmentation attacks a non-dominant term | Small model hook | Only bundle with another run, not standalone |
| D4 | **Few-shot target fine-tuning** once A2 labels exist: (1) plain oversampled mixing (weight 5–20×, tuned) as the baseline, (2) LP-FT (linear-probe→fine-tune), (3) WiSE-FT α-interpolation between pre/post-adaptation weights | DO (gated on A2) | Best-evidence stack at 50–200 labels. LP-FT beat full FT by ~+10 OOD in the ICLR'22 study (full FT *distorts* features at tiny budgets); boring oversampling is the strongest-evidenced baseline (Wiles et al.: clever DG methods are inconsistent, target data + augmentation win) | Small trainer addition | WiSE-FT's headline gains are CLIP-specific — expect the small end (+0.5–2) for ResNets |
| D5 | **Surgical fine-tuning** — freeze conv1–layer3, tune layer4+head | TRY | ICLR'23 result: tune *later* blocks when the shift is output/label-level — which is exactly what AdaBN-negative + cRT-positive already told us about our shift | 1-hour config change | Cheap ablation to bundle into the next run |
| D6 | Focal/CB/LDAM-style losses | SKIP | Settled by v8 research + results: imbalance is not the binding constraint | — | |

### E. Decision layer & product shape

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| E1 | **Two-stage cascade**: binary Farm/NotFarm (v3 binary head hit **0.90 eval F1**) → Poultry-vs-OtherFarm second stage on positives | TRY | Decouples "is it a farm" (solved) from "which animal" (hard); each stage gets a cleaner objective and its own threshold | Mostly config + small inference glue | Compare against flat 3-class on identical eval |
| E2 | **Per-country priors / thresholds** at inference (per-country logit adjustment using expected candidate-pool priors) | TRY | v8 per-country confusion structure differs by country; one global argmax is leaving F1 on the table | Post-hoc script, extends `logit_adjust_sweep.py` | |
| E3 | **Abstain + human triage tier** — expose `confidence_tier` cuts so Rachel reviews only low-margin predictions | DO (product) | Converts residual model uncertainty into a bounded review workload; feeds A1/A4 labels back | Already have `confidence_tier` — needs calibration + docs | Active-learning loop closes here |
| E4 | **Greedy soup + WiSE-FT interpolation** over the v8 checkpoint family (ssl4eo, crt + future SoftCon variants) | **DO** (first thing to run) | Model-soups paper: soups ≥ output ensembles OOD, replicated on FMoW-WILDS (satellite geographic shift). Expected +0.5–1 ID, +1–3 OOD for ~2 hours of work. cRT and the ssl4eo fine-tune share an init → α-sweep between them is free | ~2–4 h | Only average checkpoints fine-tuned from the SAME pretrained init; select greedily on per-country eval macro-F1, not test |

### F. Fusion with what we already know about each candidate

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| F1 | **Footprint-geometry probe then fusion** — LightGBM on `num_bldgs`, `total_area_m2`, `median_area`, `template_score_if` (+ per-building aspect ratios if computable) | **DO** | Published poultry signature (Robinson et al. aspect-ratio/area filter); geometry is more country-invariant than radiometry — the exact axis AdaBN told us we're missing for OOD | Probe: hours. Fusion: 1–2 days | Natural home: concat with B2's frozen embeddings |
| F2 | Auxiliary lagoon/feedlot detection channel (NDWI blob near barn cluster) | LATER | Pig-farm-specific discriminator visible at 10 m | Feature-engineering day | Fold into F1's feature set first |

### G. Evaluation & methodology (make the numbers trustworthy)

| # | Lever | Verdict | Impact | Effort | Notes |
|---|---|---|---|---|---|
| G1 | **Report eval metrics split by label_source** (registry vs visual) and by country, with bootstrap CIs | **DO** | §2 shows pooled eval F1 conflates two populations; n=98 OtherFarm ⇒ ±0.08-ish CI on F1 — we've been over-reading small deltas (e.g. crt's +0.019) | ~1 day in post-train eval | |
| G2 | **Spatial-block splits** for the headline test | DO | Random within-country splits leak near-duplicate neighbors; test is optimistic by an unknown margin | Splitter option ~1 day | Makes test honest; expect test numbers to *drop* |
| G3 | Fix `gen f1_class2` reporting — suppress/flag metrics where support < 20 | DO (trivial) | Stops the misleading 0.000 rows | Hours | |
| G4 | Log per-run split composition table (country × class × label_source) into the run dir | DO (trivial) | Would have caught §2 immediately | Hours | |

---

## 4. Ranked v9 shortlist (what to actually run next)

Ordered by expected-impact ÷ effort, with dependencies:

1. **E4 greedy soup + α-interpolation** over existing v8 checkpoints
   (`scripts/model_soup.py`): ~2–4 hours, no training, +1–3 expected on
   eval/gen. Select on per-country eval macro-F1. *(Do first — it also
   produces the best teacher for #6.)*
2. **C1 context-crop run** (`world_v9_ctx128`): clone `world_v8_ssl4eo` +
   `crop_center_px: 128`. Zero extraction cost, one GPU run.
3. **B1 SoftCon run** (`world_v9_softcon`): the Earth Index backbone;
   reuse the band-mapping machinery. One GPU run. Bundle D5 (surgical-FT
   ablation) as a second cheap run if pod time allows.
4. **A3 label audit** (`scripts/label_audit_oof.py`): 5-fold OOF
   probabilities → cleanlab + AUM rank-fusion → stratified suspects
   parquet for Rachel. CPU only. **Pair with A7**: double-annotate ~300
   eval rows to bound the label ceiling — this decides where the next
   month of effort goes.
5. **F1 footprint probe** (`scripts/footprint_probe.py`): LightGBM on
   geometry features alone → information ceiling; then concat with frozen
   embeddings (B2). CPU only.
6. **G1+G3+G4 eval upgrades**: per-source/per-country eval reporting with
   CIs (Wilson intervals) + composition logging. Makes every subsequent
   comparison mean something — n=98 eval OtherFarm means ±0.08-ish on F1,
   so v8-sized deltas (+0.019) are inside the noise.
7. **B4 AlphaEarth probe** (`scripts/alphaearth_probe.py`): GEE export of
   64-dim annual embeddings for all sites → LightGBM → compare + ensemble.
   Also yields a BGD/NGA-native candidate ranker for Rachel's labeling.
8. **D1 Noisy-Student round 1** (`world_v9_selftrain`): teacher = #1's
   soup; per-class-quota pseudo-labels from the 125k pool; student = fresh
   init, strong augmentation. Guardrails in §5.2.
9. **E1 cascade eval**: score eval/gen with binary-head × type-head
   composition; compare against flat 3-class.
10. **C2 unblock cloudfree** extraction (disk quota), then C3 seasonal
    channels only if B4/AlphaEarth suggests temporal signal pays.
11. **Rachel asks (parallel to all of the above)**: A1 non-US OtherFarm
    labeling (model-assisted lists from #4/#7/#8 outputs), A2 BGD/NGA
    50–200 labels **with class quotas** (force ≥50 OtherFarm via
    high-`prob_class2` + high-uncertainty candidates), A4 hard negatives
    as a by-product, A7 double-annotation session.

Success criteria: judge on **eval macro-F1 with per-source breakdown**
(target: eval OtherFarm-visual F1 ≥ 0.40) and **gen macro-F1 with
support-flagged classes** (target ≥ 0.50 on measurable classes), not the
friendly test split.

---

## 5. Implementation sketches

### 5.1 Label audit (A3)

```
scripts/label_audit_oof.py
  - 5-fold split over train+val+test labeled rows (stratified by class × country)
  - per fold: train the winning recipe 5–8 epochs (or B2 frozen-embedding + LightGBM for speed)
  - collect out-of-fold prob vectors
  - cleanlab confident-learning ranking (or simple: rows where OOF argmax != label AND margin > 0.3)
  - emit suspects.parquet: candidate_id, label, oof_pred, margin, country, label_source
  - stratify Rachel's review list: 100 Poultry↔OtherFarm suspects + 50 NotFarm suspects + 50 registry-match outliers (A6 distance check)
```

### 5.2 Noisy-Student self-training (D1) guardrails

- **Noisy Student, not FixMatch** — on FMoW-WILDS (satellite, geographic
  OOD) Noisy Student gained +3.0 worst-region pts; FixMatch *lost* 2.7.
- Teacher predicts clean (no augmentation/TTA noise); student trains with
  strong augmentation from a **fresh pretrained init** (not resumed from
  the teacher).
- **Per-class quotas, not a global confidence threshold** — a global 0.9
  floor starves OtherFarm of pseudo-labels (the known imbalance failure
  mode). Take top-N per class; cap NotFarm hard; accept lower confidence
  for OtherFarm. Temperature-calibrate on val first.
- Per-country caps too (e.g. ≤ 2× the labeled count of that class×country
  cell) so the pseudo-set can't amplify the US-registry skew — the whole
  point is to *diversify*.
- Soft labels + loss weight 0.5 (or confidence-weighted) on pseudo rows;
  keep eval/gen candidates **out** of the pseudo pool (no leakage).
- Watch OtherFarm *precision* on eval each round — confirmation-bias
  amplification of the Poultry↔OtherFarm confusion is the failure mode.
- One round first; iterate (max 2) only if round 1 moves eval macro-F1
  ≥ +0.02.

### 5.3 Footprint probe (F1)

```
scripts/footprint_probe.py
  - features: num_bldgs, total_area_m2, median_area, template_score_if
    (+ derivable: area_std, largest_bldg_area, bldg_density)
  - LightGBM 3-class, same splits as CNN (reuse splits CSV)
  - report same metric set (test/eval/gen, per-class) for direct comparison
  - phase 2: concat with frozen SSL4EO/SoftCon pooled embeddings → LightGBM again
```

---

## 6. External research findings (2026-07 refresh)

Background research sweep run 2026-07-06 against the seven open problems.
Verdicts are already merged into §3; this section keeps the evidence and
sources so we don't re-litigate later.

### 6.1 Label noise & the eval gap

- **Confident learning** (Northcutt et al., JAIR 2021,
  [arXiv:1911.00068](https://arxiv.org/abs/1911.00068)) is the right tool
  for our shape of noise: model-agnostic, works from out-of-fold
  probabilities, handles *class-conditional* (asymmetric Poultry↔OtherFarm)
  noise. Real-world validation
  ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)): ~51% of flagged
  candidates were human-confirmed errors across 10 benchmarks — an
  efficient review queue.
- **AUM** (Pleiss et al., NeurIPS 2020,
  [arXiv:2001.10528](https://arxiv.org/abs/2001.10528)) is nearly free to
  log during a normal run; benchmark studies
  ([AQuA](https://arxiv.org/pdf/2306.09467),
  [arXiv:2312.02200](https://arxiv.org/pdf/2312.02200)) agree no single
  detector dominates → **rank-fuse two**. A remote-sensing-specific
  assessment exists
  ([arXiv:2603.16835](https://arxiv.org/pdf/2603.16835)) — read before
  finalizing the audit pipeline. Calibrate the model first
  ([arXiv:2511.02738](https://arxiv.org/html/2511.02738v1)).
- **Label ceiling vs feature ceiling**: direct Bayes-error estimation from
  soft labels (Ishida et al., ICLR 2023,
  [arXiv:2202.00395](https://arxiv.org/abs/2202.00395)); the cheap
  decisive experiment is **blind double-annotation of ~300 eval rows** —
  inter-rater F1 upper-bounds measurable model F1 (→ A7).

### 6.2 Semi-supervised learning with the unlabeled pool

- **U-WILDS** ([arXiv:2112.05090](https://arxiv.org/pdf/2112.05090)) is
  the decisive reference — satellite imagery (FMoW), geographic OOD:
  **FixMatch 32.1 vs ERM 34.8 (hurt); Noisy Student 37.8 (+3.0, best
  method tested)**. Curated-benchmark SSL results (e.g. MSMatch's 94.5%
  EuroSAT with 5 labels/class) do not transfer to shifted, imbalanced
  data.
- Class imbalance is the known pseudo-labeling failure mode — fixed global
  thresholds starve rare classes
  ([FlexMatch](https://arxiv.org/abs/2110.08263),
  [FreeMatch](https://openreview.net/forum?id=PDrUPTXJI_A),
  [debiased pseudo-labels](https://arxiv.org/pdf/2201.01490)) → per-class
  quotas/adaptive thresholds (§5.2).
- Optional upgrade: metadata-aware teacher (teacher sees lat/lon+time,
  student doesn't —
  ["Context Matters", arXiv:2404.18583](https://arxiv.org/pdf/2404.18583)).

### 6.3 Backbones (the m-EuroSAT 64 px linear-probe table)

From the Panopticon paper's independent GEO-Bench comparison
([arXiv:2503.10845](https://arxiv.org/html/2503.10845v2), Table 3) —
m-EuroSAT is 64 px S2, our exact regime:

| Model | m-EuroSAT LP | Loadability |
|---|---|---|
| **SoftCon** | **80.0** | HF `wangyi111/softcon` (RN50 + ViT-S/14 + ViT-B/14, 13-band S2) |
| Panopticon ViT-B | 78.4 | github `Panopticon-FM/panopticon` |
| DOFA | 72.0 | torchgeo `DOFABase16_Weights.DOFA_MAE` |
| Galileo | 70.3 | HF `nasaharvest/galileo` |
| CROMA | 70.3 | GitHub |
| AnySat | 64.4 | HF |

- **SoftCon** ([arXiv:2405.20462](https://arxiv.org/pdf/2405.20462),
  [zhu-xlab/softcon](https://github.com/zhu-xlab/softcon)) is the
  successor to our SSL4EO weights, same lab, ResNet-50 with 13-band S2
  input → drop-in via our band-mapping code. Also the backbone behind
  Earth Genome's Earth Index embeddings.
- **PANGAEA** ([arXiv:2412.04204](https://arxiv.org/abs/2412.04204)):
  supervised baselines beat most geo-foundation-models at full label
  budgets; frozen GFMs win only in the ~10%-label regime → keep
  fine-tuning as primary; frozen+GBM for ranking/ensembling.
- **AlphaEarth / Google Satellite Embedding V1**
  ([arXiv:2507.22291](https://arxiv.org/abs/2507.22291); GEE
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`): 10 m, 64-dim, annual,
  multi-sensor-temporal, global. Zero-cost featurization; Google-self-eval
  claims it consistently outperforms alternatives for sparse-label
  mapping. Worth one day.

### 6.4 Few-label country adaptation (when BGD/NGA labels arrive)

Priority order, all evidence-backed:
1. **Oversampled mixing** (weight 5–20×) — the boring baseline wins most
   often (Wiles et al., ICLR 2022,
   [arXiv:2110.11328](https://arxiv.org/abs/2110.11328): across 19
   methods, pretraining + augmentation + a little target data beat clever
   DG consistently).
2. **LP-FT** ([arXiv:2202.10054](https://arxiv.org/abs/2202.10054)): full
   fine-tuning at tiny budgets *distorts* features (−7 pts OOD avg vs
   linear probe); LP-FT gets +10 OOD / +1 ID vs full FT. Our cRT win is
   the LP half of this story.
3. **WiSE-FT** ([arXiv:2109.01903](https://arxiv.org/abs/2109.01903)):
   α-interpolate pre/post-adaptation weights; headline gains are
   CLIP-specific, expect +0.5–2 here.
4. **Surgical FT** ([arXiv:2210.11466](https://arxiv.org/abs/2210.11466)):
   output-level shift → tune layer4+head only. Consistent with our
   AdaBN-negative/cRT-positive evidence.
5. **Class-quota'd labeling request** — 4 OtherFarm labels in BGD/NGA
   makes everything unmeasurable; force ≥50 OtherFarm in the ask.

### 6.5 Fusion, temporal, soups

- **Fusion**: late fusion (GBM over CNN-embedding ⊕ tabular) is best
  practice at our sample size (medical-imaging systematic review,
  [npj Digital Medicine 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7567861/);
  [AutoGluon bag-of-tricks](https://arxiv.org/pdf/2412.16243)). FiLM
  conditioning: no agricultural precedent, unjustified at 15k rows.
  CAFO-morphology precedent is strong:
  [Handan-Nader & Ho 2019](https://www.nature.com/articles/s41893-019-0246-x)
  (93% poultry-CAFO detection from footprint morphology),
  [Robinson et al. 2022](https://arxiv.org/abs/2112.10988) (barn-shape
  filtering), Stanford RegLab CAFO work.
- **Temporal**: multi-temporal gains are proven for crop phenology, thin
  for static structures. Cheapest meaningful test: wet/dry composites +
  per-band temporal std. Pixel-time-series models (Presto, Galileo) are a
  poor fit for 64 px building scenes. AlphaEarth already integrates the
  annual series.
- **Soups**: greedy soup over fine-tunes from a shared init
  ([arXiv:2203.05482](https://arxiv.org/abs/2203.05482),
  [mlfoundations/model-soups](https://github.com/mlfoundations/model-soups));
  soups ≥ output ensembles OOD, replicated on FMoW-WILDS. SWA is the
  one-run version. Constraint: same pretrained init only.

### 6.6 Hype flags (things that look good and aren't)

- FixMatch-style consistency SSL on geographic shift: **actively harmful**
  in the only directly relevant benchmark.
- Clay: strong comms, no small-patch classification evidence.
  Prithvi-EO-2.0: wrong resolution/bands (30 m HLS). SatMAE++/AnySat:
  mid-table everywhere.
- WiSE-FT's +8.7 pp headline: CLIP-zero-shot-specific.
- FiLM/DANN/CORAL-style cleverness at 15k rows: consistently beaten by
  boring late fusion / oversampling.
- Backbone-swapping past SoftCon has diminishing returns (PANGAEA) — the
  remaining gap is labels + morphology, not representation.
