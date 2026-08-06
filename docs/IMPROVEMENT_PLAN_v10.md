# Improvement plan v10 — bottlenecks + sequenced plan (post-v9)

Written 2026-07-09. Produced by a 10-agent analysis pass: 3 evidence agents
(results distillation, hands-on analysis of the scored parquets +
`rachel_labels_v5.parquet`, code audit of `training/`), 4 literature agents
(CAFO detection SOTA, geospatial foundation models, label-efficiency/OOD
methods, input modalities + eval practice), and 3 adversarial verifiers
(contradiction vs. our own results ledger, statistical measurability,
implementation feasibility). Full verifier tables + per-agent evidence
digests are in the appendices.

Relationship to [`IMPROVEMENT_ROADMAP.md`](IMPROVEMENT_ROADMAP.md): this
supersedes its prioritization. The roadmap's lever inventory (A1–F2, G)
remains the reference taxonomy; Appendix D's e1 digest maps every roadmap
item to STRENGTHENED / WEAKENED / OBSOLETED given the v9 results, the
validity audit (`notebooks/model_evaluation_analysis.ipynb`), and the v5
label drop.

---

## TL;DR

1. **The biggest bottleneck is measurement, not modeling.** At eval n=524
   the 95% CI is ±0.05 macro-F1 — nothing smaller than +0.05 is detectable,
   and almost every lever worth trying promises +0.01–0.04. Fix the
   instrument first; it is mostly S-effort.
2. **A free blind OOD benchmark already exists and breaks the v9 tie**:
   3,297 labeled sites outside the training whitelist were predicted blind
   by both models. On them **ctx128 beats softcon +0.047 macro-F1
   (0.509 vs 0.462, n=3,297)** — the wider-context model is genuinely
   better out-of-distribution; the eval-slice tie stands.
3. **v5 label reality-check**: the scored data's labels are already
   v5-consistent (17,253 overlaps, perfectly diagonal — nothing there was
   predicted blind). The genuinely new v5 labels are **14,138 clusters
   with no patches at all** (RUS 2,535, DEU 1,101, ARG 889, AUS 781,
   ITA 701…). Getting them costs one patch-extraction + inference cycle
   and is worth it twice: a ±0.010-precision benchmark *and* ~9k
   European/Russian hard negatives targeting the worst failure mode.
4. **The deployed world map is unusable in whole regions**: blind-slice
   binary farm F1 ≈ 0.01 in Russia (n=438), 0.0 in Ukraine, 0.04 in
   Malaysia; the model calls "farm" on roughly half of confirmed non-farms
   outside training countries. 44% of eval errors are NotFarm→farm FPs.
5. **Stop expecting 3-class typing from 10 m pixels.** Nobody in the
   literature discriminates poultry/pig/cattle from 10 m imagery — every
   published pipeline types via footprint geometry, registries, or
   sub-meter imagery, and human annotators with all of those still fail
   18.6% of typing calls. Reframe: the 10 m CNN does *detection* (binary
   F1 0.855 eval — parity with Earth Index's in-region 0.876), a fusion
   stage does *typing*.

---

## New evidence produced by this analysis

**Error anatomy (dedup eval, n=524).** softcon's 210 errors: 77
NotFarm→farm FPs + 88 Poultry↔OtherFarm swaps + 23 farm→NotFarm misses +
22 other. So 44% of errors are farm over-prediction and 42% are type
confusion — two distinct failure modes needing two distinct fixes (hard
negatives; footprint/registry typing). Both-wrong floor: 30% of eval rows
defeat both models.

**Ensemble headroom.** softcon and ctx128 disagree on 24.8% of eval
(oracle-union accuracy 0.700 vs 0.599 single-model) and 24.2% of gen,
where disagreements resolve 2:1 in ctx128's favor. Probability-averaging
is free (scored parquets exist) and the diversity is real.

**Pseudo-label yield.** On the ~115k unlabeled pool: both models agree
with score ≥0.9 on **14,234 sites** (11,275 Poultry). Precision proxy on
held-out labeled rows under the same filter: 0.81–0.89. Confident poultry
concentrations in never-trained countries: IND 1,063, IDN 403, PHL 251,
PAK 234. Also: **BGD has zero scored unlabeled candidates** — the flagship
deployment country cannot be inferenced today.

**v5 label facts.** 32,050 final labels → 29,477 map to 3-class (2,573 in
dropped categories). BGD/NGA get 182/119 labels but only **4 OtherFarm
total — gen f1_class2 stays unmeasurable under v5**; only targeted
acquisition fixes it. 753 labels were revoked v4→v5 (498 Australia
poultry — treat AUS labels as unstable). New all-Europe label pools
(DEU/ITA/FRA/BLR/CZE… each 100–1,100) are ~99% exactly the hard-negative
class the model over-predicts against.

---

## The five bottlenecks

| # | Bottleneck | Evidence |
|---|---|---|
| 1 | **Benchmark resolution & contamination** | n=524, CI ±0.05; duplicates inflate eval +0.03 and bias checkpoint selection (val duplicated too); 27.1% of eval within one patch-width of train; gen OtherFarm n=4; top-6 models tied. Literature: random splits inflate RS-CNN metrics 10–28% (Kattenborn 2022, Ploton 2020) |
| 2 | **Train/deployment label mismatch** | Train OtherFarm 99% registry / 56% US vs eval 71% visual / 97% non-US; eval NotFarm recall 0.319; blind-slice farm precision 0.61. Training uses 10.1k of 29.5k available 3-class labels |
| 3 | **Regional OOD collapse** | RUS/UKR/MYS binary farm F1 0.00–0.04; IND/TUR/VNM macro 0.25–0.45. Same failure mode as Earth Index's Arkansas precision collapse (87→68%), which ~242 triage labels fixed |
| 4 | **Type discrimination is information-limited at 10 m** | 42% of errors are Poultry↔OtherFarm; zero published 10 m typing baselines; the field types via footprints (+52 pts precision from geometry rules alone — Robinson 2022) or registries |
| 5 | **Input ceiling: one annual median** | Google building-detection ablation: +2.8 pp from 4 frames, +5 pp from 32 — signal the median composite destroys; no S1 double-bounce channel; no building-footprint prior despite Open Buildings 2.5D Temporal (4 m presence/count/height, GEE-ready, covers BGD/NGA/THA/LatAm) |

The representation axis is **closed**: SoftCon already tops the closest
public benchmark (m-EuroSAT 64px LP: 84.3 vs Panopticon 83.9, CROMA 79.4,
DOFA 78.8); every 2025–26 rival differs by ≤1–2 pts on saturated
benchmarks; two 2026 meta-studies (arXiv 2605.12678, 2606.13896) find GFM
rankings statistically unstable and adaptation choices more impactful than
encoder choice — externally confirming the internal top-6 tie.

---

## The plan

### Phase 0 — this week, no GPU (fix the instrument, harvest free wins)

| Action | Why / expected effect | Effort |
|---|---|---|
| Adopt the 3,297-row blind slice as a standing OOD benchmark (extend `scripts/post_hoc_evaluate.py`); freeze it | CI ±0.020 today; makes Δ≥0.03 levers decidable | S |
| Dedup `candidate_id` in `build_splits` (`training/dataset.py:568-589`) + `training/inference.py:199-268` | Removes the +0.03 eval inflation and the val-duplication checkpoint bias | S |
| Reporting fixes: bootstrap CIs + class-support flags everywhere; report gen as 2-class + Poultry F1; headline = binary detection + separate typing metric | Prevents the next cRT-sign-flip misread; gen macro with support-4 class is noise | S |
| Prob-ensemble softcon+ctx128 (average `prob_class*` offline) | Disagreements resolve 2:1 to ctx128 on gen; expect gen gain; measure on blind slice | S |
| Per-country prior/threshold correction (extend `scripts/logit_adjust_sweep.py`; note `inference.threshold` is dead code for 3-class) | RUS binary F1 0.01 → ~0.3+ is detectable at n=438; global τ=0 is exhausted but per-country is untried | S |
| Footprint stacker v0: attach `num_bldgs`/`total_area_m2`/`median_area`/`template_score_if` (already in candidate CSVs, dropped at `inference.py:118-127`) to scored parquet; fit GBM on [probs + feats] | Cheap falsification test of the footprint hypothesis before M-effort investment; targets the 44% FP error mass | S |
| Cleanlab-style audit of registry-sourced Poultry↔OtherFarm labels (per-row probs already exist) | Surfaces mislabeled rows feeding the 42% confusion mass | S |
| Hygiene: `recompute_indices: true` (indices currently inconsistent under per-band jitter), label-default −1 assert (`dataset.py:117-119`), TTA on in configs (`inference.py:64-91` already implements it) | Correctness; TTA (+0.005–0.02) measurable only on the bigger benchmark | S |

### Phase 1 — next 2–3 weeks (the big data move)

1. **Extract + score the 14,138 unscored v5 clusters** (candidates →
   patches → inference with both v9 checkpoints). Yields a ~12.2k-row
   3-class benchmark (pooled ±0.010 — unlocks the entire +0.01–0.02 lever
   class). **⚠ Freeze a held-out benchmark partition before any retrain,
   or the retrain burns the instrument.**
2. **Rebuild the master on v5 + retrain (`world_v10`)**: ~25k train rows
   including the 3,297 ex-whitelist rows (RUS/UKR/POL hard negatives) and
   the new European NotFarm mass; dedup + **spatial-block splits**
   (H3/geohash cell assignment in `build_splits`; block ≥ ~10–25 km given
   median nearest-train 5.53 km) + honest checkpoint selection, all landed
   *inside* this rebuild so nothing needs retraining twice. Keep SoftCon;
   carry the 128px-crop question as the one controlled comparison (blind
   slice already favors ctx128 OOD). **The world_v10 config should also
   adopt the hierarchical two-head architecture from the Farm-type
   recognition track below** (detection head + type head with
   partial-label loss) — same retrain window, no extra GPU cycle.
3. **Generate BGD candidates** (pool is zero — deployment prerequisite).

Expected: the NotFarm-recall failure (0.319) and the regional collapse
respond directly to ~10k new hard negatives; largest single expected gain
of any lever, and after Phase 0/1 it is actually measurable.

### Phase 2 — gated on Phase 1 benchmark results (structure & fusion)

- **Open Buildings 2.5D Temporal input channels** (presence/count/height
  at ~4 m, GEE `GOOGLE/Research/open-buildings-temporal_v1`) — injects
  exactly the morphology the AdaBN-negative result says is missing, in
  exactly the OOD countries. Prereq: per-source scaling fix
  (`dataset.py:137-140` hardcodes /10000 + [0,1] clamp) + labeled-only
  re-extraction. USA/Europe gap → Microsoft/Overture fallback +
  availability-mask channel. Effort M–L; highest-ceiling input lever.
- **Footprint-geometry fusion v1** (if the v0 stacker showed signal): OB
  v3/Overture polygons → elongation, area, parallel-row-alignment features
  → late-fusion head. The literature's single biggest precision lever
  (+52 pts for −3.3 recall, Robinson 2022; >50% FP removal, Tulbure 2024).
  Also the credible path to poultry-vs-pig typing. M.
- **Surgical last-block fine-tune on 300–500 visual-source non-US labels
  + WiSE-FT interpolation guard** (same-init only) — the
  literature-prescribed fix for output-level shift, which is what we have
  (arXiv 2210.11466, 2202.10054). S–M.
- **Multi-frame S2 (4–8 cloud-free frames)** instead of annual median
  (+2.8–5 pp evidence, arXiv 2310.11622) and/or **S1 VV/VH**
  (+0.02–0.05, double-bounce on metal barns) — both need the scaling/date
  plumbing (`patch_extraction.py:211-232`); M each. One at a time.
- **AlphaEarth embedding concat probe** (GEE export, 64-dim, no fine-tune
  possible) — S/M, judge on benchmark; caution: spatial smoothing may
  hurt barn-scale detail.

### Phase 3 — gated, larger bets

- **Noisy Student self-training** on the ~103k-row pool with **per-class
  quotas** (never global threshold, never FixMatch — FMoW-WILDS: Noisy
  Student +4.1 worst-region, FixMatch −1.6 vs ERM; arXiv 2112.05090).
  Only after the v10 teacher exists. L.
- **Earth-Index-style per-country human-in-loop head** (frozen embeddings
  + minutes-retraining MLP + ~250–500 triage labels per failing country) —
  the proven fix for Arkansas-style regional collapse. M + ongoing
  labeling.
- **SEALS-style rare-class active learning for OtherFarm in BGD/NGA/PER**
  — the only thing that ever makes gen OtherFarm measurable (v5 does
  not). M.
- **Per-country/source temperature scaling + conformal deferral tiers**
  for triage, with PR-per-review-budget curves (Stanford: 95% of CAFOs at
  <10% review effort; RegLab field data: expect ~35% ground-confirm at
  high confidence and design for it). S.

### Kill list (evidence-closed)

Backbone swaps (TerraMind/Galileo/Panopticon/Clay/Copernicus-FM),
AdaBN/MixStyle/pixel-statistics tricks, global logit-adjust variants,
cRT-on-softcon, FixMatch-style online pseudo-labeling, explicit
super-resolution, weight-soups across the softcon/ssl4eo init boundary,
TESSERA (repriced L — needs a time-series pipeline the repo lacks).

### Standing measurement rules

- Any gen-slice delta <0.07 is noise until the v5 benchmark exists.
- One lever at a time — `softcon_ctx128`'s failure-to-stack is the
  cautionary tale.
- Rachel's eval sample is the *design-based* headline estimator
  (Wadoux 2021); spatial-block CV is for model/checkpoint selection only.
- Headline = binary detection F1 + review-budget PR curves; typing is a
  separate fusion product scored separately.
- Detection-power arithmetic: n ≈ 524·(0.05/Δ)² → Δ=0.03 needs n≈1,455;
  Δ=0.02 needs n≈3,275; Δ=0.01 needs n≈13,100. Paired same-row bootstrap
  roughly halves these.

---

# Farm-type recognition track (added 2026-07-10)

Produced by a second 5-agent pass (data decomposition on the scored
parquets + v5 labels, typing-signal physics literature, imagery-market
pricing, architecture methods, adversarial verifier). Verdict table in
Appendix F; evidence digests in Appendix G. Goal: a model that actually
recognizes farm type and maximizes user-visible information per farm.

## What the data says about type (new empirical results)

1. **The Poultry↔OtherFarm confusion is overwhelmingly pigs-defaulting-to-
   Poultry, and it is an imbalance artifact, not a representation gap.**
   True pigs are predicted Poultry 64–65% of the time on eval (n=63) and
   72–75% on the blind slice (n=68); only 25–29% are typed correctly. Yet
   ranking pigs-vs-poultry by the model's own `prob_class2` gives
   **AUC 0.787** — the signal is already inside the network; argmax under
   ~5:1 poultry:pig train imbalance hands pigs to the majority class.
   Farms of any subtype are rarely called NotFarm (≤11%): **detection
   works; typing fails, and it fails toward Poultry.**
2. **Pigs are a viable named class today; Cattle is not.** v5 supports:
   Pigs 2,576 labels (train-split rows with patches: 1,384 train / 290
   test / 63 eval). Cattle: **231 labels globally** (MEX 107, USA 72,
   everything else ≤8) → 130 train / 26 test / 7 eval — unlearnable and
   unmeasurable (per-class F1 CI at n=26 ≈ ±0.2). Cattle is a
   data-acquisition problem, not a modeling one; the 2026 CONUS feedlot
   dataset (11,746 labeled + >24k detected facilities, NAIP) is a
   ready-made label source no internal doc had connected.
3. **The 3-class taxonomy throws away 2,397 confirmed farms** because
   their type is unknown (Farm: Unknown 2,171, Mixed 103, PigsOrPoultry
   102, Other 21) — free detection supervision, geographically
   concentrated exactly where OOD detection fails (ARG 1,148, ZAF 292,
   AUS 253). PigsOrPoultry is even a *candidate-set* type label.
4. **False-positive anatomy (blind NotFarm, n=1,632):** bulk FP mass is
   residential (198 rows @ 37% FP) and garages (116 @ 36%); worst
   per-capita confusers are **greenhouses (77% FP), solar (68%), railway
   (90%)** — long thin bright roofs, i.e. the poultry-barn template.
   Published cheap screens exist for all three (PGHI greenhouse index
   ~90% OA on S2; global PV masks; S1 coherence). And **only 34% of
   existing NotFarm labels even have patches** — ~9k owned hard negatives
   are sitting undownloaded.
5. **Footprint features carry typing signal but are redundant in-domain:**
   leave-one-country-out GBM on the 4 already-extracted footprint scalars
   gives pig-vs-poultry AUC 0.75 alone — but fusing them with CNN probs
   adds only **+0.004** over the CNN's 0.787. Their value is OOD
   FP-filtering (where Open Buildings covers) and capacity estimation,
   not in-domain typing. Richer *shape* features (barn length,
   elongation, height from OB 2.5D — importance 0.74 in PRISM-CAFO) are
   untested → gate.

## The physics of typing (what is knowable at which resolution)

Separability ordering at 10 m: **feedlot > swine > poultry > dairy.**
Swine lagoons are solidly multi-pixel at 10 m (median 5,200 m² ≈ 52 S2
px; pink/brown anaerobic water; Landsat-30m detection with 94% ±1 yr
construction dating). Feedlots are tens-of-ha bare-soil features —
the most 10m-compatible type (but PRISM-CAFO's *spatial-split* beef F1
of 0.16 at 0.6 m warns that feedlot models generalize poorly across
regions). Individual barn geometry (the poultry/pig discriminator)
needs ~3 m; feed bins/fans need sub-meter. The honest ceiling: even at
0.6 m with 38k patches, spatial-split type F1 is swine 0.87 / poultry
0.76 / dairy 0.58 / beef 0.16 (PRISM-CAFO 2026) — our 42%
type-confusion at 10 m is physics-consistent, and dairy/beef stay hard
at any resolution. Skip list (unanimous, evidence-blocked): satellite
thermal (barns sub-pixel at 70–100 m), hyperspectral CH4/NH3 (detection
limits far above farm scale), SAR tasking for typing, basemap scraping
(ToS).

## Architecture: hierarchical two-head (all agents converge on this)

Replace the flat 3-class softmax with:

- `head_det`: Linear(2048,1) + BCE — farm vs not-farm, trained on **all**
  labels including the 2,397 type-unknown farms and patch-ready blind
  rows.
- `head_type`: Linear(2048,K) softmax over {Poultry, Pig} (+Cattle only
  when ≥1k labels exist) — logit-adjusted CE on typed rows;
  **marginal-likelihood loss** −log Σ_{y∈S} p(y) for candidate-set rows
  (PigsOrPoultry → S={Poultry,Pig}); Unknown/Mixed masked from the type
  loss entirely. Loss `L = L_det + λ·L_type`, λ ∈ {0.5,1,2}, head LR 10×
  backbone. Inference: `p(farm)`, then type if `p(farm)>τ` else **"farm,
  type-unknown"** — an output the current pipeline cannot express (it
  silently drops those farms).

Why this beats the alternatives: flat 4/5-class softmax couples
tail-class noise into the 0.855 detection boundary (worst variant —
SKIP); a fully separate second CNN doubles inference and loses shared
features; special multi-task optimizers (PCGrad/GradNorm) are
unnecessary at 2 nested tasks (unitary scalarization matches them —
NeurIPS 2022). Nested taxonomies are the *best* case for multi-task
sharing. v3's 7-class failure precludes none of this (pre-bugfix,
pre-SSL, flat softmax); its one durable lesson: **don't split poultry
subtypes** (6,191 of 9,185 poultry labels are "Unspecified" — Meat/Eggs
are incomplete annotations, not visual classes).

**Mandatory evaluation protocol:** (i) detection PR-AUC + F1 vs the
0.855 guard (must stay within CI, else freeze det head / WiSE-FT back);
(ii) type macro-F1 + confusion **conditioned on true farms** (eval
end-to-end macro-F1 at n=524 cannot see type-head progress; test pigs
n=290 gives CI ~±0.06); (iii) collapsed 3-class macro-F1 for continuity.

## Imagery: free covers most of it; buy nothing before two gates

| Tier | What | Cost | Covers |
|---|---|---|---|
| Free ortho | NAIP 30–60 cm (US); EU national orthophotos 8–25 cm (NLD 8, ESP 25/10, FRA 20, DEU 20, POL, CZE, DNK; ITA is the gap) — WMS chip-fetch at label points | $0 | USA + most new v5 Europe labels |
| Free 3D | Open Buildings 2.5D Temporal (4 m presence/count/**height**, annual 2016–2023, GEE) | $0 | BGD/NGA/THA/BRA/MEX/PER — exactly the no-ortho countries |
| Free quota | **ESA Earthnet TPM proposal** (Pléiades/Neo/WorldView/SkySat archive, <2 wk review, ~2–3 days to write) — submit early regardless of gates; **Planet R&E** (3,000 km²/mo PlanetScope free) settles "does 3 m resolve barn geometry?" for $0 | $0 | Global South chips |
| Paid archive (list, quote-required) | Satellogic $4/km² · SkySat/Jilin $6 · Pléiades $12.5 (**academic −50%**) · Neo/Maxar-30cm $23–26 | see below | residual sites |

Chip arithmetic (0.41 km² typing chip/site, clustered AOIs): 1k sites ≈
$1.6–5k; 10k ≈ $16–51k at budget tiers. **Min-order effects are decisive
for isolated points** ($64–375/site → 10k isolated = $0.6–3.8M): buy
county-scale AOIs over clustered detections (farms cluster; overhead
~1.5–3×, verifiable from candidate coordinates before quoting).
NICFI is dead (confirmed; successor is 5 m paid — too coarse anyway).
Sequence: free ortho + OB 2.5D → ESA TPM + Planet R&E pilots → $5k
Satellogic/SkySat pilot (gated on the shape-feature probe AND the eval
expansion) → scale buy only if the pilot wins. Licensing note: EULAs
generally allow publishing *derived* labels/outputs, not chips;
Satellogic charges a 25–200% public-release uplift.

## Beyond type: attributes that make the map maximally useful

All near-zero marginal cost, none needs the macro-F1 instrument:

- **Capacity** = footprint area × published stocking density (EWG mapped
  357M birds in NC this way; Robinson's US barn dataset is public).
  Validate against US registry capacities, not eval F1.
- **Construction year / expansion alerts** — OB 2.5D annual stacks +
  Landsat/S2 changepoint (94% accuracy ±1 yr on lagoons; Temporal
  Cluster Matching dates barns without per-year labels).
- **Active/inactive flag** — temporal persistence signals.
- **Aquaculture** — pond grids are trivially mappable at S2 (published
  OA 0.83–0.95); a cheap adjacent class if "farms" is read broadly.

No published global-South farm map carries type + capacity + age — this
is where the project can be uniquely useful rather than marginally
better at one F1 number.

## Type-track sequencing (verdicts in Appendix F)

**Now (hours–days, $0):**
1. **Stage-0 frozen-embedding type probe** — fit a {Poultry,Pig} linear
   head on frozen softcon embeddings, farm rows only; extend with OB 2.5D
   shape features (length/elongation/height). This is the ceiling
   estimate that gates everything below. (DO FIRST)
2. **Pig-recovery recalibration** on the existing model — threshold/prior
   moves only (cRT-on-softcon already failed in the ledger); AUC 0.787 is
   real headroom, expect pig precision ~0.35–0.45 at recall 0.6. (DO)
3. **Submit the ESA TPM proposal**; start the Planet R&E application.
   (DO — long latency, start regardless)
4. **Capacity v0** from existing footprint features. (DO)

**With the Phase-1 world_v10 retrain (same config window):**
5. **Hierarchical two-head + partial-label loss** — the type track's
   centerpiece; merges into the v10 rebuild so nothing retrains twice.
   (DO)
6. **Hard-negative program**: download patches for the ~9k patch-less
   NotFarm labels; mine OSM greenhouse/solar/garage/residential polygons
   as targeted negatives. Attacks the single largest error mass
   (NotFarm recall 0.319). (DO)

**Gated:**
7. Footprint/OB-2.5D fusion or aux regression head — gate on probe #1
   (coarse scalars measured at only +0.004; shape features untested).
8. Lagoon channel/aux head — gate on probe residuals (the CNN's pig
   signal may already *be* the lagoon).
9. $5k VHR pilot → scale buy — gate on #7 + Planet 3 m outcome + eval
   expansion.
10. Cattle class — gate on ≥1k labels (CONUS feedlot dataset import or
    registry fusion); until then Cattle stays inside "farm,
    type-unknown" rather than polluting a softmax.

**Skip:** flat 4/5-class softmax retrain, poultry subtype splitting, OvR
multi-label heads, thermal/hyperspectral/SAR-tasking for typing, basemap
scraping, isolated-point VHR purchases.

---

# Appendices — full evidence from the 10-agent analysis

Agent reports were produced 2026-07-09 against: repo docs
(`EXPERIMENTS_LOG.md`, `EXPERIMENTS_v8.md`, `IMPROVEMENT_ROADMAP.md`,
`EVAL_FRAMEWORK.md`), the full-world scored parquets for
`world_v9_softcon` / `world_v9_ctx128` (141,954 rows, 133,026 unique
candidates), `rachel_labels_v5.parquet` (157,102 clusters / 32,050 final
labels), the `training/` source tree, and 2024–2026 web literature.

## Appendix A — Lever audit (contradiction verifier)

Every proposed lever audited against the results ledger. Verdicts: NEW-VIABLE / ALREADY-TRIED / CONTRADICTED-BY-EVIDENCE / NEEDS-PRECONDITION.

### Lever audit

| # | Lever (reports) | Verdict | Evidence / flag |
|---|---|---|---|
| 1 | Score/benchmark v5 labels before retrain (e1,e3,l2,l3,l4 — all "S, zero GPU") | NEEDS-PRECONDITION(patch-gen + inference for 14,138 unscored v5 clusters) | **Report conflict, e2 wins — verified**: web points.csv has 133,026 rows / 18,006 labeled; scored labels already v5-diagonal, so no new-label rows were scored blind. Free part today = only the 3,297-row ex-whitelist slice (verified: 1,717/1,474/106 by class; POL 726, AUS 696, RUS 438) |
| 2 | Retrain on full v5 (~25k train rows, ex-whitelist + Europe/RUS negatives) (e1,e2,l2) | NEW-VIABLE, top lever | Directly attacks label-source shift + NotFarm recall 0.319; precondition shares #1 patch-gen; must hold out a v5 benchmark slice first or it's burned; AUS unstable (498 labels revoked v4→v5) |
| 3 | Hard-negative mining from 3,297 blind rows (e1) | NEW-VIABLE (folds into #2) | RUS 437/438 NotFarm = ready-made hard negatives |
| 4 | v4→v5 churn audit as label-noise ceiling (e1) | CONTRADICTED-BY-EVIDENCE(e2 crosstab: 17,253 overlaps perfectly diagonal; churn = 753 revoked rows, all split=unlabeled, AUS 498) | No inter-release disagreement exists on train/eval rows; shrink to an AUS-revocation check |
| 5 | Per-country priors/thresholds (e1,e3) | NEW-VIABLE (deployment patch only) | Global logit-adjust exhausted (τ=0) but per-country untried; RUS/UKR prior mismatch real; fit risks overfitting n<100/country |
| 6 | Prob-ensemble softcon+ctx128 (e2,e3) | NEW-VIABLE | **Conflict**: e1 kills soups (E4, init-incompatible) — valid ONLY against weight-averaging across softcon/ssl4eo inits; prob-averaging unaffected. Oracle +0.10 is headroom not expectation; eval disagreements split 53/53 → expect gen gain (ctx128 2:1), little eval gain |
| 7 | TTA on (e3) | NEW-VIABLE S | Code exists; +0.005–0.02 is undetectable at n=524 — measure on v5 benchmark |
| 8 | Dedup candidate_id in splits/inference (e3) | NEW-VIABLE S, consensus | Removes +0.03 inflation + checkpoint bias |
| 9 | Spatial-block splits (e1,e3,l4) | NEW-VIABLE M, consensus | 27.1% proximity; do inside v5 rebuild; keep Rachel sample as design-based headline (l4/Wadoux) |
| 10 | Reporting fixes: dedup+CI+support flags; gen as 2-class + poultry-F1 (e1,e2) | NEW-VIABLE S | gen class-2 support stays 4 in v5 (e2 verified) |
| 11 | Footprint geometry — stacker on existing feats (e3-S); OB-rules 2nd stage (l1); aux-feature fusion (l4); OB 2.5D Temporal channels (l4) | NEW-VIABLE, strongest 4-report convergence | Literature +52pts precision (Robinson). Caveats: OB lacks USA/Europe (needs mask/fallback); candidates were already template-filtered on footprints → e3's S-effort stacker on `num_bldgs/template_score_if` is the cheap falsification test before M-effort variants |
| 12 | Multi-frame / seasonal S2 stack (e3,l4) | NEEDS-PRECONDITION(plumbing: single global date_range, e3 B5; + v5 benchmark to detect +0.02–0.04) | Google 1→4 frames +2.8 mIoU is solid; re-extraction cost |
| 13 | S1 fusion (l4 "S", e3 "blocked") | NEEDS-PRECONDITION(dataset scaling: /10000 + [0,1] clamp annihilates dB — e3 B4) | **Effort conflict: e3 right, min M**. Not radiometric-contradicted (pitched as structure) |
| 14 | AEF / TESSERA embedding-concat probe (l2,e1) | NEW-VIABLE S | Never run (B4); embeddings-only; smoothing may hurt barn scale; judge on v5 benchmark |
| 15 | Any backbone swap (TerraMind/Galileo/Panopticon/Copernicus-FM/Clay) | CONTRADICTED-BY-EVIDENCE(own plateau eval≈0.50, top-6 tie CI 0.10; SoftCon tops m-EuroSAT) | All reports concur: closed |
| 16 | Noisy-Student on 115k pool, per-class quota (l3,e2; e1/e3 gate it) | NEEDS-PRECONDITION(v5 retrain + benchmark first; BGD pool=0) | Pseudo-label precision proxy 0.805 on eval ≈ model's own bias — quota per CReST mandatory |
| 17 | FixMatch/online pseudo-labeling | CONTRADICTED-BY-EVIDENCE(FMoW-WILDS 32.1 < ERM 33.7) | Consensus anti-lever |
| 18 | LP-FT / last-block surgical FT on 300–500 visual non-US labels (l3) | NEW-VIABLE S, with flag | l3 cites cRT +0.019 as support — cherry-picked: cRT-on-softcon was −0.020; both inside gen noise. Mechanism differs from cRT (target-distribution labels), so not falsified — but expect ambiguity until #1 |
| 19 | WiSE-FT interpolation (l3) | NEW-VIABLE S, same-init pairs only | e1's init constraint applies to weight interpolation too |
| 20 | Earth-Index human-in-loop per-country head (l1,l2) | NEW-VIABLE M | Matches adaptation-shaped OOD (AdaBN-negative, Arkansas 87→68→restored) |
| 21 | SEALS/rare-class AL for OtherFarm in BGD/NGA/PER (l3) | NEW-VIABLE M | Only fix for gen class-2 n=4; PER has 29 v5 labels despite 1,953 scored clusters |
| 22 | Cleanlab/AUM label audit (l3) | NEW-VIABLE S | Probs already exist per row |
| 23 | Per-source temperature scaling + conformal deferral (l3) | NEW-VIABLE M | Calibration ≠ exhausted logit adjustment; re-fit under prevalence shift |
| 24 | Binary-head + separate typing stage; binary-first reporting (e2,l1) | NEW-VIABLE S | 42% of errors Poultry↔Other; 10m typing has no published baseline |
| 25 | Registry/covariate fusion head (l1) | NEEDS-PRECONDITION(covariate coverage in BGD/NGA/PER; registry rows already score worse, 0.413) | Risks amplifying label-source shift |
| 26 | BGD candidate generation (e2) | NEW-VIABLE S/M | Deployment prerequisite; pool=0 verified claim |
| 27 | AdaBN/MixStyle/global logit-adjust variants; cRT-on-softcon; explicit SR | ALREADY-TRIED(negative) / CONTRADICTED | Ledger + l4 SR evidence; keep dead |
| 28 | Hygiene: recompute_indices, BN-freeze eval, label-default −1 (e3) | NEW-VIABLE S | BN-freeze ablation worthwhile since pretraining is the only robust win |

### Notes (verification + conflicts)

**Adjudicated e1-vs-e2 "blind benchmark" contradiction by direct inspection.** `/home/filip/code/farm-mapping/web/public/data/world_v9_softcon/points.csv`: 133,026 rows, 18,006 3-class labeled, 3,297 labeled-with-split=unlabeled (class 1717/1474/106; POL 726, AUS 696, RUS 438, UKR 145, VNM 101, MYS 95, TUR 85). Both reports' counts are correct on this file; the project-context "1,072" is stale (it matches only the old 15,781-row cache at `notebooks/results_cache/data/output/world_v9_softcon/scored_candidates.parquet`). e2's decisive addition stands: the 14,138 genuinely-new v5 clusters are absent from scored data, so every "score v5 at S effort / zero GPU" claim (e1, e3, l2, l3, l4) is wrong as stated — the honest benchmark costs patch extraction + inference (M) plus an eval/train partition decision before any retrain.

**Caveat on e1's ctx128>softcon blind ranking (+0.047, n=3,297):** composition is not deployment-like (RUS 438 ≈ all-NotFarm, POL/AUS dominate); pooled macro-F1 is prior-skewed. Direction is corroborated by gen disagreements resolving 2:1 to ctx128 (e2), but treat magnitude as unsettled until the full v5 benchmark.

**Soup conflict:** e3's "soup over top-6" is invalid across the softcon/ssl4eo init boundary (e1 correct); restrict weight-averaging to same-init families; prob-ensembles (e2) are unaffected and cheapest.

**Stacking discipline:** softcon_ctx128 (worse on eval AND gen) is the ledger's warning that levers here don't compose. Reports 2, 11, 12, 13, 16 each propose multi-change bundles; require one-lever-at-a-time measurement on the v5 benchmark, whose CI (~0.02–0.03 at n≈12–15k) is the only instrument that can detect the +0.02–0.05 effects most levers promise.

**Priority-consistent ordering implied by verdicts:** (a) #8/#10 hygiene → (b) #1 patch-gen/score v5 + freeze benchmark split → (c) #2 retrain, #11 stacker probe, #6 ensemble, #7 TTA measured against it → (d) gated M/L levers (#12, #13, #16, #20, #21).

## Appendix B — Measurability audit (detection-power verifier)

| Lever (source reports) | Expected Δ | Verdict | Detectability arithmetic |
|---|---|---|---|
| Adopt existing 3,297-row non-whitelist blind slice as OOD benchmark (e1) | — (instrument) | MEASURABLE-NOW | exists, scored; CI half-width ≈ ±0.05·√(524/3297) ≈ ±0.020 |
| Score 14,138 unscored v5 clusters → ~12.2k new 3-class eval (e2; premise of e1/e3/l2/l3/l4 "rescore v5") | — (instrument) | MEASURABLE-NOW (infra, M not S) | pooled n≈15.5k → ±0.010; resolves Δ≥0.015 single, ~0.01 paired |
| Retrain on v5 32k labels (e1,e2,l2) | +0.03–0.08 | NEEDS-BIGGER-EVAL | Δ=0.03 needs n≈524·(0.05/0.03)²≈1,450; current eval sees only Δ≥0.07. Must freeze v5 holdout first — retraining consumes the benchmark |
| Hard-negative mining RUS/UKR NotFarm (e1) | NotFarm recall +0.1–0.2 | NEEDS-BIGGER-EVAL | eval NotFarm n=135, recall SE≈0.04 → recall jump visible, macro-F1 Δ not; training eats the rows |
| Per-country prior/threshold correction (e1,e3) | catastrophic-country rescue | MEASURABLE-NOW (coarse) | RUS n=438: binary F1 0.01→0.3 is ≫ CI; countries n<150 (UKR,MYS,IND,TUR) resolve only Δ≥~0.15 |
| Ensemble/soup softcon+ctx128 (e2,e3) | +0.01–0.03 | NEEDS-BIGGER-EVAL | n=524 resolves ≥0.07; blind slice ≥0.03; full v5 ≥0.01 |
| TTA on (e3) | +0.005–0.02 | NEEDS-BIGGER-EVAL | Δ=0.01 needs n≈13,100 |
| recompute_indices / BN-freeze ablation (e3) | ~±0.01 | NEEDS-BIGGER-EVAL | adopt as hygiene; unverifiable at any current n |
| Footprint late-fusion / geometry stage (e1,e3,l1,l4) | binary precision +0.1–0.5 | MEASURABLE-NOW (binary axis) | blind farm precision 0.61, n≈3.3k → SE~0.01; lit-scale effect ≫ CI. 3-class macro Δ needs v5 eval |
| OB 2.5D Temporal input channels (l4) | large, OOD-concentrated | MEASURABLE-NOW iff Δ≥0.03 on blind slice | blind slice is Global-South heavy = product coverage; USA/Europe part untestable there |
| Multi-frame/seasonal S2 (e3,l4) | +0.02–0.04 | NEEDS-BIGGER-EVAL | borderline on blind (±0.02); also requires re-extracting eval patches |
| S1 fusion (e3,l2,l4) | +0.02–0.05 | NEEDS-BIGGER-EVAL | same threshold; only top of range visible on blind slice |
| AEF/TESSERA embedding concat (l2) | +0.01–0.04 | NEEDS-BIGGER-EVAL | Δ=0.02 needs n≈3,275 |
| Noisy Student, per-class quota (e1,e2,l3) | +0.02–0.04 worst-slice | NEEDS-BIGGER-EVAL | worst-region metric needs per-country n≥300; only RUS/BGD-scale countries qualify today |
| LP-FT/surgical last-block FT on 300–500 visual labels (l3) | +0.02–0.05 | NEEDS-BIGGER-EVAL | plus label-spend conflict: tuning labels must not come from benchmark |
| WiSE-FT interpolation guard (l3) | ±0.01 protective | NEEDS-BIGGER-EVAL | certifying non-regression at ±0.01 needs n≈13k |
| Earth-Index human-in-loop per country (l1,l2) | per-country FP purge | NEEDS-NEW-SLICE (per-country holdout, ~100–200 labels each) | loop self-generates them; keep acquisition/eval labels disjoint |
| SEALS/AL OtherFarm acquisition BGD/NGA/PER (l3) | enables gen class-2 metric | NEEDS-NEW-SLICE (gen OtherFarm 4→≥50) | it IS the slice-builder; blind slice's 106 class-2 rows already give F1 SE≈0.05–0.08 |
| BGD candidate generation (e2) | deployment infra | NEEDS-NEW-SLICE (BGD inference pool) | nothing to measure until scored |
| Binary+subtype reframe; gen as 2-class+poultry-F1 (e2,l1); PR/budget curves (l1) | metric redefinition | MEASURABLE-NOW | binary F1 CI at n=524 ≈ ±0.03, tighter than macro |
| Cleanlab/AUM audit; v4→v5 churn audit (l3,e1) | flagged-row yield | MEASURABLE-NOW | validated by relabeling flagged rows, not by eval delta |
| Dedup splits/inference; G1/G3/G4 reporting (e1,e3) | bias removal +0.03 | MEASURABLE-NOW | measurement fix |
| Spatial-block splits (e1,e3,l4) | bias removal 10–28% | MEASURABLE-NOW | measurement fix |
| Decisions: close backbones, no FixMatch, skip SR (l2,l3,l4) | avoids negative EV | n/a | nothing to detect |

**Notes (measurement-fix ranking by detection power unlocked):**

Baseline arithmetic: n ≈ 524·(0.05/Δ)² → Δ=0.05:524, 0.03:1,455, 0.02:3,275, 0.01:13,100. Paired same-row bootstrap cuts required n roughly 2×; macro-F1 CI is minority-class-dominated (eval OtherFarm support 70), so pure-n scaling is optimistic unless class-2 support scales too (v5-new adds only ~499 OtherFarm).

1. **v5 eval expansion** — stage (a) adopt the 3,297-row blind slice (zero cost): ±0.020, makes Δ≥0.03 levers decidable today; stage (b) score the 14,138 unscored clusters: pooled ±0.010, unlocks the entire Δ=0.01–0.02 lever class (TTA, ensembles, embedding concat) — the only fix that does. Largest unlock by far. **Critical:** freeze a held-out benchmark split before the v5 retrain, or the retrain destroys the instrument.
2. **Dedup pipeline** — removes a known +0.03 bias and val-duplication checkpoint-selection bias; adds no variance power but the bias exceeds half the effects being chased; S-effort prerequisite for everything above.
3. **Spatial-block splits** — removes 10–28%-scale optimism from test/val and fixes checkpoint selection; converts test into a usable secondary instrument; no n gain.
4. **Gen OtherFarm acquisition** — unlocks one axis only (class-2 OOD F1, needs support ≥50 for ±0.1) and is partially pre-empted by the blind slice's 106 class-2 rows.

Flags: (i) e1 vs e2 conflict — "v5 blind predictions already exist" (repeated by e3/l2/l3/l4) is false per e2's diagonal crosstab; the 3,297 rows are untrained-but-previously-labeled, the truly new ~12.2k need patch generation + inference (M, not S). (ii) Blind slice is not deployment-representative (RUS 437 ~all-NotFarm = 13% of rows); per Wadoux/l4, keep Rachel's design-based sample as the headline estimator and use the blind slice for paired model ranking only. (iii) Any lever whose evidence is a gen-slice delta <0.07 (cRT sign-flip, softcon-vs-ctx128 non-stacking) is currently unmeasurable noise — do not spend GPU adjudicating it before fix 1.

## Appendix C — Feasibility audit (effort/dependency verifier)

| # | Lever (proposers) | Audited effort | Dependencies | Implementation path |
|---|---|---|---|---|
| 1 | v5 blind benchmark — in-scored slice (e1,e3,l2,l3,l4) | S | none | join scored parquets to the 3,297 `split=unlabeled` labeled rows; extend `scripts/post_hoc_evaluate.py` |
| 2 | v5 blind benchmark — full 12.2k unscored clusters (e2) | **M ⚠** | v5→candidates rerun (`training/rachel_to_candidates.py`) → patch extraction 14,138 clusters (`training/patch_extraction.py`, EE+disk quota) → `training/inference.py` both v9 ckpts | as row 1 after scoring |
| 3 | Retrain on v5 labels (e1 M, e2 L, l2 M) | **L ⚠** full / M partial | row 2; new splits (`training/dataset.py:build_splits:548`); rows 4–5 should land first | new master candidates CSV → `world_v10` config → train |
| 4 | Dedup by candidate_id (e3) | S | none | `drop_duplicates("candidate_id")` in `dataset.py:568-589` + `inference.py:199-268` |
| 5 | Spatial-block splits (e1,e3,l4) | M | row 4; benefits require retrain | geo-cell (H3/geohash) assignment inside `build_splits`; lat/lng already in candidates |
| 6 | Honest checkpoint selection (e3) | S | rows 4–5 | `train.py:386-394` val-metric on dedup/blocked val |
| 7 | Enable TTA (e3) | S | none | `tta: true` in `configs/rachel_clusters/world_v9_*.yaml` (wired at `inference.py:64-91`); add TTA to `train.py:_evaluate:79-103` |
| 8 | softcon+ctx128 prob ensemble (e2,e3) | S | none | offline averaging of `prob_class*` across existing scored parquets |
| 9 | Per-country thresholds/prior correction (e1,e3,l1) | S | none (CV guard for n<100 countries) | extend `scripts/logit_adjust_sweep.py`; note `inference.threshold` dead for 3-class (`inference.py:94-99`) |
| 10 | Footprint late-fusion stacker (e3,l1,l4) | S stacker / M polygon geometry | none (4 feats exist: `rachel_to_candidates.py:255`) | add feats to `inference.py:118-127` attach-list → GBM on dedup eval; polygon aspect/area rules need OB v3/Overture download (`training/building_footprints/` starting point) |
| 11 | OB 2.5D Temporal input channels (l4) | **L ⚠** (priced M) | per-source scaling fix (`dataset.py:137-140` /10000 + [0,1] clamp, `config.py:276-291`); new provider in `training/imagery/`; full re-extraction (hash: `config.py:260-261`) | M if labeled-only re-extraction; L incl. 118k inference pool |
| 12 | Multi-frame / seasonal S2 (e3,l4) | M labeled-only / L full | per-source `date_range` plumbing (`patch_extraction.py:211-232`) + 4× disk | add source-level dates to opts+hash; re-extract |
| 13 | S1 VV/VH concat (l4 priced S ⚠) | M | same scaling fix as row 11 (S1 dB annihilated by clamp — e3 bottleneck 4); re-extraction | `training/imagery/earth_engine_s1.py` exists; fix `dataset.py` per-channel scale spec |
| 14 | Noisy Student / self-training (e1,e2 M ⚠,e3,l3) | L | rows 1–2 benchmark gate; zero pseudo-label code exists (e3 grep) | new script: teacher parquet → per-class-quota pseudo-labels → student retrain |
| 15 | LP-FT / last-block surgical FT (l3) | S/M | v5 visual non-US slice (row 1 suffices) | invert freeze logic in `train.py:348-359`; cRT config precedent (`world_v9_softcon_crt.yaml`) |
| 16 | WiSE-FT interpolation (l3) | S | same-init pairs only (e1 caveat) | state-dict lerp script + rescore |
| 17 | Cleanlab label audit (l3) | S (AUM = M, needs dynamics logging) | none | per-row probs already in scored parquets |
| 18 | AEF embedding probe (l2) | S/M | GEE export quota, 157k clusters | standalone export + LR/MLP; no pipeline changes |
| 19 | TESSERA fusion (l2 priced S/M ⚠) | L | S1+S2 time-series input pipeline (nothing in `training/imagery/` supports it) | only viable if precomputed embeddings cover our years/AOIs |
| 20 | Earth-Index human-in-loop head (l1,l2) | M + ongoing labeling | penultimate-embedding export (small `inference.py` edit); Rachel loop | embed → MLP retrain-in-minutes script |
| 21 | BGD candidate generation (e2 S/M) | M | candidate-gen run (`training/osm_farm_finder.py`, `building_footprints/`) + extraction + inference | not config-only |
| 22 | SEALS/AL in BGD/NGA/PER (l3) | M | rows 20 (embeddings), 21 (BGD pool=0) | kNN-of-positives acquisition script |
| 23 | Temp scaling + conformal deferral (l3 priced M ⚠) | S | none | post-hoc on scored parquets via `post_hoc_evaluate.py` |
| 24 | Reporting fixes: dedup+CI+support flags; gen 2-class+poultry F1; per-country binary PR; detection-vs-typing reframe (e1,e2,l1) | S | none | `scripts/post_hoc_evaluate.py` + web artifact gen |
| 25 | v4→v5 churn audit (e1) | S | none; e2 already showed overlap diagonal — scope = 753 revoked (AUS 498) | pandas join |
| 26 | recompute_indices / jitter consistency; freeze-BN ablation (e3) | S each (+1 GPU run) | none | `world_v9_softcon.yaml:97-98`; `train.py:161`/`model.py:126-131` |
| 27 | Hard negatives from blind NotFarm (e1) | S | folds into row 3 | flip 3,297 labeled `split=unlabeled` rows to train in `build_splits` |
| 28 | Registry/covariate fusion head (l1 priced M ⚠) | L | per-country external data assembly (none in repo) | — |
| 29 | Decisions: close backbone search; no FixMatch; skip SR (l2,l3,l4) | 0 | none | ledger entry |

**Notes (mispricing flags):**

1. **Biggest repricing: the "free v5 blind benchmark" (row 1/2).** Five reports price it S on the premise "blind predictions already exist." e2's crosstab disproves this: all 17,253 overlapping 3-class rows already carry identical labels (nothing blind), and the 14,138 genuinely new clusters have **no patches** — so the headline benchmark costs a full patch-extraction + inference cycle (M, EE quota + disk). Only the 3,297-row in-scored slice (e1's analysis) is truly S, and it is 4× smaller than advertised and Europe-poor.
2. **Retrain-on-v5 (row 3): e1's M and l2's M are mispriced** — full version inherits row 2's extraction plus re-split/retrain (= e2's L). A same-week M variant exists: retrain on scored data only, adding the 3,297 rows (row 27) — captures RUS/UKR/POL hard negatives but none of the 9k unscored European NotFarm.
3. **All new-modality levers (rows 11–13) share two unpriced prerequisites** from e3: the hardcoded /10000 + [0,1] clamp in `dataset.py:137-140` and the imagery-hash-driven full re-extraction. l4's "S" for S1 concat and "M" for OB-Temporal ignore both; realistic floor is M with labeled-only re-extraction and a fresh train run.
4. **TESSERA (row 19) is the most mispriced lever in the dossier** (S/M claimed): local inference needs a per-cluster S1+S2 time-series pipeline the repo entirely lacks.
5. **Overpriced (cheaper than claimed):** temperature scaling/conformal (l3: M → S, pure post-hoc); footprint stacker v0 (features already flow to candidates CSV, one-line attach at `inference.py:118-127`).
6. **Sequencing constraint:** rows 4–6 (dedup, blocking, checkpoint selection) are S/M and gate the *validity* of everything measured after; they should land inside the row-2/3 rebuild, not after it — done later they force a second retrain.
7. **Ensemble caveat (row 8):** offline averaging is S, but deploying it doubles inference cost; no ensemble code exists (e3 grep), so productionizing is M.

## Appendix D — Evidence digests (key findings verbatim)

### D.e1-results — Results distillation (docs + web artifacts)

**Ledger verification — all headline numbers check out** against EXPERIMENTS_LOG.md and the web artifacts: softcon eval 0.504 raw / 0.4692 dedup (meta.json headline matches), ctx128 gen 0.440, softcon_crt eval −0.007 / gen −0.020, imagenet9ch eval 0.393 ≈ v8_three_class 0.403, AdaBN gen −0.024, logit-adjust τ=0 optimum, eval NotFarm recall 0.319 (recomputed from points.csv: 43/135). Two corrections and several extensions:

1. **CORRECTION — "1,072 labeled rows outside the whitelist" is stale.** The v9 map artifacts (labels synced from Rachel's Drive through 2026-07-06, 167 countries) contain **3,570 labeled rows outside training countries**: 3,297 in ~96 countries with `split=unlabeled` (predicted fully blind) + 273 BGD/NGA. All already scored by both v9 models.

2. **NEW — a 3,297-row blind OOD benchmark already exists and it breaks part of the "statistical tie."** Pooled over the 96 blind countries (paired rows, recomputed from points.csv): **ctx128 macroF1 0.509 vs softcon 0.462 (+0.047, n=3,297 — 12× the gen slice)**. ctx128 wins every class axis: NotFarm 0.646 vs 0.598, Poultry 0.693 vs 0.676, OtherFarm 0.188 vs 0.111. The ctx128>softcon OOD ranking is real, not noise; the eval-slice tie (n=524) stands.

3. **NEW — OOD OtherFarm is finally measurable, and it is the worst axis in the project.** The blind slice has 106 OtherFarm rows (vs gen's 4): F1 0.111 (softcon) / 0.188 (ctx128), vs ~0.27–0.30 on eval. Note eval dedup OtherFarm support is 70 (98 was pre-dedup), recall 0.286.

4. **NEW — farm over-prediction confirmed at scale.** Blind-slice NotFarm recall 0.474/0.535; binary farm precision only 0.606/0.632 (F1 0.717/0.732 vs eval 0.855). The model calls farm on roughly half of confirmed non-farm clusters outside training countries.

5. **NEW — per-country blind performance is bimodal, catastrophic in whole regions.** Decent: BGD 0.61–0.65, NGA 0.58–0.65, POL 0.53–0.57, MMR 0.61–0.65, Australia binary-farm 0.93–0.94. Catastrophic: **RUS binary farm F1 0.010–0.012** (n=438, ~all NotFarm, model predicts farm on ~half), UKR farm precision 0.0 (n=145), MYS farm F1 0.037/null (n=95), VNM macro 0.37–0.45 (n=101), IND 0.25–0.28 (n=67), TUR 0.29–0.30 (n=85). The global map is effectively noise in S/SE/Central Asia and Eastern Europe.

6. **Caveat on summary.json `overall_metrics`** (macro 0.759, n=18,006): mixes train/val/test rows — never quote it.

**Unresolved puzzles:**
- Why softcon and ctx128 don't stack (softcon_ctx128 gen 0.378, worst of family) — no mechanism; single seed, CI ~0.10; the blind slice could adjudicate but softcon_ctx128 has no published points.csv.
- cRT sign-flip (helps ssl4eo +0.019 gen, hurts softcon −0.020) — both deltas are inside gen's n=273 noise; the ledger over-reads them as real effects.
- RUS/UKR label composition (~all NotFarm): rejected-region export, or mapping artifact? Determines whether "catastrophic Eastern Europe" is model failure or prior mismatch.
- Whether v5's ~15k new labels alter the train/eval label-source composition (registry vs visual) — decides if the eval gap closes "for free" on retrain.

**Roadmap (A1–F2, G) status:** DONE: B1 (softcon, delivered only +0.015 eval = within noise), B3 (ablation confirmed pretraining), C1 (ctx128, adopt for OOD). OBSOLETED: B5/B6 and the whole backbone axis (log's own verdict: plateau at eval≈0.50); A2's *measurement* half (blind slice already provides 106 OOD OtherFarm rows) — its few-shot-data half stands, and v5 may already contain the labels. STRENGTHENED: A1 (blind OtherFarm 0.11–0.19), A4 (blind farm precision 0.61; RUS/UKR/MYS exports are ~600+ ready-made hard negatives currently unused), A3+A6 (registry eval rows 0.413 vs visual 0.470), A7 (tie ⇒ can't measure progress; v4→v5 label churn on overlapping rows gives a free inter-release agreement estimate), B4 AlphaEarth (as ranker for the 96 failing countries), D1 (unlabeled pool covers exact failing countries — but v5 converts ~15k pseudo-candidates to real labels, so retrain-on-v5 outranks it), D4 (now applicable to many countries via v5), E2 per-country priors (RUS/UKR over-prediction is a textbook prior mismatch), F1 (morphological OOD story), G2 (27.1% eval-train proximity quantified), G3 (still emitting misleading 0.000 rows). WEAKENED: D2/D3 (cRT-on-softcon negative; AdaBN negative), B2 (as metric lever; keep as fusion vehicle), E4 soup (plateau + non-stacking; also softcon and ssl4eo are *different inits* — soup families limited to {ssl4eo,crt,ctx128} and {softcon,*}), E1 cascade (the binary stage itself is what fails OOD). UNCHANGED: A5, C2–C4, D5, D6-SKIP, E3, F2, G1 (notebook did the CI half; pipeline formalization pending), G4. Never run despite DO-verdicts: E4, A3, A7, F1, B4, D1, G1–G4.

### D.e2-data — Hands-on data analysis (scored parquets + rachel_labels_v5)

**A. The premised "fresh blind benchmark" does NOT exist in the scored data — NEW-label rows = 0.** v5 has 32,050 final labels; 29,477 map to 3-class (2,573 dropped categories: Unknown 2,171, Ambiguous 176, Mixed 103, PigsOrPoultry 102, Other 21). Of these, 17,912 v5-labeled clusters exist in scored data (dedup 133,026), and the y5↔true_label crosstab is **perfectly diagonal**: all 17,253 overlapping 3-class rows already carry the identical `true_label`; the other 659 overlaps are dropped-category rows with true_label=-1. So the scored parquets' "v4-era" labels are already v5-consistent — nothing was predicted blind. The genuinely new labels are the **14,138 v5-labeled clusters absent from scored data entirely (no patches)**: class mix NotFarm 9,030 / Poultry 2,695 / Other 499 / dropped 1,914; top countries RUS 2,535, DEU 1,101, ARG 889, AUS 781, ITA 701, UKR 701, FRA 665, BLR 457, CZE 414, USA 368. Reverse churn: 753 scored-labeled rows lost their v5 label (639 poultry; AUS 498) — all in split=unlabeled, so eval/gen/train slices are unaffected. PER: only 29 v5 labels (14 poultry) despite 1,953 PER clusters scored.

**B. Error structure (dedup by candidate_id).** eval n=524: softcon acc 0.599 / macroF1 0.469; ctx128 0.599 / 0.459. Disagreement 24.8%; both-right 49.8%, exactly-one-right 20.2%, both-wrong 30.0% → **oracle-union acc 0.700 (+0.101 over either model)**. Disagreements split evenly (sc-right 53, cx-right 53, neither 24). gen n=273: softcon acc 0.685 / mF1 0.402; ctx128 0.758 / 0.440; disagreement 24.2%, both-wrong 16.8%, oracle 0.832; **disagreements resolve to ctx128 2:1 (40 vs 20)**. Eval error class-pairs (softcon, 210 errors): 0→1 =77, 1→2 =45, 2→1 =43, 1→0 =23, 0→2 =15, 2→0 =7 — i.e. 44% of errors are NotFarm→farm false positives and 42% are Poultry↔OtherFarm confusion; ctx128 nearly identical. Gen errors: 0→1 =32, 1→0 =29, 1→2 =17 (softcon). By label_source (eval): Visual inspection n=446 acc 0.594/0.578, Chile_Liebsch n=38 acc 0.579/0.632, Mexico_official n=33 acc 0.697/0.818.

**C. Pseudo-label yield.** Unlabeled pool (split=unlabeled & true_label=-1) n=115,020; model agreement 73.4%. Both-agree & both score≥0.9: **14,234 rows** (NotFarm 2,483 / Poultry 11,275 / Other 476). Precision proxy on held-out labeled rows with the same filter: test coverage 33%, acc 0.892 (poultry precision 0.881 n=445, class2 0.914 n=93); eval acc 0.805 (poultry 0.819); gen acc 0.816 (poultry 0.860). **BGD has ZERO unlabeled candidates in the scored files** (only its 171 generalization rows exist); NGA: pool 454, confident-agree poultry 37. Top confident-poultry countries: USA 4,746, IND 1,063, BRA 804, MEX 778, THA 666, IDN 403, ARG 349, PHL 251, PAK 234, VNM 169.

**D. Unused labeled data in scored files: 3,297 rows** (split=unlabeled & true_label≠-1; larger than the documented 1,072): class 0/1/2 = 1,717 / 1,474 / 106. Top: POL 726 (370 poultry), AUS 696 (641 poultry, but note 498 more AUS labels were revoked in v5), RUS 438 (437 NotFarm), UKR 145, VNM 101, MYS 95, TUR 85, ARG 74.

**E. v5 delta.** 32,050 v5 finals vs 18,006 scored labeled (dedup) / ~16.7k v4. BGD: 182 labels = 143 poultry, 24 NotFarm, **4 OtherFarm**; NGA: 119 = 68 poultry, 34 NotFarm, **0 OtherFarm** → gen class-2 support stays at 4; gen f1_class2 remains uninformative. Countries with ≥100 v5 labels and zero prior scored labels (all Europe): DEU 1,101, ITA 701, FRA 665, BLR 457, CZE 414, ROU 325, GBR 218, PRT 205, HUN 195, ESP 158, SVK 106, DNK 105, SWE 105. Other big non-train pools: RUS 2,963 (99% NotFarm), ARG 1,366, AUS 1,054 (701 poultry), UKR 847, POL 730.

### D.e3-code — Code audit (training/)

**Implemented but unused (in the winning v9 recipe):**
- **TTA**: full 8-way dihedral TTA exists (`training/inference.py:64-73`), gated by `cfg.inference.tta` (default False, `training/config.py:527`); no config in `configs/` sets `tta: true`, and train-time slice evaluation (`train.py:_evaluate`, :79-103) never uses it — reported eval/gen numbers never benefit.
- **Cutout + recompute_indices**: implemented (`dataset.py:64-77`, :50-61) but `cutout.enabled: false`, `recompute_indices: false` in `world_v9_softcon.yaml:97-98`. Side effect: `per_band_jitter` is on while `recompute_indices` is off, so NDVI/NDBI/NDWI channels are physically inconsistent with the jittered bands during training (`dataset.py:210-228`).
- **SCL pixel cloud mask**: implemented (`imagery/earth_engine_s2.py:53-81`, `cloud_mask: "scl"`), but v9 config uses default `"none"` — only the scene-level `CLOUDY_PIXEL_PERCENTAGE<15` filter applies.
- **`composite: least_cloudy`** is a declared config option (`config.py:196`) that no provider implements — `build_image` hardcodes `.median()` (`earth_engine_s2.py:81,93`). Dead knob.
- **S1 provider + multi-source stacking**: registry, provider, and channel-stacking all exist (`imagery/earth_engine_s1.py`, `imagery/__init__.py:12-42`, `patch_extraction.py:119-134`) but are unusable end-to-end (see Bottleneck 4).
- **FocalLoss** (`losses.py:21-38`) never selected by any current config; band-name-mapped first-conv (`model.py:190-220`) IS used (softcon) — good.
- **Footprint features flow into candidate CSVs and stop there**: `rachel_to_candidates.py:255` keeps `num_bldgs,total_area_m2,median_area,template_score_if`; nothing downstream reads them — `PatchDataset` ignores them, and `inference.py:_attach_labels:119-127` deliberately omits them from `scored_candidates.parquet`.
- **Confidence tiers** are computed (`inference.py:30-45,253`) but semantically broken for 3-class: `scores` = top-1 softmax ≥ 1/3 always, so the `very_low` (<0.4) tier is near-empty and tiers don't distinguish classes. `inference.threshold` is ignored entirely for num_classes≥3 (`inference.py:94-99`).

**Recipe facts:** checkpoint metric = macro-F1 on the val split (`train.py:386-394`, `world_v9_softcon.yaml:73`); AdamW + cosine; freeze backbone 5 epochs then rebuild optimizer at 0.1×LR with a *fresh* cosine schedule of T_max=50 (`train.py:348-359`, `_build_scheduler:579-590`); no warmup, no EMA, no mixup/cutmix, no label smoothing, no gradient clipping (grep over `training/` + `scripts/` confirms zero hits for ema/mixup/cutmix/soup/ensemble/calibration/pseudo).


## Appendix E — Literature digests (key findings verbatim, with citations)

### E.l1-cafo — CAFO detection literature

**1) SOTA CAFO/poultry detection is 1m-aerial, US-only, binary — and our 10m numbers are competitive with it.**
- Robinson et al. 2022 (Microsoft, U-Net on NAIP 1m, Delmarva 6,013 barn polygons): recall 87% / precision 83% in-region; recall 86.9% / precision 83.0% on 10 validated CA counties (≈F1 0.85). Self-reported. Extreme lower-bound precision outside validated areas: 18.8%. Rotated-rect footprint filter (area [525, 8106] m², aspect [3.4, 20.49], road-intersection) gave **+52.3 pts precision for −3.3 pts recall** on average — the filter, not the CNN, is their precision engine. Rotation augmentation added +22–27 pts recall. Full-US: 7.1M raw polygons → 360,857 after filter+dedup. https://arxiv.org/abs/2112.10988 (2021/22), https://github.com/microsoft/poultry-cafos
- Handan-Nader & Ho 2019 (Nature Sustainability, NAIP 1m, NC): image-level PR-AUC 0.917 poultry / 0.923 pig; facility-level **precision 73% / recall 70%** vs manual census; found +15% previously unknown CAFOs. Self-reported. https://www.nature.com/articles/s41893-019-0246-x (2019)
- **Earth Genome Earth Index (Jan 2024) is the direct 10m Sentinel-2 comparator**: ViT-DINO embeddings on 32×32-px patches + MLP head, human-in-the-loop notebook labeling (~242 labels added to 15,655 RegLab samples, ~1 week of work): NC validation **precision 0.94 / recall 0.82** (≈F1 0.876), acc 0.98, AUC 0.99; 16,372 detections across 6 SE-US states. Self-reported, validated in-region (NC, same label source as training). https://medium.com/earthrisemedia/finding-5-billion-chickens-with-human-in-the-loop-ai-model-tuning-via-earth-index-1d3f5cc89aec (2024)
- Independent re-analysis deflates published maps: Tulbure et al. 2024 (GeoHealth) applied literature heuristics to Robinson's dataset and removed **51.8% (NC) / 61.5% (US) of detections as misclassified** (swine barns, hangars, nurseries), cutting poultry-density overestimation 54%. Independent. https://pubmed.ncbi.nlm.nih.gov/39697399/ (2024)
- 2025–26 newcomers: cattle feedlot CONUS detection, YOLOv11 on NAIP, 11,746 labeled feedlots, **F1 0.72–0.86** (self-reported) https://www.sciencedirect.com/science/article/pii/S0048969726001117 (2026); Myanmar poultry-fish (Nature Food 2025, only LMIC example found) used sub-meter Google Earth imagery + YOLOv4, "99% counting accuracy" vs census 1,508 farms — high-res, not 10m https://www.nature.com/articles/s43016-025-01192-1 (2025); EWG NC poultry+swine map (2024, no P/R published) https://www.ewg.org/research/innovative-ewg-study-uses-ai-find-357m-poultry-north-carolinas-factory-farms (2024); AlphaEarth Foundations 10m embeddings + linear SVM found 68–71% of Sumatra's 867 palm-oil mills from ~124 positives over 480,000 km² in 174s (Google, Apr 2026, self-reported) https://medium.com/google-earth/seeding-the-search-alphaearth-foundations-satellite-embeddings-for-detecting-agricultural-43cf78e1cc5f (2026), model: https://arxiv.org/abs/2507.22291 (2025).

**2) NOBODY published poultry-vs-pig-vs-cattle discrimination from 10m imagery.** Type is resolved by: (a) sub-meter footprint geometry — Robinson explicitly flags hog barns as having different aspect ratios; Tulbure's heuristics remove swine from poultry maps geometrically; (b) separate per-type 1m models (Handan-Nader: pigs cue on lagoons); (c) registry/permit fusion + human annotation — the California factory-farm dataset (Scientific Data, Nov 2025) typed animals using permits + Google Earth/StreetView, and **even human annotators discarded 18.6% of facilities at the animal-typing stage** (kappa 0.73) https://www.nature.com/articles/s41597-025-06082-6 (2025); (d) no-imagery covariate ML: parcel-scale random forest on 58 socio-environmental variables hits 87% AFO mapping accuracy https://pubmed.ncbi.nlm.nih.gov/39765170/ (2025). Our 3-class-from-10m task has **no published baseline**; macro-F1 0.47 deployment / 0.70 ID cannot be called weak against literature — it is unpublished territory, and binary F1 0.855 eval is at parity with Earth Index's in-region 0.876 while covering 5 training countries + OOD.

**3) Operational triage lessons.** Stanford: capture 75% of CAFOs reviewing <2% of images, 95% with <10%; facility consolidation → 70% recall at 0.28% manual effort (2019, above). Robinson: pick segmentation threshold on validation PR curve for the desired operating point, then rule-filter. Earth Genome: per-region human-in-loop retraining with fresh negatives whenever new terrain (Mississippi valley, S. Georgia) spawned FP families (2024, above). RegLab Wisconsin field trials (manure-application detection, arXiv 2501.04902, 2025): threshold 0.5 to dispatch, desk review screens out the majority; **ground confirmation ~35% at confidence >0.8, <10% at 0.5–0.6** — production precision runs far below paper metrics, and agency workflow design mattered more than model quality. https://arxiv.org/abs/2501.04902 (2025)

### E.l2-gfm — Geospatial foundation models

**1. Landscape map vs SoftCon (our RN50, arXiv 2405.20462).** Closest proxy to our regime is GEO-Bench m-EuroSAT (64px, 13-band S2, linear probe/kNN). The Panopticon paper's comparison table (arXiv 2503.10845, CVPR EarthVision 2025 best paper; self-reported but includes rivals as baselines) shows **SoftCon is already at the top**: SoftCon 84.3% > Panopticon ViT-B 83.9 > CROMA 79.4 > DOFA 78.8 > AnySat 76.8. On Copernicus-Bench EuroSAT-S2 OA (arXiv 2503.11849, ICCV 2025, self-reported): Copernicus-FM 97.9 vs DOFA 97.2 vs supervised-from-scratch 97.6 vs SoftCon 96.7 — deltas ≤1.2 OA pts on a saturated benchmark. Per-model:

- **TerraMind** (IBM/ESA, arXiv 2504.11171; HF `ibm-esa-geospatial/TerraMind-1.0-{tiny..large}`, TerraTorch integration, accepts S2L2A **band subsets** incl. our 6 raw bands): tops PANGAEA segmentation (mIoU 58.35, +1–4pts over U-Net/ViT baselines); classification evidence is few-shot EuroSAT (70–88% 1/5-shot). Segmentation-oriented; no evidence of large patch-classification gains over SoftCon.
- **Galileo** (NASA Harvest/Ai2, arXiv 2502.09356, ICML 2025; HF `nasaharvest/galileo`, nano/tiny/base): strength is **multimodal pixel time series** (CropHarvest, crops); "beats AnySat by 10.8% on EuroSAT" (self-reported) — but AnySat is below SoftCon anyway. Expects time-series/multi-modality inputs; our single annual composite wastes its design.
- **Panopticon** (arXiv 2503.10845; GitHub Panopticon-FM, DINOv2-based, any-channel via wavelength-conditioned cross-attention): SOTA among *any-sensor* models, still ≈SoftCon on m-EuroSAT (83.9 vs 84.3). Value = channel flexibility (could ingest our 9ch natively), not accuracy.
- **Copernicus-FM** (arXiv 2503.11849): +1.2 OA over SoftCon on EuroSAT-S2, +3.6 on EuroSAT-S1. Marginal for pure-S2 classification.
- **Prithvi-EO-2.0** (arXiv 2412.02732; HF ibm-nasa-geospatial): HLS **30m** pretraining, temporal ViT — resolution mismatch with 10m barn morphology; AEF paper shows Prithvi below AEF on mapping evals; skip.
- **DOFA/DOFA-2**: no "DOFA-2" backbone exists; successor is **DOFA-CLIP** (arXiv 2503.06312), zero-shot VLM-oriented. DOFA m-EuroSAT 78.8 < SoftCon.
- **Clay v1.5** (clay-foundation.github.io): EuroSAT 98% self-reported on a benchmark the community itself calls saturated ("ImageNet weights get 98%+", GitHub discussion #269); independently, AEF paper shows Clay among the weaker featurizers. No rigorous small-patch S2 evidence.
- **AnySat**: 76.8 m-EuroSAT; below SoftCon.
- **AlphaEarth Foundations / Satellite Embedding V1** (arXiv 2507.22291, 2025; GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` + AWS open registry): 64-dim, 10m, annual 2017–2024 (v1.1). **Weights not released — embeddings only**; cannot fine-tune. Self-reported: −23.9% mapping error vs SatCLIP/Prithvi/Clay/MOSAIKS across 15 evals. Independent: "Harvesting AlphaEarth" (IJAEOG 2026, arXiv 2601.00857) — matches handcrafted RS features but "rivals local models, lacks transferability"; TESSERA (arXiv 2506.20380, Cambridge, **open** 128-dim S1+S2 pixel embeddings) beats AEF on canopy height (RMSE 12.2 vs 16.1m) and Swiss LCZ mapping. Practitioner reports (yag.xyz structure-search post) flag **spatial smoothing hurting small-structure retrieval** — a concern for ~10–100m barns.

**2. Frozen vs fine-tuned at 10–30k labels.** PANGAEA (arXiv 2412.04204): frozen GFM encoders often fail to beat an end-to-end supervised U-Net; fine-tuning improves results **non-systematically** (model- and task-dependent). "No One Knows the SOTA in GFMs" (arXiv 2605.12678, 2026): published GFM rankings are unstable, riddled with unacknowledged statistical ties; no model dominates. Transfer study arXiv 2606.13896 (2026, 6 GeoFMs): **decoder/adaptation choices are as impactful as the choice of GeoFM**. Net: at our 16–32k labels, fine-tuning still wins over linear probing in-domain, but the marginal gain from *which* pretrained encoder is within benchmark noise — externally confirming our internal top-6 tie.

**3. Earth Index recipe** (Earth Genome; Frontiers in Climate 2025, doi 10.3389/fclim.2025.1520242 + Medium "Finding 5 billion chickens"): frozen ViT-DINO S2 embeddings (32×32px patches) + MLP head + iterative human triage. Poultry CAFOs: 16,372 detections across 6 US states; NC holdout precision 0.94 / recall 0.82 starting from 15,655 RegLab labels + only **242 human-triage clicks**; head retrains in minutes on a laptop for 100,000 km². Critical datum: **naive extrapolation to Arkansas dropped 87%→~68% precision; the human-in-the-loop head restored it** — i.e., their OOD fix is cheap labels + fast head iteration, not a better backbone. This is the same failure mode as our BGD/NGA gap and eval NotFarm recall 0.319.

**4. Verdict: backbone axis is CONFIRMED DEAD for us.** SoftCon already sits at the top of the closest public proxy (m-EuroSAT linear probe); all published rivals differ by ≤1–2 OA pts on saturated benchmarks; two 2026 meta-studies say GFM rankings are statistically indistinguishable; our own eval CI width (~0.10, n=524) exceeds every published delta. No swap has plausible expected gain >+0.05 eval macro-F1. The exception is not a backbone but **auxiliary frozen embeddings as extra features** (AEF/TESSERA), which encode multi-sensor/temporal signal our S2-composite model never sees — "Better Together" (arXiv 2605.18667, 2026) shows fused embeddings beat the best single model in 4/6 tasks.

### E.l3-labels — Label-efficiency + geographic OOD

**(1) Self-training for geographic OOD — the FMoW-WILDS evidence HOLDS.** Verified against the U-WILDS paper (Extending the WILDS Benchmark for Unsupervised Adaptation, ICLR 2022, [arXiv:2112.05090](https://arxiv.org/abs/2112.05090)), FMoW OOD worst-region acc: ERM 33.7, **FixMatch 32.1 (hurts)**, Pseudo-Label 33.7 (flat), CORAL 34.1, DANN 34.6, **Noisy Student 37.8 (+4.1)**, SwAV pretrain 36.3. Stated mechanism: batch-level *dynamically updated* pseudo-labels (FixMatch) hurt generalization; Noisy Student's staged teacher-trained-to-convergence → static pseudo-labels → student is what works. No 2024–2026 SSL result overturns this; newer FMoW leaderboard gains (D³G + location encoders, ~51.8, [arXiv:2503.02036](https://arxiv.org/pdf/2503.02036), 2025) use *labeled* domain metadata, not unlabeled data. **Per-class quota > global threshold** is well supported: CReST (CVPR 2021, [arXiv:2102.09559](https://arxiv.org/abs/2102.09559)) shows minority-class pseudo-labels have *high precision* under imbalance, so sample minority classes more aggressively per estimated class prior; 2025 continuations: CAT class-aware adaptive thresholding (+3.4–10.9 pts on DG benchmarks, [PLOS One 2025](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329799)); CSTN self-training + contrastive DA for cross-region crop mapping (ISPRS 2025, [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1569843225000263)); pseudo-labels for label-scarce smallholder regions ([arXiv:2312.08384](https://arxiv.org/pdf/2312.08384)).

**(2) Few-shot target-domain adaptation (50–500 labels) — 2022 trio NOT superseded.** LP-FT ([arXiv:2202.10054](https://arxiv.org/pdf/2202.10054)): ~+10 pts OOD vs naive full FT, best on 10/10 OOD sets; LP alone beats FT at very small n, FT wins as n grows ([arXiv:2205.07874](https://arxiv.org/abs/2205.07874) gives ResNet-scale few-sample recipes: update head only + data/test-time augmentation). Surgical FT ([arXiv:2210.11466](https://arxiv.org/abs/2210.11466)): **tune LAST block/layer for output-level (label/spurious-correlation) shift**, first block for input-level shift; Auto-RGN auto-selects layers. Our diagnosed shift (label-source, not radiometric — AdaBN negative) is output-level, and cRT (+0.019 gen) is exactly last-layer retraining, so literature predicts last-block FT on target-distribution labels is the right regime. WiSE-FT ([arXiv:2109.01903](https://arxiv.org/abs/2109.01903)) and model soups ([arXiv:2203.05482](https://arxiv.org/pdf/2203.05482)) weight interpolation remain the standard regression-guard; ICLR 2026 extends robust FT to non-robust (non-CLIP) backbones ([arXiv:2509.23325](https://arxiv.org/pdf/2509.23325)) — confirms interpolation works from ordinary ResNet inits. Expected ResNet effect sizes: +1–5 pts, largest when head/late layers are the bottleneck.

**(3) Active learning at 100–1000 budgets.** Regime is decisive: at low budget, uncertainty sampling has a cold-start problem and loses to random; typicality/diversity wins — TypiClust ([arXiv:2202.02794](https://avihu111.github.io/Active-Learning/)), ProbCover ([arXiv:2205.11320](https://arxiv.org/pdf/2205.11320)). For **rare-class search**: SEALS (AAAI 2022, [arXiv:2007.00077](https://arxiv.org/pdf/2007.00077)) — restrict acquisition to kNN of labeled positives in a pretrained embedding, near-optimal recall at a fraction of labeling cost; GALAXY (ICML 2022, [OpenReview](https://openreview.net/forum?id=G67PtYbCImX)) collects markedly more class-balanced batches than vanilla uncertainty. With strong frozen features, simple uncertainty on top recovers ([arXiv:2401.14555](https://arxiv.org/pdf/2401.14555), 2024). **Deployed satellite loops (direct analogs):** Microsoft poultry-CAFO USA ([arXiv:2112.10988](https://arxiv.org/abs/2112.10988), [github.com/microsoft/poultry-cafos](https://github.com/microsoft/poultry-cafos)) — model-assisted labeling + post-hoc FP filtering; brick kiln India/Bangladesh AL deployment ([arXiv:2402.13796](https://arxiv.org/html/2402.13796v1), 2024; [AL paper](https://www.researchgate.net/publication/380263920)) — "subset entropy" acquisition beat BALD, 35k images labeled fast with a purpose-built tool; SentinelKilnDB (NeurIPS 2025, [poster](https://neurips.cc/virtual/2025/poster/121530)) shows S2-scale rare-structure search at 62k kilns / 2.8M km².

**(4) Calibration + selective prediction.** Temperature scaling fitted in-distribution degrades under shift; multi-domain temperature scaling ([arXiv:2206.02757](https://arxiv.org/pdf/2206.02757)) and consistency-guided TS (CVPR 2024, [researchgate](https://www.researchgate.net/publication/379278019)) are the robust variants — fit per-country/per-source temperatures. Conformal cost-aware deferral for triage: clinical study ([Sci Reports 2026](https://www.nature.com/articles/s41598-026-40637-w)) cut retained-case error **49.6% ID / 46.7% OOD** via selective deferral; caution: conformal coverage breaks under prevalence shift ([arXiv:2605.20956](https://arxiv.org/html/2605.20956), 2026) — directly relevant given eval NotFarm recall 0.319 = prevalence mismatch between train and deployment.

**(5) Label noise.** Cleanlab confident learning ([arXiv:1911.00068](https://github.com/sailfish009/cleanlab), JAIR 2021) is explicitly robust to *sparse class-conditional* noise (tiger↔lion analog = Poultry↔OtherFarm); AUM ([arXiv:2001.10528](https://arxiv.org/pdf/2001.10528), NeurIPS 2020) ranks by training-dynamics margin. 2026 remote-sensing-specific assessment ([arXiv:2603.16835](https://arxiv.org/pdf/2603.16835)) validates these data-centric methods on RS benchmarks at 10–70% noise; AQuA benchmark tool ([arXiv:2306.09467](https://arxiv.org/pdf/2306.09467)).

### E.l4-inputs — Input modalities + evaluation practice

**(1) Multi-temporal for STATIC structures — yes, and the mechanism is multi-frame super-resolution, not phenology.** Google's "High-Resolution Building and Road Detection from Sentinel-2" (arXiv:2310.11622, 2023, https://arxiv.org/abs/2310.11622) is the definitive datapoint: teacher-student distillation from 50cm imagery, student consumes a stack of raw cloud-filtered S2 frames. Ablation (building mIoU): 1 frame 71.7 → 2: 73.2 → 4: 74.5 → 8: 75.8 → 16: 76.1 → 32: 76.7 (**+5.0pp from 1→32; +2.8pp from just 1→4**). Monotonic; past-only 16 frames (76.6) ≈ full window, so no future data needed. Gains come from sub-pixel aliasing across frames + shadow/illumination diversity — exactly the signal an annual median composite destroys. Effective resolution of the resulting product: ~4m from 10m inputs. Seasonal composites also encode shadow-derived height: 4-seasonal-composite building-stories estimation, RSE 2024 (https://www.sciencedirect.com/science/article/abs/pii/S0034425724000282). Crop-phenology literature (+5-10pp from time series) is not the relevant evidence class here; the Google ablation is.

**(2) S1+S2 fusion — real but modest gains, late/dual-stream ≥ early concat.** Hafner et al., dual-stream U-Net for urban change detection (IEEE GRSL 2021, https://ieeexplore.ieee.org/document/9570476/): late-fusion F1 0.600 vs 0.555 for early channel-concat of the same inputs (**+0.045 F1**); follow-up semi-supervised multi-modal consistency version (Remote Sensing 2023, 15:5135, https://doi.org/10.3390/rs15215135). SAR alone can map buildings at ~20m globally (MDPI RS 2018, https://www.mdpi.com/2072-4292/10/11/1833); S1 backscatter is strongly sensitive to large metal-roofed agricultural buildings via double-bounce, but orientation-dependent (Koppel et al. 2017, "Sensitivity of Sentinel-1 backscatter to characteristics of buildings"). World Settlement Footprint used S1+S2 jointly as best-performing built-up input. Realistic expectation for our task: **+0.02-0.05 macro-F1-scale gain**, additive structural signal (long metal barns = bright persistent double-bounce amid dark fields). Note our AdaBN result already showed the OOD gap is not radiometric, so S1's value here is structure contrast, not domain robustness.

**(3) Building-footprint products — coverage fits our OOD countries; the temporal raster is the sleeper asset.** Google Open Buildings v3 (2023): 1.8B polygons, 58M km², Africa + South Asia (BGD ✓) + SE Asia (THA ✓) + LatAm/Caribbean (BRA/MEX/PER ✓); **no USA/Europe**; confidence 0.5-1.0, threshold 0.7 for precision (https://sites.research.google/gr/open-buildings/). **Open Buildings 2.5D Temporal (2024)**: annual 2016-2023 rasters of building presence, fractional count, AND height, ~4m effective resolution, built from S2 stacks via the arXiv:2310.11622 model; same Global South coverage; in GEE as `GOOGLE/Research/open-buildings-temporal_v1` (https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1). Caveat: height ground truth was Global North only (https://research.google/blog/open-buildings-25d-temporal-dataset-tracks-building-changes-across-the-global-south/). Microsoft Global ML Footprints: global incl. USA/CHL, claimed 94.4% precision in Africa, weaker rural recall. Overture (≥2.3B) conflates Google+Microsoft+OSM+Esri — best single access path (https://overturemaps.org/blog/2023/overture-buildings-theme-hits-2-3b-buildings-with-addition-of-google-open-buildings-data/). Products disagree most in rural Africa (CEUS 2024, https://www.sciencedirect.com/science/article/pii/S0198971524000334; Planet/Sentinel Hub analysis: small structures + canopy occlusion + imagery vintage). CAFO precedent: Handan-Nader & Ho 2019 (Nature Sustainability, https://www.nature.com/articles/s41893-019-0246-x) found +15% more NC poultry CAFOs with CNNs on high-res imagery; Microsoft footprints already used as CAFO training labels. **Nobody appears to publish footprint-geometry × S2 fusion for livestock-facility classification — open niche**, and poultry barns' signature (parallel rows of ~150×15m rectangles) is precisely the morphological signal our OOD gap points at.

**(4) Super-resolution — 2024-2026 evidence still says skip explicit SR.** Single-image SR hallucinates structure; image-quality metrics don't correlate with downstream accuracy ("Beyond Pretty Pictures", arXiv:2505.24799, 2025; TU Wien downstream evaluation, 2025, https://repositum.tuwien.at/bitstream/20.500.12708/221077/1/...pdf). Multi-image SR is more trustworthy but the better pattern is arXiv:2310.11622's: let the task model consume the raw multi-frame stack (implicit SR inside the detector) — or consume Open Buildings Temporal rasters, which already embody validated multi-frame SR.

**(5) Spatial CV — random splits inflate 10-30%; block ≥ autocorrelation range.** Kattenborn et al. 2022 (Science of Remote Sensing, https://www.sciencedirect.com/science/article/pii/S2667393222000072): random holdouts overestimate CNN performance **up to 28%** vs block CV. Karasiak et al. 2021 (Machine Learning, "Spatial dependence between training and test sets…"): up to 17% bias. Ploton et al. 2020 (Nat Commun, https://www.nature.com/articles/s41467-020-18321-y): random 10-fold R²=0.50 → ~0 under buffered-LOO as buffer approaches the autocorrelation range. Standards: Roberts et al. 2017 (Ecography 40:913) — block size ≥ residual autocorrelation range, estimated by variogram; Milà et al. 2022 NNDM / Linnenbrink et al. 2024 kNNDM — match CV nearest-neighbor distance distribution to the deployment prediction distribution. Counterpoint (Wadoux et al. 2021, Ecological Modelling, https://www.sciencedirect.com/science/article/abs/pii/S0304380021002489): if you have a probability sample of the deployment population, design-based validation beats spatial CV — **our Rachel eval slice is exactly that; treat it as the unbiased estimator and use block CV only for model/checkpoint selection.**


## Appendix F — Farm-type track: verdict table (adversarial synthesis auditor)

Cross-audit of the 4 type-track reports against each other, the results ledger, pricing arithmetic, class-support feasibility, and post-Phase-0/1 measurability.

### Verdict table

| # | Option (source) | Verdict | One-line reason |
|---|---|---|---|
| 1 | Stage-0 frozen-embedding type probe (t4#2) | **DO FIRST** | Hours, no GPU; produces the ceiling estimate that gates #2, #8, #9; zero detection risk. |
| 2 | Hierarchical two-head + partial-label loss (t1#2 ≡ t4#1) | **DO** | Only option all 4 reports independently converge on; +2,397 free detection rows; type gain measurable conditioned-on-farm (test n=290 pigs). |
| 3 | Pig recalibration: inference-side threshold/logit-adjust (t1#1) | **DO** (amended) | AUC 0.787 is real headroom, but the cited cRT evidence is **falsified on the current backbone** — ledger 2026-07-07: cRT on softcon eval −0.007, gen −0.020. Threshold moves only; expect pig precision ~0.35–0.45 at recall 0.6 (5:1 imbalance), so net macro-F1 gain is not guaranteed — cheap enough to try anyway. |
| 4 | Hard-negative mining (OSM greenhouse/solar/garage) + download the 9k patch-less NotFarm labels (t1#4+t2#4) | **DO** | Biggest headroom in the system (NotFarm recall 0.319); effect size ≫ blind ±0.020; data already owned. |
| 5 | Capacity/size output via footprint area × stocking density (t2#6, t4#4-part) | **DO** (cheap version) | Direct user value; needs no macro-F1 instrument — validate against US registry capacities. |
| 6 | ESA Earthnet TPM proposal (t3#3) | **DO** (submit early) | 2–3 days effort, free VHR quota; long latency means it should start now regardless of gates. |
| 7 | Footprint/OB2.5D geometry fusion or aux head (t2#1, t3#2, t4#4) | **GATE-ON-PROBE** | t1 *empirically measured* fused gain = **+0.004** in-domain — directly contradicts t2's "biggest typing gain available." But t1's probe used only 5 coarse scalars; barn length/elongation/height (the features PRISM-CAFO says carry 0.74 importance) are untested. Extend probe #1 with OB2.5D shape features first (1–2 days, $0). |
| 8 | Lagoon channel / aux flag (t2#2, t4#5) | **GATE-ON-PROBE** | 52-px lagoons are physically plausible at 10m, but no labels exist and the model's existing AUC 0.787 may already *be* the lagoon signal; test via probe residuals before building. |
| 9 | cRT pass on type head (t4#3) | **TRY** | Minutes; but prior is now mixed-negative on softcon — treat as lottery ticket, not plan. |
| 10 | NAIP/EU free-ortho teacher + VHR typing corpus (t2#5, t3#1) | **TRY after #2** | Real and $0, but US-only for train countries (EU labels are non-train); label-audit value first, typing head second. |
| 11 | Planet R&E free tier (3,000 km²/mo) for 3m pilot | **TRY before any purchase** | Missed synthesis: free quota covers a 10k-site AOI in ~2 months; resolves the t2-vs-t3 contradiction on whether 3m resolves barn geometry (12–18m barn = 3–5 px — genuinely uncertain) at $0. |
| 12 | $5k Satellogic/SkySat pilot (t3#4) | **GATE** on #7+#11 outcomes AND eval expansion | Arithmetic checks out, but list prices, Satellogic "50cm" is super-resolution marketing (native 70cm–1m), and rural archive coverage unverified. |
| 13 | Scale VHR buy $25–100k (t3#5) | **GATE** on #12 | Correctly self-gated. |
| 14 | Cattle class now (any form) | **SKIP** | Unlearnable AND unmeasurable: 130 train / 26 test / 7 eval; per-class F1 CI at n=26 is ~±0.2 — the v5 benchmark's ±0.010 is overall macro-F1 and does not rescue a 26-row class. |
| 15 | Cattle via registries/feedlot data (t1#5) | **GATE-ON-LABELS** (≥1k) | Missed synthesis: the 2026 CONUS feedlot paper's 11.7k labeled + 24k detected facilities is a ready cattle-label source no report connected; feedlot bare-soil is the most 10m-friendly cue — but PRISM-CAFO spatial beef F1 0.16 (at 0.6m!) warns spatial generalization is the failure mode. |
| 16 | Flat 4/5-class softmax retrain (t1#3) | **SKIP** | Direct t1-vs-t4 conflict; t4 wins: dominated by #2, couples tail noise into the 0.855 detection boundary. |
| 17 | OvR multi-label (t4#6) | **SKIP** | t4 itself concedes softmax+marginal dominates; 103 Mixed rows don't justify per-head calibration. |
| 18 | Thermal / hyperspectral / SAR tasking / BlackSky / basemap scraping | **SKIP** | Unanimous across reports; resolution- or detection-limit-blocked; verified. |

### Contradictions & over-claims found

1. **cRT precedent (t1, t4)**: both cite v8 ssl4eo +0.019; neither weights the ledger's softcon cRT **negative** (eval −0.007, gen −0.020, `docs/EXPERIMENTS_LOG.md:35,453`). On the *current best backbone* this lever already failed once.
2. **Footprints**: t2's #1 option vs t1's measured +0.004 fusion gain — the sharpest inter-report conflict; resolved as gate (coarse scalars ≠ shape features).
3. **PlanetScope at 3m**: t2 "aspect ratio measurable" vs t3 "not building geometry" — unresolved; free Planet R&E quota settles it for $0.
4. **Label counts disagree**: t1 (v5) Poultry 12,979/Cattle 231 vs t4 (all_clusters) 9,185/174. Different snapshots; conclusions survive, but pick one canonical count before the retrain config.
5. **t2 internal tension**: "feedlot easiest at 10m" vs its own PRISM-CAFO spatial beef F1 0.16.
6. Verified locally: `_DROP_3` at `training/rachel_to_candidates.py:155` (re-route target confirmed); footprint CSVs exist (5 country files).

### Pricing arithmetic check

All recomputed, correct: 0.64-km chip = 0.41 km², 1.28-km = 1.64 km²; 1k/10k/100k-site products match at all $/km² tiers ($1.6k → $3.8M); min-order per-isolated-site $64 (Satellogic) / $150 (SkySat) / $312 (Pléiades) / $375 (Maxar) → 10k isolated = $0.64–3.75M ✓. Caveats: all list prices pending quotes; clustering overhead (1.5–3×) asserted, not measured against actual farm geography — cheap to verify from candidate coordinates before quoting.

### Measurability audit (post Phase-0/1: blind ±0.020, v5 benchmark ±0.010)

- **Detectable**: #3/#9 (pig F1 shift on test n=290, CI ~±0.06); #4 (NotFarm recall from 0.319 — large effect); #2 detection delta (needs v5 ±0.010; blind ±0.020 is marginal for an expected +0.01–0.03).
- **Requires conditional metrics**: all type options — end-to-end macro-F1 at eval n=524 cannot see them; t4's per-level protocol is mandatory, adopt it as-is.
- **Not measurable by any planned instrument**: cattle per-class F1 (#14) and VHR typing lift at per-class granularity before the eval expansion — t3's own caveat, enforced as gates above.
- **Instrument-free**: #5 capacity (validate vs registries), #6 (proposal outcome).

## Appendix G — Farm-type track: evidence digests (key findings verbatim)

### G.t1-data — Data decomposition (v5 taxonomy, subtype confusion, FP anatomy, footprint probe)

**(A) Taxonomy inventory (v5, 32,050 final labels).** Full final_label counts: NotFarm 13,691; Poultry:Unspecified/Other 7,939; Poultry:Meat Chickens 4,269; Pigs 2,576; **Farm:Unknown 2,171**; Poultry:Eggs 771; **Cattle 231**; Ambiguous 176; Mixed 103; PigsOrPoultry 102; Farm:Other 21. The 3-class mapping drops 2,573 rows, of which **2,397 are confirmed-farm-but-type-unknown** (Unknown/Mixed/PigsOrPoultry/Other) — usable by a hierarchical detection head. standardized→final disagreements: 2,454, mostly standardized=null→final set (2,129) plus 242 NotFarm→Poultry relabels (Rachel overrides). Per-country: Pigs are healthy in train countries (USA 1,206, MEX 405, BRA 170, THA 140, CHL 117 = 2,038) plus ~540 in non-train Europe (ROU 90, ARG 75, CZE 52, HUN 30, POL 29…). **Cattle has only 231 rows globally — MEX 107, USA 72, everything else ≤8.** Nowhere reaches even 500. Farm:Unknown mass is concentrated outside train countries (ARG 1,148, USA 314, ZAF 292, AUS 253).

**(B) Confusion decomposition — the Poultry↔OtherFarm confusion is overwhelmingly PIGS by mass, and BOTH subtypes leak to Poultry.** Joined scored parquets (dedup 133,026) to v5 on candidate_id; true_label==mapped final_label agreement is 1.000 (n=17,253), so the join is exact. True=OtherFarm predicted-class rates (ctx128 / softcon):
- **eval**: Pigs (n=63): 64%/65% predicted Poultry, only 25%/29% correct, 11%/6% NotFarm. Cattle (n=7): 57%/29% →Poultry.
- **test** (larger n): Pigs (n=290): 40% →Poultry, 54% correct. Cattle (n=26): 65%/58% →Poultry, 35%/42% correct, **0% →NotFarm**.
- **blind** (unlabeled+v5 label): Pigs (n=68): 72%/75% →Poultry, only 16%/7% correct. Cattle (n=9): 67%/78% →Poultry.

Reverse direction: true Poultry predicted OtherFarm at 12% (eval softcon) / 5–6% (test) / 6–9% (blind); →NotFarm 4–17%. So in eval the confusion cell is ~half pigs-mispredicted-as-poultry (43 errors) and ~half poultry-as-otherfarm (39–45 errors). Cattle contributes almost nothing by count (n=7 eval) but is proportionally as confused as pigs. Notably farms of any subtype are rarely called NotFarm (≤11%) — detection works; **typing is what fails, and it fails toward Poultry (the majority farm class)**.

**Crucially, the CNN already carries the pig signal**: among farm rows in test/val/eval, ranking pigs-vs-poultry by prob_class2 gives **AUC 0.787** — the argmax just hands pigs to Poultry (class imbalance ~5:1 poultry:pigs in train). This is a calibration/imbalance problem, not a representational one.

**(C) FP anatomy (blind NotFarm, n=1,632; ctx128).** Overall predicted-farm rate 46% (eval 71%, but eval NotFarms are visual-inspection picks with no OSM tag, i.e. pre-screened lookalikes). By OSM category — count (FP rate): residential/apartments 198 (0.37), garages 116 (0.36), military/airfield 81 (0.42), **greenhouses 77 (0.77)**, **solar installations 73 (0.68)**, commercial/retail 45 (0.48), school/university 20 (0.44), industrial 21 (0.57), railway 9 (0.90). Bulk of FP mass = residential/garages; worst per-capita confusers = greenhouses, solar, railway — long thin bright roofs, exactly the poultry-barn template.

**(D) Footprint typing probe (local CSVs found: data/rachel_geometry_candidates/candidates/*.csv, 15,141 rows, 5 train countries, 0% nulls).** GradientBoosting on {num_bldgs, total_area_m2, median_area, template_score_if, mean_area}, leave-one-country-out: **Pigs-vs-Poultry AUC 0.67–0.81 (mean 0.75)**; Cattle-vs-otherFarms AUC 0.75 (USA) / 0.79 (MEX); Farm-vs-NotFarm reference AUC mean 0.84. Top features: median_area, template_score_if, total_area_m2 (pig sites: median bldg 1,529 m² vs poultry 2,090 m²; total 3.9k vs 8.0k m²). **But fusion test shows redundancy in-domain**: on joined rows, CNN prob_class2 alone 0.787, footprints alone 0.663, fused 0.791 — footprints add +0.004. Their value is (a) confirmation that 4 scalar geometry features nearly match the CNN at typing, (b) FP suppression / OOD where Open Buildings covers (BGD/NGA/THA/LatAm), not extra in-domain typing signal.

**(E) 4-class support arithmetic (rows with patches in scored set):** train / val / test / eval / gen: NotFarm 1,986/427/423/135/58; Poultry 6,591/1,162/1,166/319/211; **Pigs 1,384/301/290/63/0**; **Cattle 130/26/26/7/4**. Plus patch-ready but unused: unlabeled-split rows with v5 labels = 2,544 3-class-mappable (1,632 NF / 835 P / 68 pigs / 9 cattle) + 631 FarmUnknownType. Patch coverage of v5: Poultry 79%, Pigs 82%, Cattle 87%, **NotFarm only 34% (9k NotFarm labels lack patches)**, FarmUnknownType 26%.

### G.t2-signal — Typing-signal physics + literature

**Per-type overhead signal table** (min-res = smallest GSD where the cue is reliably usable; all metrics self-reported by paper authors unless noted):

| Type | Discriminative overhead cues | Min res | At S2 10m | At Planet 3m | Sub-meter |
|---|---|---|---|---|---|
| **Poultry** | Clusters of 2–20+ long, narrow, parallel barns; ~12–18 m wide × 120–200 m long (EWG mean barn 24,430 ft² ≈ 2,270 m²); feed bins at barn ends; NO lagoon | ~3 m for barn geometry; 1 m for feed bins/fans | Barn width is 1–2 px — individual barns unresolved; only cluster texture/alignment survives | Footprints + aspect ratio measurable → main poultry discriminator works | Feed bins, tunnel fans, ridge vents → near-certain ID |
| **Swine** | Shorter/wider barn groups + **anaerobic lagoon** (rectangular, pink/brown, adjacent to barns) | 30 m suffices for lagoons; 3 m for barns | Lagoons ARE multi-pixel (see below) — best 10m pig cue | Barns + lagoon shape clear | Full infrastructure |
| **Beef feedlot** | Open dirt pens (many ha), fence-grid texture, mounds, feed mill/silage, **persistent bare soil year-round** | 10 m arguably enough (huge open features) | Large bright bare-soil polygon amid vegetation; annual-median composite preserves it; likely the EASIEST type at 10m | Pen grid visible | Animals countable |
| **Dairy** | Wide freestall barns + feed alley, silage bunkers, lagoons, mixed infrastructure | hard even at 0.6 m | Weak | Partial | Still the hardest class (PRISM-CAFO dairy F1 0.68 at 0.6 m) |
| **Aquaculture** | Rectangular water-pond grids | 10 m | Trivially mappable: Asia-wide/global S2/S1 pond maps, OA 0.83–0.95 ([MDPI 2022](https://www.mdpi.com/2072-4292/14/1/153), [global S2](https://www.sciencedirect.com/science/article/pii/S1569843222002886)) | — | — |

**Swine lagoons are solidly multi-pixel at 10 m.** NC Coastal Plain study of 3,405 lagoons: mean surface 6,600 m², median 5,200 m² (≈50–80 m across = 5–8 S2 px; ≈52 px area); algorithm failed only below 900 m². Detected in 30 m Landsat-5 NIR via changepoint (BinSeg), construction year ±1 yr with ~94% accuracy ([Sci Rep 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8810989/), independent validation on 340 sites). Color: rectangular brown/**pink** (purple sulfur bacteria → elevated red, NIR-dark water → NDWI-positive but red-shifted vs clean water). Older EPA IKONOS 4m work: 76% barns / 79% lagoons ([EPA](https://cfpub.epa.gov/si/si_public_record_report.cfm?dirEntryId=82029&Lab=NERL)). "The prototypical distinguishing feature of a swine CAFO is the lagoon" ([Handan-Nader & Ho 2019, Nat. Sustain.](https://www.nature.com/articles/s41893-019-0246-x); found +15% previously unenumerated NC poultry CAFOs).

**Feedlots (2026 CONUS paper):** YOLOv11m on NAIP 1m, 11,746 labeled feedlots (NE/KS/TX) + 13k negatives → P 0.88 / R 0.85 / F1 0.86 test (self-reported), >24,000 facilities detected ([Sci. Total Environ. 2026](https://www.sciencedirect.com/science/article/pii/S0048969726001117)). Feedlots' cues (tens-of-ha open pens, persistent bare soil) are the most 10m-compatible of any type; temporal bare-soil persistence (annual max-NDVI stays soil-like) is a cheap S2 feature no barn type shares.

**Typing ceiling even at 0.6 m (critical calibration):** [PRISM-CAFO 2026](https://arxiv.org/html/2601.11451) (NAIP 0.6 m, 38k patches, 20 states, YOLOv8+SAM2 infrastructure masks): 4-type F1 random split — swine 0.93, poultry 0.85, beef 0.69, dairy 0.68; **spatial split — swine 0.87, poultry 0.76, dairy 0.58, beef 0.16**. Barn area dominates importance (0.74); manure ponds add signal for swine/beef; poultry is nearly pure barn-geometry. [CAFOSat 2026](https://arxiv.org/pdf/2606.00548) (CVPR-W, 45k NAIP patches, infrastructure-level labels incl. manure ponds) is a ready label/pretraining source. So: your 42% Poultry↔OtherFarm confusion at 10m is physics-consistent, and dairy/beef confusion persists even sub-meter.

**10 m farm detection precedent exists:** Mason 2025 pilot ([Faunalytics summary](https://faunalytics.org/using-satellite-imagery-to-spot-industrial-animal-farms/), self-reported, working paper): CNN on S2, ~10k pig+poultry farms (Iowa/Romania/Chile/Mexico) → P 0.91 / R 0.89 in trained regions; explicit confusers: feedlots, industrial buildings, non-standard/small farms.

**Confusers (published screens):** Plastic greenhouses: S2 spectral indices, PGHI best across 4 world regions ([IEEE JSTARS 2023 benchmark](https://ieeexplore.ieee.org/document/10180103/)), OA ~90%, precision >0.9 for plastic covers ([MDPI 2021](https://www.mdpi.com/2072-4292/13/21/4195)) — cheap index, S2-native. Solar PV: global inventories already exist (Kruitwagen 2021 Nature, S2+SPOT; [Global Renewables Watch 2025](https://arxiv.org/html/2503.14860v1)) — use as an exclusion mask rather than learning it. Industrial sheds: S1 double-bounce + InSAR temporal coherence separates built-up reliably ([20m global building map from S1](https://www.mdpi.com/2072-4292/10/11/1833)); context (no feed bins, road/urban adjacency) is the sub-meter discriminator.

**Beyond-type attributes:** Capacity = footprint area × industry stocking density: EWG mapped 357M birds in NC this way ([EWG 2024](https://www.ewg.org/research/innovative-ewg-study-uses-ai-find-357m-poultry-north-carolinas-factory-farms), self-reported); [Robinson barn dataset](https://github.com/microsoft/poultry-cafos) gives US-wide footprints. Construction year: Landsat changepoint 94% ±1 yr (lagoons); [Temporal Cluster Matching](https://arxiv.org/abs/2103.09787) (Robinson 2021, COMPASS) dates poultry barns (NAIP) and solar farms (S2) without per-year labels. Near-real-time expansion alerts: [Chugg et al. 2021](https://www.sciencedirect.com/science/article/pii/S0303243421001707). **Open Buildings 2.5D Temporal**: annual 2016–2023, effective ~4 m, height MAE 1.5 m, presence/count/height, free in GEE, covers BGD/NGA/THA/LatAm ([Google](https://sites.research.google/gr/open-buildings/temporal/)) — building geometry + growth history exactly where you lack NAIP.

**Marginal sensors, honest verdicts:** (a) **Thermal**: Landsat TIRS 100 m native, ECOSTRESS ~70 m vs 12–18 m-wide barns → fully sub-pixel; I found zero published satellite-thermal livestock-barn work. Verdict: NO for typing. (b) **S1 SAR**: double-bounce flags large metal structures all-weather and adds orientation texture, but no published barn-TYPE separation; verdict: cheap extra channel, presence not type. (c) **Hyperspectral**: EnMAP/PRISMA/EMIT (30–60 m) detect CH4 point sources ~100+ kg/h ([RSE 2025 comparison](https://www.sciencedirect.com/science/article/abs/pii/S0034425725006285)) — landfill/O&G scale; individual barns/lagoons are below limits (California dairy-lagoon CH4 was found with airborne AVIRIS-NG, not satellites). NH3: IASI/CrIS see feedlot hotspots (Tulare, Torreón) only at ~10 km oversampled resolution ([Van Damme 2018, Nature](https://www.nature.com/articles/s41586-018-0747-1)) — regional prior at best. Verdict: none supports per-farm typing.

### G.t3-imagery — Imagery procurement + pricing

**FREE tier covers USA + most of Europe entirely — better resolution than anything you can buy.**
- **USA — NAIP**: aerial 2025 cycle flies ~half the states at 30cm, half at 60cm, refresh ≤3yr; free via AWS Open Data / GEE / USGS EarthExplorer ([fpacbc.usda.gov](https://www.fpacbc.usda.gov/geospatial-services/customer-services/naip-coverage-map), [registry.opendata.aws/naip](https://registry.opendata.aws/naip/), 2025).
- **Europe national orthophotos (10–25cm, open)**: NLD PDOK luchtfoto **8cm** CC-BY; ESP PNOA **25cm** CC-BY 4.0 (10cm regional mosaics even in GEE: [Spain_PNOA_PNOA10](https://developers.google.com/earth-engine/datasets/catalog/Spain_PNOA_PNOA10)); FRA IGN BD ORTHO **20cm** open license; DEU DOP20 **20cm** per-Land (most Länder open, WMS/download, e.g. [MV 20cm](https://data.europa.eu/data/datasets/0dea084c-5d2f-4aa0-a974-481dcd85a0ab)); POL geoportal.gov.pl ortho **free for any purpose** ([geoportal.gov.pl](https://www.geoportal.gov.pl/en/data/orthophotomap-orto/)); CZE ČÚZK ortofoto via open WMS ([data.europa.eu](https://data.europa.eu/data/datasets/cz-cuzk-wms-ortofoto-p?locale=en)); DNK GeoDanmark ortho ~12.5cm open (verify portal); ITA is the weak one — no open national ortho, only patchy regional geoportals. Access method: WMS/WMTS chip-fetch at point + buffer, or bulk tile download.
- **Global South (BGD/NGA/THA/BRA/MEX/PER)**: no free VHR optical. But **Google Open Buildings 2.5D Temporal** (free, GEE, `GOOGLE/Research/open-buildings-temporal_v1`): annual building presence/fractional-count/**height** 2016–2023 at 4m effective res, 130+ countries across Africa, S/SE Asia, LatAm — exactly our no-ortho countries; NOT USA/Europe ([developers.google.com](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1), 2024).
- **NICFI is dead (confirmed)**: no new basemaps after Dec 2024; Norway cancelled next-phase procurement Sep 2025; successor = Planet's paid Tropical Forest Observatory (~$180/mo, 5m — too coarse for typing anyway) ([nicfi.no](https://www.nicfi.no/2025/01/28/nicfi-satellite-data-program-enters-new-phase/), 2025).
- Other free: **Umbra SAR Open Data** (25cm SAR but only ~20 fixed sites — unusable for our points); **EnMAP/PRISMA** hyperspectral free via registration/proposal but 30m GSD — coarser than S2 spatially, marginal for typing.

**Research-program tier — the cheap VHR route for the Global South:**
- **ESA Earthnet Third Party Missions**: free quota of Pléiades, Pléiades Neo, **WorldView** (via EUSI), PlanetScope/SkySat, ICEYE for approved R&D proposals; academics/researchers/qualifying startups; review typically **<2 weeks**; quota valid 1 year, non-commercial use ([earth.esa.int](https://earth.esa.int/eogateway/missions/third-party-missions), [euspaceimaging.com](https://www.euspaceimaging.com/open-access-data/), 2025). Quota is per-proposal (no published cap) — request what you justify; thousands of km² of archive is a normal ask.
- **Planet Education & Research (Basic)**: free, university affiliation required, **3,000 km²/month** PlanetScope download (3–5m; campus license 5,000 km²/mo) ([planet.com](https://www.planet.com/industries/education-and-research/), 2025). Good for activity/change signals, not building geometry.
- **NASA CSDA**: Maxar for NASA-funded investigators; Planet only for US federal + **NSF-funded** researchers ([earthdata.nasa.gov](https://www.earthdata.nasa.gov/about/csda), 2025). Irrelevant unless we get a US grant.
- Academic discounts on commercial buys: **Airbus −50%, Maxar −30%** (LandInfo reseller list, May 2024).

**Commercial $/km² (archive unless noted; LIST prices where marked, else quote-required):**

| Provider | Res | Archive $/km² | Tasking $/km² | Min order | Source/status |
|---|---|---|---|---|---|
| Satellogic | 70cm–1m (50cm SR) | **$4** | $8–10 (rush $23) | 1 tile ≈16 km² / 25–50 km² task | LIST, [satellogic.com](https://satellogic.com/products/multispectral-imagery/) 2025 |
| Jilin-1 (CGSTL) | 75cm / 50cm | $6 / $8 | $12 / $15 | ~25 km² | LIST via [Apollo Mapping](https://apollomapping.com/blog/get-your-updated-price-list-with-new-jilin-1-pricing) 2023 — re-quote |
| Planet SkySat | 50cm | **$6** | $12 flex / $40 assured | 25 km² | LIST, [Planet Select/SkyFi](https://skyfi.com/en/products/planet-select) 2025 |
| Airbus Pléiades | 50cm | $12.50 | $21.25 | 25 km² arch / 100 task | LIST (reseller, [landinfo.com](https://landinfo.com/satellite-imagery-pricing/) May 2024); acad −50% |
| Airbus Pléiades Neo | 30cm | $22.50 | $32.50 | 25 km²; **OneAtlas min AOI 0.1 km²** ([api.oneatlas.airbus.com](https://api.oneatlas.airbus.com/guides/oneatlas-data/g-order-product/)) | LIST 2024; contract needed |
| Maxar/Vantor WV | 50cm / 30cm | $15 / $25.50 | $25 / $32.50 | 25 km² arch; fresh capture min **64 km²** ([Geoimage](https://geoimage.com.au/news/all-new-pricing-vantor-satellite-data) Jan 2025) | LIST 2024, N.Am; acad −30% |
| BlackSky | 35cm–1m | $120–550/scene | from $1,100/task | per-scene | reseller-listed ([eos.com](https://eos.com/find-satellite/blacksky-gen-3/) 2025) |
| Umbra / ICEYE SAR | 16–50cm SAR | ~$450/scene | ~$675/scene; ICEYE quote-only | per-scene | aggregator-listed 2025–26, quote to confirm |

**Arithmetic** (archive, clustered AOIs billed per km²; 0.41 km² chip → ×4 for 1.64 km²):

| Sites | Area (0.41/1.64 km²) | Satellogic $4 | SkySat/Jilin $6 | Pléiades $12.5 (acad $6.25) | Neo/WV-30 ~$23–26 |
|---|---|---|---|---|---|
| 1k | 410 / 1,640 | $1.6k / $6.6k | $2.5k / $9.8k | $5.1k / $20.5k | $9–10k / $37–42k |
| 10k | 4.1k / 16.4k | $16k / $66k | $25k / $98k | $51k / $205k | $94k / $377k |
| 100k | 41k / 164k | $164k / $656k | $246k / $984k | $512k / $2.05M | $0.9M / $3.8M |

**Minimum-order effect is decisive for isolated points**: one chip billed at min = Satellogic ~$64/site, SkySat $150, Pléiades $312 ($156 acad), Maxar-50 $375 → 10k isolated sites = $0.6–3.8M. Mitigations: (a) farms cluster — buy county-scale AOIs covering many sites (expect 1.5–3× area overhead, not 15–60×); (b) OneAtlas pay-per-order 0.1 km² min AOI (Pléiades/Neo) is the only true per-chip channel; (c) one Satellogic 16 km² tile covers ~9 adjacent 640m chips.

**Reality checks**: Google/Esri/Bing basemap scraping remains ToS-prohibited (unchanged — keep skipping). Commercial EULAs (Maxar/Airbus/Planet) generally permit publishing **derived information** (labels, footprints, model outputs) but not imagery chips themselves; Satellogic charges 25–200% uplift for public-release licensing — check license tier before publishing a labeled chip dataset.

### G.t4-arch — Architecture / multi-task methods

1. **Joint multi-head vs MTL folklore.** The negative-transfer danger zone is *unrelated* tasks (Standley et al., "Which Tasks Should Be Learned Together", ICML 2020, arXiv:1905.07553). Farm-type is a strict refinement of farm-detection (nested taxonomy), the best case for sharing. RS-specific coarse+fine two-head networks report gains, with a gradient-control module only to damp cross-granularity noise (Appl. Sci. 12(17):8705, 2022); FAIR1M formalizes hierarchical fine-grained RS recognition (arXiv:2103.05569, 2021).
2. **Special MTL optimizers are not needed.** Kurin et al., "In Defense of the Unitary Scalarization" (NeurIPS 2022, arXiv:2201.04122) and Xin et al. (arXiv:2209.11379, 2022) show tuned unit-weight loss sums + standard regularization match PCGrad (arXiv:2001.06782), GradNorm (arXiv:1711.02257) and uncertainty weighting (arXiv:1705.07115). Mitigation for a 2-head model = sweep one λ and use a higher head LR — not gradient surgery.
3. **Decoupling wins exactly in our regime.** cRT (Kang et al., ICLR 2020, arXiv:1910.09217): learn representation jointly, retrain classifier class-balanced — beats end-to-end rebalancing on long tails; matches our ledger (cRT +0.019 gen on ssl4eo; −0.020 on softcon, both ~noise). Linear-probe beats fine-tuning at very small n (arXiv:2205.07874); Cattle n≈174 (all_clusters count, `docs/CODEBASE.md`) is squarely frozen-embedding territory. Two-stage beats joint when the fine task is tiny/noisy and the coarse model is already good; with a shared backbone, "joint with masked type loss" ≈ "two-stage with shared features," so the real decision is *whether the type gradient may touch the backbone*, not one-model-vs-two.
4. **Partial labels are a 5-line loss.** Classifier-consistent candidate-set loss = marginal likelihood −log Σ_{y∈S} p(y|x) (Feng et al., NeurIPS 2020, arXiv:2007.08929); PRODEN's EM-style within-set reweighting is equally simple (ICML 2020, arXiv:2002.08053). Mapping to our dropped labels: `Farm: PigsOrPoultry` (102) → S={Poultry,Pig}; `Farm: Unknown` (2,171 dropped in v5 per IMPROVEMENT_PLAN_v10) + `Mixed` (103) + `Other` (21) → S=all farm types, which contributes zero type-head gradient — i.e., detection-only supervision. Loss masking and "candidate set = everything" are the same code path. That recovers ~2,400 farm rows for detection plus 102 genuinely informative candidate-set rows.
5. **The CAFO literature types farms via infrastructure, not patch spectra.** Handan-Nader & Ho 2019 (Nature Sustainability) separate poultry vs swine by barn geometry + manure lagoons; Robinson et al. 2022 (arXiv:2112.10988) detect individual barns at 1m; CAFOSat (arXiv:2606.00548, CVPR-W 2026: 45k NAIP patches, 20 states, 4 CAFO categories) annotates barns/manure ponds/grazing features explicitly; PRISM-CAFO (arXiv:2601.11451, 2026) segments infrastructure with priors. All ≤1m imagery; nothing at 10m — consistent with "no published 10m typing baseline." Lagoon presence is *the* pig discriminator in US practice; feedlots (cattle) are large bare-soil signatures — plausibly separable even at 10m.
6. **Auxiliary tasks: cheap guard exists.** Gradient-cosine gating of an auxiliary loss (Du et al., arXiv:1812.02224, 2018) is ~10 lines if an aux task hurts; Standley 2020 shows aux benefits are asymmetric and must be measured. Our candidate CSVs already carry `num_bldgs, total_area_m2, median_area, template_score_if` — free regression targets that distill 4m footprint geometry into the S2 encoder (predict-at-train, not required-at-inference — sidesteps Open Buildings' missing USA/Europe coverage).
7. **Imbalance for the type head** (Poultry ≈9.2k = 6,191 Unspec + 2,382 Meat + 612 Eggs; Pigs 1,994; Cattle 174): literature-standard = train with logit adjustment (Menon 2021 — already implemented in `training/losses.py`) then cRT-rebalance the type head only. No loss trick rescues n=174; the data fix is SEALS-style rare-class harvesting in embedding space (arXiv:2007.00077, AAAI 2022).
