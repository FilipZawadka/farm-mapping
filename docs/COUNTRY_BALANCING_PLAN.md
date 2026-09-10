# Country balancing: data investigation, proposal, pre-registered campaign

*Written 2026-09-06. Companion code: `training/balancing.py` (sampler),
`scripts/audit_country_balance.py` (data audit + what-if), `experiments/gen_balancing_configs.py`
(arms G/H/I/J), `experiments/evaluate_balancing.py` (evaluation), `tests/test_balancing.py`.*

## TL;DR

1. **The training pool is confounded on two sides.** Roughly three quarters of the farm
   positives come from USA + MEX (registry labels); the round_4 absorption of the world
   review pool added ~9k rows that are ~82% NotFarm, and about twenty of those countries are
   **100% NotFarm in the labels** (RUS alone is ~a quarter of all negatives; UKR, BLR, MYS,
   IND, TUR, GBR, KAZ, EGY, ... are all-NotFarm too). A classifier that only knew the country
   would score ~0.85 accuracy against a 0.55 majority baseline. Exact numbers for the volume
   data come from `scripts/audit_country_balance.py` (section 1.4); the figures here are a
   reconstruction from the repo record.
2. **Balancing *where* the epoch comes from does not remove the region→label shortcut, and can
   strengthen it.** Uniform country shares, three buckets, or a per-country cap only move
   region marginals; P(farm | USA)≈0.8 and P(farm | RUS)=0 survive unchanged. The audit shows
   the share-law-only variant leaves label~region dependence untouched (NMI 0.46 → 0.46) while
   shifting the class prior to 58% NotFarm — the prior shift that already sank `v9_bal`.
3. **The anti-shortcut ingredient is class conditioning inside every region group** (Idrissi
   et al. 2022; Sagawa et al. 2020): within each group the sampled class mix follows the global
   prior, the global prior is pinned by raking (so no `v9_bal`-style prior shift), and a
   single-label country forfeits the part of its share it cannot support. All three requested
   experiments carry this term; an optional fourth arm (J) isolates it.
4. **Campaign (round_5, 2026-09-10): four runs at seed 44** — the round_5 baseline A (Rachel's
   split-level cap, no sampler) plus G (grouped countries), H (us / europe / rest) and I (20% cap)
   on the same round_5 data. Round_5 replaced the round_4 data (label fixes, new eval sets, a
   split-level cap), so the round_4 arm A is no longer a valid control — section 8 has the
   details, the run list and the one-command runbook (`scripts/run_round5_campaign.sh`).
   ~$8–12 and ~4 h of wall clock. Single runs support artifact-level claims only; the
   pre-registered rule in section 4 says how they are read. Section 6 lists what has to be in
   place to launch (unchanged: the laptop has everything, the Claude web sandbox has nothing).

---

## 1. What the data looks like

### 1.1 Where the round_4 training rows come from

`Splits (explicit): train=21478 val=5022 test=2094 eval=523 generalization=617`
(`experiments/R4_RUN_NOTES.md` §5). The train split is the union of two very different pools:

| Pool | Rows | Label source | Composition (from the record) |
|---|---|---|---|
| v9 train (round_3) | 12,062 | 80% registry (Farm Transparency), 56% USA | USA 6,762 · MEX 1,614 · THA 575 · BRA 447 · CHL 330 (`notebooks/v6_fullworld_fourclass_analysis`), plus ≤70 promoted rows per country from 106 countries (`scripts/rebalance_splits_v8.py`, round_3). Classes: Poultry 7,616 · Pigs 1,576 · Cattle 153 · NotFarm 2,750 (`docs/EXPERIMENTS_LOG.md`, v9_bal) |
| absorbed `qual_eval` (round_4) | ~9,400 (80% of 11,772) | visual review, 131 countries | The frozen blind benchmark is that pool (`experiments/results/e01_blind_benchmark_frozen.csv`, 11,365 rows): NotFarm 9,338 / Poultry 1,748 / Pigs 273 / Cattle 6 = **82% NotFarm** |

Per-country composition of the absorbed pool (top of the frozen benchmark):

| Country | rows | NotFarm rate | Country | rows | NotFarm rate |
|---|---|---|---|---|---|
| RUS | 2,880 | **1.00** | CZE | 321 | 0.76 |
| DEU | 962 | 0.85 | MYS | 239 | **1.00** |
| UKR | 760 | **1.00** | ROU | 229 | 0.19 |
| AUS | 681 | 0.06 | IND | 223 | **1.00** |
| POL | 619 | 0.49 | TUR | 205 | **1.00** |
| FRA | 542 | 0.38 | GBR | 186 | **1.00** |
| ITA | 506 | 0.70 | KAZ | 152 | **1.00** |
| BLR | 411 | **1.00** | ARG / EGY / ESP / SAU / SWE / ... | 60–120 each | 0.3 / 1.00 / 1.00 / 1.00 / 1.00 |

So after round_4 the train split is roughly **49% NotFarm overall** — pooled, the binary task
looks balanced — but the balance is assembled from countries that are individually extreme:
USA ~80% farm, MEX ~92% farm, AUS ~94% farm, versus RUS/UKR/BLR/MYS/IND/TUR/GBR/KAZ at 0% farm.
Independent confirmation from the v5 label drop: "RUS 2,963 (99% NotFarm)", "new all-Europe
label pools (DEU/ITA/FRA/BLR/CZE…) are ~99% exactly the hard-negative class"
(`docs/IMPROVEMENT_PLAN_v10.md`).

### 1.2 The shortcut, and the evidence the model already takes it

Under this composition the cheapest way to lower the training loss is to recognise *where* a
patch is: Delmarva/Southeast-US barn rows → farm, Russian/Ukrainian steppe → NotFarm. That is
the textbook spurious-correlation set-up (Waterbirds' land/water background in Sagawa et al.
2020; Geirhos et al. 2020). Four pieces of in-repo evidence say the model does use region cues:

- **Round 2**: 962 NotFarm corrections from 106 countries were added with no positives. AUC did
  not move (0.777 → 0.772), recall at matched FPR was identical, and mean P(farm) fell for
  *both* true classes — the model learned "unfamiliar country ⇒ not a farm"
  (`docs/EXPERIMENT_COMPENDIUM.md` Part 4).
- **Seed variance is cross-country score drift**: per-country score offsets, not within-country
  ranking, dominate the OOD seed spread; within-country all round_4 arms sit at 0.83–0.85
  (`experiments/R4_RUN_NOTES.md` §10).
- **Regional collapse on the blind slice**: RUS binary farm F1 0.01 (n=438, model called farm
  on ~half of confirmed non-farms), UKR 0.0, MYS 0.04 — before those negatives entered train
  (`docs/IMPROVEMENT_PLAN_v10.md`). Now that they are in train, the risk flips: the model can
  learn the region prior instead of the object.
- **`v9_bal`**: class-balanced sampling shifted the prior (NotFarm ×2.75 weight) and cost −0.054
  generalization macro-F1 (≈7σ) — "the model turns cautious exactly where it is least
  informed". Any new sampler must leave the class prior alone.

Why the obvious fix was closed: "26 train countries have ≤2 rows and `_compute_region_weights`
would draw a 1-row country ~100×/epoch" (`paper/experiments_justification_plan.md`, closed
axes). The three schemes below are exactly the ways around that — pool the small countries,
coarsen to buckets, or only cap the large ones — and every per-row weight is clipped.

### 1.3 Reconstructed audit (synthetic stand-in for the volume data)

`scripts/audit_country_balance.py` run on a synthetic pool built from the counts above
(`tests/test_balancing.py::R4_LIKE`, 17,814 rows, 34 countries). **Illustrative only — rerun
on the volume for the real table.**

```
labelled rows: 17,814 across 34 countries | NotFarm 8,114 / farm 9,700 | farm rate 0.545
where the positives come from: USA 57.7%, MEX 15.5%, AUS 5.3%, THA 4.3%, BRA 3.3%, CHL 2.8%
where the negatives come from: RUS 28.3%, USA 17.3%, DEU 8.1%, UKR 7.5%, BLR 4.1%, ITA 3.5%, POL 3.0%
single-label countries: 18 all-NotFarm (4,449 rows = 54.8% of all negatives)
NMI(label; country) = 0.476   NMI(label; macro-region) = 0.316
accuracy of a country-only classifier = 0.846  vs majority class = 0.545  (shortcut gain +0.30)
```

What each sampler does to that pool (same script, `--schemes all`):

| Arm | Sampler | groups | label~region NMI natural → sampled | class prior (NotFarm) natural → sampled | ESS ratio | max weight |
|---|---|---|---|---|---|---|
| G | grouped countries, uniform, class-cond. | 18 | 0.458 → **0.222** | 0.456 → 0.471 | 0.31 | 10 (0.8% rows clipped) |
| H | us / europe / rest, class-cond. | 3 | 0.244 → **0.000** (bucket level; country level +0.30 → +0.20 shortcut gain) | 0.456 → 0.456 | 0.67 | 3.2 |
| I | 20% cap per country, class-cond. | 34 | 0.476 → **0.263** | 0.456 → 0.457 | 0.49 | 10 (0.01% clipped) |
| J | grouped, **no** class-cond. (ablation) | 18 | 0.458 → 0.456 | 0.456 → **0.581** | 0.40 | 10 |

ESS ratio = Kish effective sample size / n: 0.31 means an epoch of arm G carries the information
of ~31% of the rows drawn uniformly (rows from small groups repeat). That is the cost of uniform
country shares; the cap (I) is the mildest intervention and buckets (H) the cheapest in ESS.

### 1.4 Get the real numbers

On any machine with the volume mounted (or the candidates dir synced):

```bash
# composition of the round_4 train split + what-if for all four samplers
python scripts/audit_country_balance.py \
    --config configs/rachel_clusters/world_v10_fourclass_r4.yaml --schemes all \
    --json experiments/results/r4_train_country_audit.json
# straight from Rachel's master parquet
python scripts/audit_country_balance.py \
    --parquet data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet --schemes all
```

Read: the per-country table ('*' marks single-label countries), the "shortcut gain" line, and
for every scheme the `nom%/tgt%/ach%` columns, the NMI change, and the ESS ratio. If the real
ESS ratio of arm G comes out below ~0.25, switch G to `temperature: 2` (square-root shares,
section 6) before launching.

---

## 2. What the literature says about this situation

**Spurious correlations / group shift.** Sagawa et al. (2020) formalise the problem as
worst-group risk and show that group-DRO needs group labels plus strong regularisation.
Idrissi et al. (2022) then show that *simple data balancing* — subsampling or reweighting so
that every (class × attribute) group has equal mass — matches group-DRO on worst-group accuracy,
and that balancing on the **joint** of class and attribute is what matters, not either
marginal. Kirichenko et al. (2023, DFR) show that retraining only the last layer on a small
group-balanced held-out set removes most reliance on the spurious feature — a cheap follow-up
here because cRT machinery already exists (`training.resume_reset_epoch`, `world_v8_crt`).
Byrd & Lipton (2019) warn that *loss* reweighting washes out over training for separable data;
sampling (drawing rows with the target frequency, as `WeightedRandomSampler` does) and early
stopping on val F1 keep the effect, which is what this pipeline does.

**Domain generalisation.** DomainBed's ERM baseline draws each minibatch with equal numbers from
every training domain (Gulrajani & Lopez-Paz 2021), and most published DG methods fail to beat
it; on WILDS the geographic-shift tasks (FMoW-wilds regions, PovertyMap-wilds countries) show
the same picture, with reweighting buying modest worst-region gains (Koh et al. 2021). Wiles et
al. (2022) find that data-side interventions (augmentation, target-like data) beat method-side
ones. Domain-balanced batching is arm H at its coarsest and arm G at country granularity.

**Data-mixture balancing at scale.** Long-tailed sources are routinely tempered rather than
equalised: square-root sampling of hashtag frequencies beat both natural and uniform
(Mahajan et al. 2018); massively multilingual NMT samples languages with temperature T≈5
(Arivazhagan et al. 2019) and XLM-R with exponent α=0.3 (Conneau et al. 2020); per-source caps
are standard in pretraining mixtures. `region_balancing.temperature` implements the tempered
law (T=1 natural, T=2 square-root, null = uniform); the cap is arm I.

**Class imbalance.** Repeating 153 Cattle images 20×/epoch memorised sites, not the concept
(`v9_bal`); the "effective number of samples" argument (Cui et al. 2019) says the marginal value
of the k-th repeat decays fast, hence the `max_weight` clip and the ESS diagnostic. Logit
adjustment (Menon et al. 2021; `training.loss: logit_adjusted`) and decoupled classifier
retraining (Kang et al. 2020) are the loss-side and two-stage alternatives; a group-conditional
logit adjustment is the natural loss-side twin of arm G/H/I (section 6).

**Remote sensing specifics.** Random splits inflate scores under spatial autocorrelation
(Roberts et al. 2017; Ploton et al. 2020) — the generalization slice is already a
leave-country-out block design, which is why it is the primary slice here. Rolf et al. (2024)
argue geographic imbalance is a defining property of satellite datasets and that models
should be evaluated per region, not pooled; our within-country AUC and per-bucket diagnostics
follow that. Cross-region crop mapping (Kerner et al. 2020) reports the same failure mode of
models trained on data-rich regions transferring poorly to data-sparse ones.

**What this implies for the design.**
1. Balance the joint (region × class), not the region marginal alone.
2. Keep the class prior fixed — the one intervention this project measured to hurt OOD.
3. Prefer down-weighting the dominant cells over repeating rare rows; clip; report ESS.
4. Evaluate per region and at a fixed operating point, not only pooled AUC.

---

## 3. The proposal: three experiments (plus one optional ablation)

All arms share: round_4 data (`candidates_world_v10_r4`, `all_clusters_v10.parquet`), the
arm-A recipe byte-for-byte (SoftCon ResNet-50, 9 channels, 64 px crop, lr 1e-4, freeze 5,
50 epochs, `val_f1` checkpointing), **seed 44**, `runpod.github_branch: develop`. The only
delta is `training.region_balancing`. The control is the existing round_4 run
`world_v10_fourclass_r4_a_s44` (same recipe, same seed, so the same weight init and augmentation
stream; already collected). Arm A's other two seeds (42, 43) are used only to measure σ_seed.
Configs: `world_v10_fourclass_r4_{g,h,i}_s44.yaml`; the fleet order is
`experiments/balancing_order_r4.txt`.

| Arm | Config block | What it does to an epoch | Hypothesis |
|---|---|---|---|
| **G** grouped countries | `scheme: grouped_country, min_country_rows: 300, class_conditional: true, max_weight: 10` | Every country with ≥300 train rows is its own group; smaller ones pool into `rest_<macro-region>` (UN M49: europe, asia, africa, latin_america, north_america, oceania). Groups get equal shares, single-label groups keep only the p(NotFarm) part of theirs, the class mix inside every group follows the global prior, weights clipped to [0.1, 10]. | The most aggressive de-confounding at country granularity; largest gain on within-country ranking in non-focal countries; largest ESS cost, so in-domain (`test`) may dip. |
| **H** three buckets | `scheme: bucket, buckets: {us: [USA], europe: [EUROPE], rest: ["*"]}, class_conditional: true` | One third of the epoch from each bucket; inside each bucket the class mix is the global prior (Europe's ~1,000 positives are up-weighted against its ~5,000 negatives; US negatives are up-weighted against US positives). Mildest ESS cost. RUS sits in `europe` (M49); move it to `rest` in the config if the visual-domain argument for Eastern Europe is not wanted. | Removes the coarse "European-looking ⇒ NotFarm / US-looking ⇒ farm" prior at low cost; leaves country-level cues inside Europe. |
| **I** per-country cap | `scheme: capped, max_share: 0.20, class_conditional: true` | Natural shares except no country above 20% of an epoch; the excess is redistributed pro rata (water-filling). Only USA (~33–39%) is touched by the cap on round_4 data; RUS drops from ~13% to ~10% through the class-availability rule, not the cap. | The mildest intervention (ESS ~0.5): if this already matches arm A on generalization while narrowing the region-driven score offsets, it is the production candidate. |
| **J** optional ablation (not generated by default; `gen_balancing_configs.py --with-ablation`) | as G but `class_conditional: false` | Uniform group shares with each group's natural class mix; class prior drifts to ~58% NotFarm. | Isolates the anti-shortcut term: if J ≈ G on the diagnostics, the share law is doing the work; if G ≫ J, class conditioning is. |

Mechanics common to all arms (`training/balancing.py`): one epoch is still N draws with
replacement; weights are normalised to mean 1, so `w_max = 10` reads "drawn at most 10× the
average rate"; the run writes `sampling_report.json` (natural / nominal / target / achieved
shares per group, class marginal, NMI before/after, ESS, clipping) next to `best_model.pt`, logs
it, and pushes the headline numbers to MLflow, so an evaluation can *verify* an arm sampled
balanced rather than trust the flag. `train.py` refuses to start a run whose config asks for
balancing but whose dataset carries no weights.

Things deliberately **not** changed, for attribution: epochs (ESS drop means fewer distinct
rows per epoch; a fair "more epochs" arm would be a separate lever), class weights, loss, the
4-class head (conditioning acts on the binary farm axis; the 4-class mix inside each
(group, family) cell stays natural).

---

## 4. Pre-registration (fixed before any campaign score is inspected)

**Design.** 3 single-lever arms (G, H, I), **one run each at seed 44**, against round_4 arm A.
Same estimands and statistics as `experiments/EVAL_METHODS.md`: `dAUC_rec` (difference of mean
per-seed AUC; for a single-run arm, its AUC) is primary, `SE_total = sqrt(SE_boot² +
σ_seed²(1/n_a + 1/n_arm))` with σ_seed measured per slice from arm A's three seeds (n_a = 3,
n_arm = 1), Holm across the confirmatory family, practical floor 0.005 AUC.

**What one run per arm can and cannot claim.** With n_arm = 1 the contrast is artifact-level —
"checkpoint G ranks these rows better than checkpoint A" (Dietterich's Q3) — not recipe-level
(Q8). The seed term is still in `SE_total`, so a delta inside arm A's own seed band prints as
"not distinguishable"; round_4 showed that band is ±0.02–0.06 AUC on generalization. Expect the
primary contrasts to be underpowered; the diagnostics below (bucket spread, per-country FPR,
within-country AUC) are where a single run can show a qualitative change. If an arm looks
promising, promote it to three seeds (`gen_balancing_configs.py --seeds 42 43 44`) before any
production decision — never adopt from a single run (the freeze0 lesson).

**Primary endpoint.** Binary farm ROC-AUC on `generalization` (BGD / NGA / ALB / COD / IND / MAR;
zero training rows, ~116 km median distance to train).
**Confirmatory family (Holm, m=3):** `g > a`, `h > a`, `i > a`.
A win = point estimate > 0.005 **and** Holm p < 0.05. A loss beyond the floor with p < 0.05 is
a rejection of that arm. Anything else is "not distinguishable below Δ = MDE₈₀" — never "no
difference" (round_4 MDE₈₀ on this slice was ~0.05 with 3 seeds).

**Secondary (exploratory, unadjusted), all in `experiments/evaluate_balancing.py`:**
- `j` vs `g` and `j` vs `a` (what class conditioning adds); `test` / `eval` slices.
- Within-country AUC on generalization (mean per-country AUC per seed → arm mean ± sd): ranking
  skill with the cross-country calibration drift removed.
- Per-bucket behaviour (us / europe / rest) on `val`, `test`, `generalization`: AUC, mean
  P(farm | NotFarm), mean P(farm | farm), FPR@0.4, recall@0.4. **The shortcut signature is the
  spread of mean P(farm | NotFarm) across buckets** (US negatives scored high, European negatives
  scored low). A balanced arm should narrow it.
- Per-country FPR@0.4 on `val` for the NotFarm-dominated countries (RUS, UKR, BLR, DEU, MYS, ...)
  and AUC where both classes exist (POL, ROU, HUN, CZE, FRA, ITA, DEU). `val` chose the
  checkpoints, so it is optimistic for every arm alike; it is the only held-out slice with
  European / Russian rows now that qual_eval was absorbed.
- Calibration (ECE) per slice, diagnostic only.
- `sampling_report.json` per run: achieved NMI, ESS, clipping — the arm must show it sampled
  balanced.

**Decision rule for adoption.**
1. An arm that wins its confirmatory contrast is a production candidate; publish per the
   round_4 selection rule (best seed of the winning arm, arm mean in the release notes).
2. If every confirmatory contrast is "not distinguishable", an arm is still a candidate for the
   next production run if it (a) is within the floor of arm A on generalization, (b) is
   within-country ≥ arm A, and (c) narrows the P(farm | NotFarm) bucket spread and does not
   raise farm FPR@0.4 in RUS/UKR/BLR val rows or lower recall@0.4 in the mixed European
   countries. Prefer I over H over G at equal evidence (least intervention, highest ESS).
3. An arm that loses on generalization beyond the floor is rejected regardless of diagnostics;
   an arm whose only gain is on `test`/`eval` is not adopted (the `v9_bal` lesson).

**Voiding conditions.** A winning arm whose advantage rests on a single generalization country
(check the per-country table), or whose ECE gap to arm A exceeds 0.05, is escalated with the
table instead of adopted. A run whose `sampling_report.json` is missing or whose achieved NMI
equals the natural NMI did not sample balanced and is excluded, not silently included.

**Known confounds, stated up front.** (i) All arms bundle a share law with class conditioning;
only the optional arm J decomposes G. (ii) ESS differs by arm (G ≈ 0.3, I ≈ 0.5, H ≈ 0.7), so
"same epochs" is not "same number of distinct rows seen". (iii) The `val` diagnostics are on the
checkpoint selection slice. (iv) One run per balanced arm: the seed band comes from arm A alone
and the comparison is artifact-level (above).

**Cost.** 3 runs at ~3 h on an RTX 4090 (~$0.34–0.74/h) plus, optionally, one full-world
scoring pass each ≈ **$5–10 total**, ~3 wall-clock hours at 3 concurrent pods (plus the 30-min
idle watchdog per pod unless the collector's reaper is running).

---

## 5. How to run it

*This is the round_4 flow (three arms against the existing round_4 arm A). For the live
round_5 campaign use `bash scripts/run_round5_campaign.sh` — section 8.4.*

```bash
git checkout develop && git pull                     # the pod runs the code STAGED from this tree

# 0. (optional but recommended) the real audit, then adjust min_country_rows / temperature
python scripts/audit_country_balance.py --config configs/rachel_clusters/world_v10_fourclass_r4.yaml --schemes all

# 1. configs (already committed; regenerate after editing ARMS) + self-test
python3 experiments/gen_balancing_configs.py --selftest

# 2. launch the three runs; separate state file so the round_4 fleet state is untouched
python3 experiments/launch_fleet.py --order-file experiments/balancing_order_r4.txt \
    --state experiments/results/balancing_fleet_state.json --max-concurrent 3 --budget 10
#    (or one at a time: python -m training.runpod_launch \
#        --config configs/rachel_clusters/world_v10_fourclass_r4_g_s44.yaml --steps train inference)

# 3. collect while the pods are alive (also reaps finished pods; pulls sampling_report.json)
python3 experiments/collect_results.py --watch --names-file experiments/balancing_order_r4.txt

# 4. (optional) full-world scoring passes, for complete evaluation slices + publishing
python3 experiments/gen_score_configs.py             # picks up world_v10_fourclass_r4_{g,h,i}_s44
python3 experiments/launch_fleet.py --order-file <file with the *_score names> --steps inference \
    --state experiments/results/balancing_score_state.json

# 5. evaluate against round_4 arm A (needs experiments/gpu_results/world_v10_fourclass_r4_a_s4?/
#    and data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet locally)
python3 experiments/evaluate_balancing.py            # -> experiments/results/balancing_evaluation.json
```

Verify locally first: `python tests/test_balancing.py` and
`python tests/test_evaluate_balancing_smoke.py`. Every generated config sets
`runpod.github_branch: develop`, but note that the pod actually runs whatever
`training/runpod_launch.py::_stage_code` copies from the launching machine's working tree, so
the checkout that launches must contain `training/balancing.py` (i.e. be on `develop` at or
after commit `e261706`).

---

## 6. Launch checklist — what has to be in place

Verified on 2026-09-06 from the Claude Code web container: `api.runpod.io` is **blocked** by the
sandbox proxy (HTTP 403 on CONNECT), there is no `RUNPOD_API_KEY` and no SSH key, only
`RUNPOD_NETWORK_VOLUME_ID`. So the three runs cannot be started from that environment as
configured. Two ways forward:

**A. Launch from the laptop (no setup, ~2 minutes of typing).** Everything the fleet needs is
already there: `.env` with `RUNPOD_API_KEY`, the SSH key RunPod knows, the round_4 outputs under
`experiments/gpu_results/`, and `all_clusters_v10.parquet`. Steps: `git pull` on `develop`, then
sections 5.1–5.3 above. That is the recommended path.

**B. Make the web environment able to launch** (`docs/CLAUDE_WEB_SETUP.md` §1–3), in which case
the launch can be driven from a Claude session:
1. Network policy of the environment must allow `api.runpod.io` (HTTPS) **and** outbound SSH to
   RunPod pod IPs on arbitrary high ports — `launch_pod` stages the code and starts the run over
   SSH. If the sandbox only allows HTTPS, pods can be created but never started.
2. Secrets: `RUNPOD_API_KEY`, and `RUNPOD_SSH_PRIVATE_KEY` whose public half is registered in
   RunPod → Settings → SSH Public Keys (generate a dedicated key; §2 of that doc).
3. `bash scripts/bootstrap_cloud.sh` to materialise `.env`, the key, and check connectivity.
4. For the evaluation step, the arm-A scored parquets and `all_clusters_v10.parquet` must be
   present locally (`bootstrap_cloud.sh --pull-data` from a live pod, or upload).

**Independent of where it is launched:**
- The network volume `r8nyom4e4e` (EU-RO-1) must still hold `data/patches/` (the round_4 patch
  store), `data/rachel_geometry_candidates/candidates_world_v10_r4/` and
  `all_clusters_v10.parquet`, and `data/output/world_v10_fourclass_r4_a_s44/` — nothing in the
  repo indicates they were removed since the round_4 campaign, but check with
  `python3 experiments/launch_fleet.py --status` + a look at the volume before spending.
- Budget: ~$5–10 for the three runs at RTX 4090 prices; the launcher's `--budget` reserve stops
  launches below that balance.
- GPU stock in EU-RO-1 (RTX 4090 with L4 fallback, already in the configs).
- Keep `collect_results.py --watch` running so finished pods are reaped; `auto_terminate`
  on the pod never fires (`experiments/R4_RUN_NOTES.md` §6).

---

## 7. Follow-ups not in this campaign

- **Three seeds per arm** (`gen_balancing_configs.py --seeds 42 43 44`) for whichever scheme
  looks promising: the only way to make a recipe-level claim and the precondition for adopting
  a sampler in production.
- **Arm J** (`--with-ablation`): grouped countries without class conditioning, to attribute a
  G effect to the share law or to the class term.
- **Tempered shares (T=2, square-root)** as arm K if uniform country shares (G) prove too
  aggressive (ESS < 0.25 on the real audit, or a `test` regression): `temperature: 2.0`.
- **Group-conditional logit adjustment** (loss-side twin): subtract τ·log p(class | group) from
  the logits during training so region priors earn no credit, with no row repetition at all.
  Needs the dataset to return a group id per row; a small change to `PatchDataset.__getitem__`
  and `losses.py`.
- **Last-layer retraining on a group-balanced `val` (DFR)**: reuse `resume_reset_epoch` / cRT
  with the arm-A checkpoint and a (bucket × class)-balanced val subset. Zero-GPU-hour-scale.
- **Positives where there are none.** RUS, UKR, BLR, MYS, IND, TUR, GBR, KAZ have zero farm
  positives in the labels. No sampler can teach "farm in Russia"; only labels can. The
  per-country FPR table from `evaluate_balancing.py` is the shortlist for that label round.
- **Spatially blocked splits and per-country thresholds** remain open items from
  `paper/experiments_justification_plan.md` (E0.2, E2.1) and compose with any sampler.

---

## 8. Round 5 — Rachel's split-level cap (2026-09-10)

### 8.1 What changed in the data

From Rachel's notes (relayed 2026-09-10). Two deliveries: the tidied `for_analysis` files, and
the `round_5` files built from them.

| Change | Consequence for this campaign |
|---|---|
| ~30 bad labels (from the round_2 false-positive review) corrected in the `for_analysis` files | every round_5 model starts without known bad labels; round_4 models did not |
| Label-group definitions made consistent when building evaluation sets | `eval` composition changes "a bit" — round_4 `eval` numbers are not a reference for round_5 |
| All BGD clusters are `split=generalization` | the generalization slice grows; BGD rows that leaked into other splits are gone |
| Three new fully held-out generalization countries: **PER, IDN, MOZ** | do not use them for selection or thresholds. In round_4 their labelled rows sat in train/val (qual_eval was absorbed), so **round_4 models are contaminated on them** — another reason the round_4 arms cannot be compared on round_5 slices |
| Eval clusters that had dropped out for lack of a valid Sentinel patch are restored | the evaluator's slices come from the parquet; rows without a patch are still dropped by `build_splits` — compare its "Explicit splits" log counts with the parquet's split counts to see how many |
| **Split-level cap**: HICs may contribute at most 1.3× the LMIC total in the Poultry and NotFarm classes; per-country ceilings for HICs — NotFarm 789 (DEU 879→789, RUS 2,960→789, USA 2,096→789), Poultry 1,205 (USA 6,377→1,205); Pigs/Cattle uncapped; 20,435 of 29,175 eligible clusters kept; excluded rows carry `split=qual_eval` | see 8.2 |
| Same train/val/test proportions for the focal countries, 80:20 train/val elsewhere | **the test set differs from round_4** — never compare round_5 `test` to round_4 `test` |
| `generalization` and `eval` membership preserved (apart from the fixes above) | the primary slice is stable within round_5 |

### 8.2 What Rachel's cap does and does not do to the confound

Her cap is a *class-conditional, split-level* cap: a per-class budget (HIC ≤ 1.3 × LMIC) with
per-country ceilings inside it. It removes mass exactly where section 1 said the dominant
cells were — USA poultry and RUS/DEU/USA NotFarm — and it does so by *dropping rows*, not by
re-weighting, so no row is repeated. Three things follow:

1. **The round_4 baseline is dead as a control.** Different labels, different eval sets, a
   different test set and 8,740 fewer training rows. The control for the round_5 arms is a
   round_5 baseline with the same recipe and no sampler: `world_v10_fourclass_r5_a_s44`.
2. **The 20% sampler cap (arm I) almost certainly does not bind any more.** After her cap USA
   is roughly 15% of the train split (789 NotFarm + 1,205 Poultry + uncapped Pigs/Cattle) and
   RUS about 4%. Arm I therefore reduces to *natural shares + class conditioning* — which is
   useful: on round_5 it isolates the class-conditional term on top of Rachel's cap. The run's
   `sampling_report.json` states whether the cap bound (`nominal_share` vs `natural_share`).
3. **The within-country confound survives her cap.** The cap changes how much of the epoch
   each country contributes, not what its rows say: RUS, UKR, BLR, MYS, IND, TUR, GBR, KAZ are
   still 100% NotFarm, USA is still ~75% farm. That is the part the class-conditional sampler
   addresses (section 1.3), so arms G and H still have a job. The audit stage of the runbook
   prints the round_5 numbers (`experiments/results/r5_train_country_audit.txt`).

Also worth knowing: the 8,740 excluded rows now carry `split=qual_eval`. For round_5 models
that is a large never-trained-on slice, but it is the *capped-out remainder of the training
countries* (USA poultry, HIC NotFarm), not a world review sample — report it, do not treat it
as deployment-like, and do not compare it with any earlier `qual_eval` figure. `evaluate_r4.py`
and `evaluate_balancing.py` never read it.

### 8.3 Run list

| Run | Sampler | Role |
|---|---|---|
| `world_v10_fourclass_r5_a_s44` | none | control; also the "does Rachel's cap help?" model (vs round_4 arm A on the *shared* generalization countries only: BGD, NGA, ALB, COD, IND, MAR) |
| `world_v10_fourclass_r5_g_s44` | grouped countries, uniform, class-cond. | arm G on round_5 |
| `world_v10_fourclass_r5_h_s44` | us / europe / rest, class-cond. | arm H on round_5 |
| `world_v10_fourclass_r5_i_s44` | cap 20% (likely non-binding) + class-cond. | class-conditional term alone |

All four: seed 44, recipe of round_4 arm A, `candidates_world_v10_r5` built from
`all_clusters_v11.parquet`, `runpod.github_branch: develop`. Configs come from
`experiments/gen_balancing_configs.py` (round_5 is the default; `--round r4` rebuilds the
round_4 arms), fleet order `experiments/balancing_order_r5.txt`. Evaluation:
`experiments/evaluate_balancing.py --prefix world_v10_fourclass_r5 --v10 <all_clusters_v11.parquet>`
(both are the defaults). The pre-registered contrasts of section 4 apply unchanged (g>a, h>a,
i>a on generalization AUC; the seed term falls back to the v9 five-seed σ because no round_5
arm has more than one seed — the evaluator says so in its output).

### 8.4 Getting the data onto the volume and running

The pods do not have Rachel's files: her Drive folder is synced to the network volume with
`scripts/sync_rachel.sh` (Drive → local staging → rsync over SSH), the per-country files are
merged into one parquet with `scripts/merge_clusters_v7.py` (it asserts cluster_id stability
and zero geometry drift against v10 before joining, and carries `viz_status` / `viz_label` /
`template_score_if` over), and the candidates dir is built by the candidates step of the first
run. `scripts/run_round5_campaign.sh` strings this together from the laptop:

```
sync → merge → upload → audit → baseline (with candidates step) → wait → arms → collect → evaluate
```

Every stage is idempotent and can be run alone (`bash scripts/run_round5_campaign.sh audit`);
`--dry-run` prints the commands. Two details matter:

- **`wait` exists because the candidates dir is shared.** `rachel_to_candidates.convert()` writes
  ~165 country CSVs; a training pod that starts while another pod's candidates step is still
  writing reads a partial directory and trains on a subset of countries *without any error*.
  `convert()` now writes `_COMPLETE.json` when every CSV is in place, and the runbook launches
  the balanced arms only after that marker exists on the volume.
- **The pods run the code staged from the launching working tree**, so the laptop must be on
  `develop` with `training/balancing.py` and the round_5 configs present.

Nothing in this section can run from the Claude Code web sandbox (section 6): it cannot reach
`api.runpod.io`, has no RunPod API key or SSH key, and has no Google Drive credentials for the
sync. The sandbox was used to prepare and test the tooling; the laptop launches.

### 8.5 Points to raise with Rachel

- **Russia counted as a HIC.** The World Bank lists the Russian Federation as upper-middle
  income, i.e. LMIC under the usual split; her cap treats it as HIC (ceiling 789). Deliberate?
  It is a reasonable choice for the *purpose* (it is the largest all-NotFarm source), but the
  rule should say so.
- **The cap leaves the region→label shortcut inside single-label countries untouched** (8.2,
  point 3); the sampler arms are the complement, and the class-conditional term is the part
  that does not double-count her cap.
- **`qual_eval` changed meaning** (capped-out training-country rows, no longer a world hold-out).
- **New generalization countries need patches**: PER/IDN/MOZ rows without an extracted patch
  are silently dropped from the evaluation slices; the `Explicit splits` log line of the
  baseline run gives the effective counts.

---

## References

- Arivazhagan, N. et al. (2019). Massively multilingual neural machine translation in the wild: findings and challenges. arXiv:1907.05019.
- Byrd, J. & Lipton, Z. (2019). What is the effect of importance weighting in deep learning? ICML.
- Conneau, A. et al. (2020). Unsupervised cross-lingual representation learning at scale. ACL.
- Cui, Y. et al. (2019). Class-balanced loss based on effective number of samples. CVPR.
- Deming, W. E. & Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table when the expected marginal totals are known. Ann. Math. Stat. 11(4).
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence 2.
- Gulrajani, I. & Lopez-Paz, D. (2021). In search of lost domain generalization. ICLR.
- Idrissi, B. Y., Arjovsky, M., Pezeshki, M. & Lopez-Paz, D. (2022). Simple data balancing achieves competitive worst-group-accuracy. CLeaR.
- Kang, B. et al. (2020). Decoupling representation and classifier for long-tailed recognition. ICLR.
- Kerner, H. et al. (2020). Rapid response crop maps in data sparse regions. KDD Humanitarian Mapping Workshop.
- Kirichenko, P., Izmailov, P. & Wilson, A. G. (2023). Last layer re-training is sufficient for robustness to spurious correlations. ICLR.
- Kish, L. (1965). Survey Sampling. Wiley. (effective sample size)
- Koh, P. W. et al. (2021). WILDS: a benchmark of in-the-wild distribution shifts. ICML.
- Liu, E. Z. et al. (2021). Just train twice: improving group robustness without training group information. ICML.
- Mahajan, D. et al. (2018). Exploring the limits of weakly supervised pretraining. ECCV.
- Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. ICLR.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nature Communications 11.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40.
- Rolf, E., Klemmer, K., Robinson, C. & Kerner, H. (2024). Mission critical — satellite data is a distinct modality in machine learning. ICML.
- Sagawa, S., Koh, P. W., Hashimoto, T. B. & Liang, P. (2020). Distributionally robust neural networks for group shifts. ICLR.
- Wiles, O. et al. (2022). A fine-grained analysis of distribution shift. ICLR.
