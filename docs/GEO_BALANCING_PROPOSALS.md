# Geolocation-based balancing: what the literature suggests we try next

**Status:** proposal, 2026-09-11. Written after the round_5 campaign, before any of
these has been run.

## Where we are

Round_5 tested three *discrete* region-balancing schemes on the round_5 labels
(HIC ≤ 1.3× LMIC split cap, 9 fully held-out generalization countries), one seed
each, against a no-sampler control (`experiments/results/balancing_evaluation.json`):

| Arm | Grouping unit | Gen. AUC (n=915) | vs control | p_holm |
|---|---|---|---|---|
| H | 3 buckets: us / europe / rest | 0.8657 | +0.031 | 0.060 |
| I | per-country cap 20 % | 0.8607 | +0.026 | 0.102 |
| G | grouped countries (min 300 rows), uniform | 0.8537 | +0.019 | 0.164 |
| A | none | 0.8343 | — | — |

Three schemes moving the same direction is encouraging; none clears the seed
band (±0.023 for single runs), and all three still leave a measurable geographic
shortcut in the training set — a region-only classifier beats majority class by
**+0.254** before balancing and **+0.214** after (`r5_train_country_audit.txt`).

Every scheme so far uses **country (ISO3) as the unit**. That is coarse in two
directions at once: a country is not a homogeneous domain (Brazil's south and
Amazon frontier are different farm landscapes), and countries are not comparable
units (Luxembourg and India get one bucket each). The obvious next step is to let
the *coordinates themselves* define the balancing — which is what most of the
adjacent literature does.

## What comparable work does with coordinates

Four families, roughly in order of how directly they map onto our sampler.

### 1. Spatial declustering weights (geostatistics)

The classical answer to "my samples are spatially clustered, weight them so the
weighted set is representative of the area." Two standard estimators:

- **Cell declustering** — grid the area, each sample's weight ∝ 1 / (samples in its
  cell). Weights depend on cell size; the accepted procedure is to sweep cell size
  and pick the one that minimises (or maximises, for negatively clustered data) the
  weighted mean of the target ([Deutsch, *Cell Declustering Parameter Selection*](https://geostatisticslessons.com/lessons/celldeclustering);
  [CCG *Declustering and Debiasing*](https://ccg-server.engineering.ualberta.ca/CCG%20Publications/Other/CVD%20Papers/02-Conference/2003/DeclusterDebias-CCG.pdf);
  worked examples in [GeostatsPy](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_declustering_all_methods.html)).
- **Polygonal (Voronoi) declustering** — weight ∝ area of the sample's polygon of
  influence. Sensitive to the boundary; cell declustering is preferred as more robust.

This is a strict generalisation of what arms G and I already do: arm I (per-country
cap) *is* cell declustering with countries as cells. Using an equal-area hex grid
(H3) at 2–3 resolutions instead makes the unit comparable across countries and
lets a dense corner of one country be down-weighted without touching its sparse
regions.

**Fit:** drops straight into `training/balancing.py` as a new scheme
(`spatial_cell`, parameters: H3 resolution, class-conditional, max_weight).

### 2. Density-ratio / kernel weighting toward the *prediction* distribution

Declustering makes the training set look spatially uniform. But we don't deploy
uniformly — we deploy on the 157k candidate set, and generalization is judged on
nine specific countries. Importance weighting toward that target is the
principled version:

- **Kernel Mean Matching on coordinates.** Compute weights so the RBF-kernel mean
  of the weighted training coordinates matches that of the target coordinates;
  bandwidth from the median pairwise distance (γ = 1/(2·median²)), weights bounded
  (B ≈ 1000). The key finding for us: *plain* importance weighting fails on
  clustered spatial data because sparse-source/dense-target regions produce
  explosive weights; KMM's bounded quadratic formulation is what keeps it stable,
  with reported error reductions of 12–86 % over naive weighting
  ([Kernel mean matching enhances risk estimation under spatial distribution shifts](https://pmc.ncbi.nlm.nih.gov/articles/PMC12917278/)).
- The species-distribution-modelling literature reaches the same place from the
  other side — occurrence data are clustered by observer effort, and the accepted
  corrections are geographic thinning, environmental thinning, or weighting
  background points toward the sampled region
  ([Inman et al. 2021, *Ecosphere*](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecs2.3422);
  [spThin](https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/ecog.01132);
  [GeoThinneR](https://arxiv.org/pdf/2505.07867)). Their standing warning applies
  to us verbatim: thinning "without testing model sensitivity is strongly
  discouraged" — i.e. seeds.

**Fit:** a `kmm` scheme in `balancing.py` with the target set selectable: all
candidates (deployment prior), or the generalization countries' candidates
(explicitly optimising transfer). The second is the sharper experiment but must be
declared up front — it uses the *locations* of held-out countries, never their
labels, so it is legitimate but it is not a neutral prior.

### 3. Coordinates as a model input (location encoders)

Instead of hiding geography from the model, give it a structured version:

- **SatCLIP** learns a lat/lon → embedding encoder by contrasting Sentinel-2
  imagery with coordinates (spherical harmonics + SIREN). Used as a geographic
  prior it improves species classification, poverty mapping and land-cover
  classification, and "improves geographic generalization by encoding visual
  similarities of spatially distant environments"
  ([Klemmer et al., SatCLIP](https://arxiv.org/abs/2311.17179)).
- **Latent domain modeling** conditions the classifier on a *learned, continuous*
  domain latent derived from coordinates via a location encoder, rather than on
  discrete group labels; reports new state of the art on two WILDS datasets for
  worst-group accuracy ([Latent Domain Modeling Improves Robustness to Geographic Shifts](https://arxiv.org/abs/2503.02036)).
  This is the closest published analogue to "balance by geography without
  choosing the groups by hand."

**The risk, stated plainly:** our audit shows a region-only classifier already
gets +0.25 over majority class from location alone. Feeding coordinates in makes
that shortcut *easier* to learn, not harder. Any location-input arm therefore
needs (a) the balanced sampler underneath it, and (b) an explicit check that
in-domain gains are not bought with out-of-domain losses — which is exactly the
DenseNet failure signature we saw in round_4 (best on eval, worst on generalization).

### 4. Robust objectives over spatial groups — with a skeptical note

Group DRO and friends optimise worst-group loss. Two benchmarks that split
satellite imagery by region are sobering:

- **WILDS / FMoW** (five UN regions): CORAL, IRM and Group DRO "generally fail to
  improve upon ERM baselines" ([WILDS](https://arxiv.org/pdf/2012.07421)).
- **DSGR** (362k images, 49+ countries, six UN regions, built specifically for
  under-represented regions): plain ERM with a CLIP backbone outperforms the
  domain-generalisation methods ([Al-Emadi et al., IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02518-z);
  [project page](https://rwgai.com/dsgr/)).

Our own experience rhymes: the *representation* (SoftCon pretraining, +0.10)
moved generalization more than any objective or sampler has. So a Group-DRO arm
with coordinate-derived groups is worth one slot, not three.

Mixture-of-regional-experts ([CoDEx](https://arxiv.org/abs/2504.19737)) reports
consistent gains on FMoW/DynamicEarthNet but is a much larger change to our
pipeline; park it.

### Diagnostics that use only coordinates (run these first, they are free)

- **Area of applicability / dissimilarity index** — for each generalization row,
  distance to the nearest *training* point in (weighted) predictor space, with the
  threshold calibrated by leave-cluster-out CV. Tells us which held-out countries
  are inside the model's applicable area at all
  ([Meyer & Pebesma 2021](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13650);
  [CAST vignette](https://cran.r-project.org/web/packages/CAST/vignettes/cast04-AOA-tutorial.html);
  their [Nature Comms 2022](https://www.nature.com/articles/s41467-022-29838-9) on
  global maps is the reference for why random CV over-reports).
- **GeoSpOT** — optimal-transport distance between geographic domains via location
  encoders; the distance "emerges as an effective predictor of cross-domain
  transfer difficulty" and can guide data selection
  ([OT on the Map](https://arxiv.org/abs/2604.16220)).

Both give a per-country difficulty score *before* training. If MAR and IND come
out far from everything in coordinate/embedding space, that is direct evidence
for the "label where it is far" recommendation from round_4, and a baseline any
weighting scheme has to beat.

### The CAFO-specific literature does not solve this for us

[Handan-Nader & Ho 2019](https://www.nature.com/articles/s41893-019-0246-x) and
[Robinson et al. 2022](https://arxiv.org/pdf/2112.10988) map CAFOs within the US;
the newest, [PRISM-CAFO](https://arxiv.org/abs/2601.11451), conditions on
*infrastructure* priors (barn counts, lagoon geometry) and evaluates across US
regions only. None evaluates transfer to unseen countries. Our generalization
slice is, as far as this search found, the harder problem.

## Proposal: a three-phase plan

Ranked by expected information per dollar. Every training arm is **3 seeds**,
evaluated on the round_5 generalization slice with pooled *and* within-country AUC,
Holm-corrected against the 3-seed control, with the audit's shortcut gain and NMI
reported for each — the round_4 rules (`experiments/EVAL_METHODS.md`) unchanged.

### Phase 0 — coordinate diagnostics (CPU, ~$0, half a day)

1. Per generalization country: nearest-training-neighbour distance distribution,
   AOA dissimilarity index, and a GeoSpOT-style OT distance (SatCLIP embeddings
   are pretrained and free to compute).
2. Correlate those with per-country AUC from the four round_5 runs. If distance
   predicts error, we have both a targeting tool for the next label round and a
   yardstick for Phase 1.
3. Re-run the audit's region-only-classifier shortcut with H3 cells instead of
   countries at 2–3 resolutions, to pick the cell scale where the shortcut lives.

### Phase 1 — coordinate-weighted sampling (~$12, two arms + control × 3 seeds)

| Arm | Scheme | Why this one |
|---|---|---|
| **J** | `spatial_cell`: H3 cell declustering at the Phase-0 resolution, class-conditional, `max_weight` 10 | the direct generalisation of arm I; if country-level capping helped, sub-country capping should help more and less arbitrarily |
| **K** | `kmm`: kernel mean matching of training coordinates to the full candidate set, bounded weights | the principled version — weights the training set toward where we actually predict |
| A′ | control, 3 seeds | round_5 had one control seed; the band needs three |

Also re-run **H** (the best round_5 arm) at seeds 42/43 so the discrete-vs-continuous comparison is fair. Pre-register: J > A′, K > A′, K > H.

### Phase 2 — coordinates as input (~$5, one arm × 3 seeds), only if Phase 1 shows anything

**L**: arm K's sampler + a frozen SatCLIP location embedding concatenated to the
classifier head, with location dropout p=0.5 so the model cannot rely on it.
Pass/fail is the DenseNet criterion in reverse: it must not gain in-domain while
losing out-of-domain.

### What would make us stop

- Phase 0 shows generalization error is uncorrelated with coordinate distance →
  the problem is not geographic coverage, it is label or imagery quality
  (audit areas B and D in `docs/AUDIT_PROMPT.md`), and no sampler will fix it.
- Phase 1 arms improve pooled AUC but not within-country AUC → we are moving
  cross-country calibration, not skill, exactly as in round_4 §4.3; the fix is
  per-country thresholds, not sampling.

## Implementation notes

- `training/balancing.py` already computes per-row `sample_weights` from a
  scheme name; `spatial_cell` needs `h3` (pure-python fallback: round lat/lon to a
  grid), `kmm` needs a small QP (`cvxpy` or scipy `minimize` with bounds; n≈15k
  training rows is fine with a Nyström approximation of the kernel).
- Both schemes must reuse the existing `class_conditional` and `max_weight`
  machinery and emit the same `sampling_report.json` fields (NMI, ESS, clipped)
  so `evaluate_balancing.py` works unchanged.
- ESS < 0.25 is the existing trip-wire for "weights too extreme"; KMM's bound B
  and cell size are the knobs.

## Sources

- Deutsch, C. *Cell Declustering Parameter Selection* — <https://geostatisticslessons.com/lessons/celldeclustering>
- CCG. *Declustering and Debiasing* — <https://ccg-server.engineering.ualberta.ca/CCG%20Publications/Other/CVD%20Papers/02-Conference/2003/DeclusterDebias-CCG.pdf>
- GeostatsPy declustering demos — <https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_declustering_all_methods.html>
- *Kernel mean matching enhances risk estimation under spatial distribution shifts* — <https://pmc.ncbi.nlm.nih.gov/articles/PMC12917278/>
- Inman et al. 2021, *Comparing sample bias correction methods for SDM* — <https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecs2.3422>
- spThin — <https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/ecog.01132>; GeoThinneR — <https://arxiv.org/pdf/2505.07867>
- Klemmer et al., *SatCLIP* — <https://arxiv.org/abs/2311.17179>
- *Latent Domain Modeling Improves Robustness to Geographic Shifts* — <https://arxiv.org/abs/2503.02036>
- Koh et al., *WILDS* — <https://arxiv.org/pdf/2012.07421>
- Al-Emadi, Yang, Ofli, *Analysing Satellite Imagery Classification under Spatial Domain Shift across Geographic Regions*, IJCV 2025 — <https://link.springer.com/article/10.1007/s11263-025-02518-z>; DSGR project page — <https://rwgai.com/dsgr/>
- *CoDEx: Combining Domain Expertise for Spatial Generalization* — <https://arxiv.org/abs/2504.19737>
- Meyer & Pebesma 2021, *Predicting into unknown space? Area of applicability* — <https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13650>; CAST AOA tutorial — <https://cran.r-project.org/web/packages/CAST/vignettes/cast04-AOA-tutorial.html>
- Meyer & Pebesma 2022, *ML-based global maps of ecological variables and the challenge of assessing them*, Nat. Commun. — <https://www.nature.com/articles/s41467-022-29838-9>
- *OT on the Map: Quantifying Domain Shifts in Geographic Space* — <https://arxiv.org/abs/2604.16220>
- Handan-Nader & Ho 2019, *Deep learning to map CAFOs*, Nat. Sustain. — <https://www.nature.com/articles/s41893-019-0246-x>
- Robinson et al. 2022, *Mapping industrial poultry operations at scale* — <https://arxiv.org/pdf/2112.10988>
- *PRISM-CAFO* — <https://arxiv.org/abs/2601.11451>
