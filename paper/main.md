# Global Mapping of Concentrated Animal Feeding Operations from Sentinel-2 Imagery: Pipeline, Ablations, and Measurement Validity

**Filip Zawadka**, Rachel [surname TBC]

> Markdown edition of `paper/main.tex`. Content is identical; tables are rendered
> natively and citations use author–year keys resolving to the [References](#references)
> section (BibTeX source: `paper/references.bib`).

> Design choices marked **†(E*x.y*)** are inherited defaults *not* supported by a
> controlled ablation at the time of writing; each identifier refers to an entry in
> the companion experiment plan (`paper/experiments_justification_plan.md`).
> Where such an ablation has since been run, the result is reported in
> [§5.5](#55-controlled-ablation-of-the-inherited-defaults).

---

## Abstract

Concentrated animal feeding operations (CAFOs) are a significant source of regional
water and air pollution and a focal point for zoonotic-disease surveillance, yet no
openly available inventory covers them at global scale. We describe an end-to-end
pipeline that detects and types livestock facilities worldwide from freely available
Sentinel-2 imagery. The pipeline is two-stage: a geometric stage clusters candidate
barn-like structures from the Google Open Buildings footprint database using
morphometric filters, and a convolutional stage classifies a 1.28 km Sentinel-2 patch
centred on each candidate cluster into NotFarm, Poultry, Pigs, or Cattle. The
production model is a ResNet-50 initialised from SoftCon, a Sentinel-2–native
self-supervised backbone, adapted to a nine-channel input (six surface-reflectance
bands plus three spectral indices) by band-name–matched first-convolution transfer.
We score 157,099 of 157,102 candidate clusters across 167 countries.

Beyond the system description, we report a systematic experimental record covering
backbone choice, spatial context, class-imbalance interventions, label acquisition,
and regularisation. Three findings recur. First, after de-duplication and bootstrap
confidence intervals, the only comparison that separates robustly from noise is
Sentinel-2–native self-supervised pretraining versus ImageNet initialisation
(≈ +0.10 macro-F1, roughly twice the confidence-interval half-width); a nine-channel
ImageNet control shows the gain is attributable to pretraining rather than band count,
and the six leading Sentinel-2–native configurations are statistically tied. Second,
class-balanced sampling is a reproducible in-domain/out-of-domain trade rather than an
improvement. Third, adding negative labels alone shifts the decision prior without
improving ranking quality: area under the ROC curve is unchanged while both
false-positive rate and recall fall together.

We also document, rather than omit, two data-integrity problems that materially
affected earlier results: a patch-store key misalignment that paired approximately
34.6% of labelled candidates with imagery from the wrong location, and a systematic
label-source shift between training and evaluation data.

Finally, we argue that the dominant bottleneck in this problem is measurement rather
than modelling. Training the same configuration under five random seeds gives a
standard deviation of 0.033 macro-F1 on the standard evaluation slice — a decision
band wider than nearly every intervention previously ranked on it — and we show that
roughly 80% of that variance comes from a single class with six held-out examples.
Re-running twenty single-lever ablations against a purpose-built blind benchmark, on
which the same model is reproducible to σ = 0.0008, we find that exactly one inherited
default is worth changing, that a third of the input channels contribute nothing
measurable, and that two settings previously believed to matter do not.

**Keywords:** concentrated animal feeding operations; Sentinel-2; self-supervised
pretraining; remote sensing; class imbalance; spatial cross-validation; measurement
validity.

---

## 1. Introduction

Intensive livestock production is concentrated in facilities that are large, spatially
clustered, and visually distinctive from above: long, narrow, regularly spaced barns,
often accompanied by waste lagoons. These facilities are of standing interest to
environmental regulators, because manure lagoons and land application drive nutrient
loading in surface and ground water; to public-health agencies, because high-density
confinement is a recognised interface for zoonotic transmission; and to animal-welfare
and food-system researchers, who need facility counts to estimate populations under
confinement. In several jurisdictions the necessary inventory does not exist, is not
public, or is maintained only as an address list without geometry.

Remote sensing is an attractive substitute, and prior work has demonstrated it
convincingly at national scale. Handan-Nader and Ho (2019) detect CAFOs in North
Carolina from aerial imagery and show that model-assisted enforcement targeting
substantially outperforms complaint-driven inspection. Robinson et al. (2022) map
industrial poultry operations across the contiguous United States using
high-resolution aerial imagery. Both rely on imagery products — sub-metre aerial
orthophotography such as the USDA National Agriculture Imagery Program — that have no
global equivalent. Extending this class of system worldwide therefore requires working
from a sensor with global, repeated, free coverage. Sentinel-2 (Drusch et al., 2012)
provides 10 m multispectral imagery with a five-day revisit, roughly two orders of
magnitude coarser in area per pixel than the aerial imagery used in prior CAFO work.
At 10 m, an individual poultry barn is on the order of 2 × 15 pixels: the object is
resolvable as an oriented bright rectangle, but its texture, roof detail, and
ventilation hardware are not.

This paper describes a pipeline built around that constraint, and reports what we
learned attempting to make it work at global scale. Our contributions are:

1. **A two-stage global pipeline.** Rather than searching imagery exhaustively, we
   first generate candidates geometrically from the Google Open Buildings footprint
   database (Sirko et al., 2021) using morphometric filters calibrated on a clean
   reference region, then classify a Sentinel-2 patch centred on each candidate
   cluster. This converts a global dense-search problem into a 157,102-candidate
   classification problem spanning 167 countries ([§3](#3-data)).
2. **A four-class facility-typing model.** We adapt a Sentinel-2–native
   self-supervised ResNet-50 (Wang et al., 2024; He et al., 2016) to a nine-channel
   input via band-name–matched first-convolution transfer, and train it to distinguish
   NotFarm, Poultry, Pigs, and Cattle ([§4](#4-methods)).
3. **A systematic ablation record with honest uncertainty.** We report a controlled
   comparison of backbones, spatial context, band composition, optimisation,
   augmentation, imbalance interventions, label-acquisition strategies, and
   regularisation. Crucially, we calibrate every comparison against measured
   run-to-run variance rather than against zero, and report which effects survive.
   Most do not ([§5](#5-experiments-and-results)).
4. **A data-integrity and measurement-validity methodology.** We document a
   patch-store key misalignment that silently contaminated approximately 34.6% of
   labelled candidates for several months, quantify its effect, show that it
   *reversed* two experimental verdicts, and describe the code-level invariants that
   now prevent it. We also quantify a train/evaluation label-source shift and the
   resulting ceiling on measurable progress
   ([§6](#6-data-integrity-and-measurement-validity)).

A theme runs through the experimental sections and we state it at the outset: for most
of this project's history, the measurement apparatus has been less precise than the
effects being measured. At an evaluation-set size of n = 523, the 95% confidence
interval on macro-F1 is approximately ±0.05, while the interventions under test move
the metric by 0.01 to 0.05. Reporting such comparisons as wins would be unjustified.
We instead report them as ties, identify the single comparison that does separate, and
treat improving the benchmark as the highest-priority experiment rather than a
housekeeping task.

---

## 2. Related work

**CAFO and livestock-facility detection.** Handan-Nader and Ho (2019) established the
template for this task: detect facilities with a convolutional network over aerial
imagery, and evaluate not only classification metrics but downstream regulatory
utility. Robinson et al. (2022) scaled poultry-barn detection across the contiguous
United States, emphasising the operational cost of false positives in a screening
deployment. Both operate at sub-metre resolution in a single country with a reliable
registry for supervision. Our work differs on all three axes: 10 m imagery, 167
countries, and heterogeneous supervision mixing national registries with visual
annotation. The resolution change means our model cannot rely on the fine texture cues
available to prior work, and our results should be read as a different operating point
— broad, low-cost, globally uniform screening — rather than as a direct comparison.

**Self-supervised pretraining for Earth observation.** ImageNet initialisation
transfers poorly to multispectral satellite input, both because the domain gap is
large and because non-RGB bands have no pretrained counterpart. A line of work
pretrains directly on Sentinel imagery: SSL4EO-S12 (Wang et al., 2023) applies momentum
contrast (He et al., 2020) to a large multitemporal Sentinel-1/2 corpus, and SoftCon
(Wang et al., 2024) adds multi-label–guided soft contrastive learning. TorchGeo
(Stewart et al., 2022) distributes these weights with band metadata, which our
first-convolution adaptation depends on. GEO-Bench (Lacoste et al., 2023) provides the
standardised comparison we used to shortlist candidate backbones, alongside broader
multimodal foundation models such as DOFA (Xiong et al., 2024) and CROMA (Fuller et
al., 2023). Common downstream benchmarks in this literature — EuroSAT (Helber et al.,
2019), BigEarthNet (Sumbul et al., 2019) — are scene-classification tasks with large,
balanced label sets; our task is an object-centred, severely imbalanced classification
with a long tail, and we find the benchmark ranking only partially predicts our
downstream ranking.

**Long-tailed recognition.** Our label distribution is severely skewed (roughly 50:1
Poultry to Cattle in training). We evaluate the standard interventions: class-weighted
and class-balanced sampling; classifier retraining with a frozen representation (cRT)
(Kang et al., 2020); train-time logit adjustment and its post-hoc variant (Menon et
al., 2021); and focal loss (Lin et al., 2017). A recurring result in that literature —
that decoupling representation learning from classifier balancing outperforms
end-to-end resampling — holds only partially here.

**Regularisation.** We tested label smoothing (Szegedy et al., 2016) together with
cutout (DeVries and Taylor, 2017) and a longer schedule. Müller et al. (2019) note
that label smoothing improves accuracy and calibration in many settings but can
distort the confidence structure of the penultimate representation; our measurement
was that it worsened expected calibration error on the slice we care about ([§5.4](#54-regularisation-bundle)).
Optimisation follows decoupled weight decay (Loshchilov and Hutter, 2019) with cosine
annealing (Loshchilov and Hutter, 2017).

**Evaluation validity in spatial problems.** Random train/test splits over spatially
autocorrelated data inflate apparent performance, because held-out points are often
near training points (Roberts et al., 2017; Ploton et al., 2020). This applies directly
to our setting; we quantify the exposure in [§6.3](#63-measurement-limits). We report
bootstrap confidence intervals (Efron, 1987), expected calibration error (Guo et al.,
2017), and Matthews correlation coefficient (Chicco and Jurman, 2020) alongside F1,
the last because our slices vary widely in class balance.

---

## 3. Data

### 3.1 Candidate generation

The unit of analysis is a **building cluster**, not a pixel or a tile. This choice is
what makes global coverage tractable: instead of running a detector over all land area,
we enumerate structures that are already known to exist and filter them to those that
are barn-shaped.

We start from Google Open Buildings (Sirko et al., 2021), tiled by H3 hexagons at
resolution 3. Footprints in urban areas are removed. For each remaining building we
compute morphometric descriptors — area, length, width, aspect ratio, orientation —
and apply the filters in Table 1. Thresholds were calibrated on the Delmarva peninsula
("DMV"), a dense, well-documented United States broiler-production region that serves
throughout this project as the clean reference distribution.

**Table 1 — Morphometric candidate-selection parameters.** Calibrated on the Delmarva
reference region and applied globally without per-country tuning.

| Parameter | Meaning | Value |
|---|---|---|
| `MIN_SIZE` / `MAX_SIZE` | building footprint area | 850–5,200 m² |
| `MIN_AR` / `MAX_AR` | aspect ratio (elongation) | 4.5–15 |
| `MIN_WIDTH` / `MAX_WIDTH` | building width | 10–28 m |
| `MAX_LENGTH` | building length | 205 m |
| `BLDG_SEP` | edge-linkage clustering distance | 50 m |
| `CLUSTER_MIN` | minimum buildings per cluster | 2 |
| `TOTAL_AREA` | minimum total cluster footprint | 2,500 m² |
| `DIST_TO_WATER` | lagoon-proximity search radius | 150 m |
| `WATER_PROBABILITY` | Dynamic World water threshold | 0.1 |
| `FARM_TO_CLUSTER` | registry-point to cluster match radius | 100 m |

Buildings whose edges lie within 50 m of one another are linked into a single cluster;
clusters of at least two buildings and at least 2,500 m² total footprint become
candidates. A lagoon feature is derived from Dynamic World (Brown et al., 2022) water
probability within 150 m of the cluster. Registry points are attached to clusters
within 100 m.

This stage is deliberately high-recall and low-precision with respect to the final
task: it produces 157,102 candidate clusters worldwide, of which roughly three quarters
of the labelled subset are in fact farms. That base rate is important context for every
metric in this paper: **a trivial classifier that labels every candidate as a farm
achieves precision 0.767 at recall 1.0** ([§4.5](#45-geometric-baseline)). The
convolutional stage must be evaluated against that baseline, not against a 50% prior.

### 3.2 Labels

Labels come from two qualitatively different sources, and the distinction matters
enough that we return to it in [§6.2](#62-label-source-shift).

*Registry labels* derive from official or curated facility lists: United States state
registries (including the Delmarva and Iowa sets), Mexican and Chilean official
registries, Thailand and Brazil facility data. A registry point within 100 m of a
candidate cluster labels that cluster with the registry's species, subject to
animal-count thresholds (20,000 birds for poultry, 400 head for pigs) to exclude
smallholdings.

*Visual labels* are assigned by human annotators inspecting high-resolution basemap
imagery, and include explicitly adjudicated hard negatives — greenhouses, warehouses,
industrial sheds, plastic-covered horticulture — that the morphometric stage cannot
reject.

The upstream taxonomy is finer than the model's. We map it to four classes: NotFarm,
Poultry (all poultry subtypes), Pigs, and Cattle. Rows whose upstream label is ambiguous
or non-specific are dropped from supervised training but retained with label −1 so that
they are still scored at inference.

**Table 2 — Label composition of the current release (`all_clusters_v9`).** 157,102
clusters across 167 countries, of which 32,214 carry a label.

| Upstream label | n | Upstream label | n |
|---|---:|---|---:|
| NotFarm | 13,778 | Poultry: Eggs | 774 |
| Poultry: Unspecified | 8,005 | Cattle | 228 |
| Poultry: Meat Chickens | 4,265 | Ambiguous | 182 |
| Pigs | 2,576 | PigsOrPoultry | 111 |
| Farm: Unknown | 2,172 | Mixed | 103 |
| | | Other | 20 |

The label set was built in rounds, and the rounds are themselves an experimental
variable ([§5.3](#53-label-acquisition)). Round 2 added 962 adjudicated NotFarm
corrections across 106 countries — exclusively negatives. Round 3 promoted farm
positives from the qualitative-evaluation pool into training under a per-country cap,
first at 50 and then at 70 per country, with an 80:20 train/validation assignment.

### 3.3 Splits

Split assignment is explicit and upstream: each cluster carries a `cnn_split_assigned`
value, and the training code treats it as the single source of truth. No random split
is computed, and no country whitelist or region-balancing logic is applied on this path.
This replaced an earlier RNG-based splitter whose behaviour changed between runs and
which, at one point, silently dropped an entire class ([§6](#6-data-integrity-and-measurement-validity)).

The splits carry distinct semantics rather than being three draws from one pool:

- **train / val / test** — standard fitting, model selection, and held-out measurement.
  `test` is drawn from the same five heavily-labelled countries as `train`, so it is the
  most optimistic slice.
- **eval** — held-out clusters from training countries, weighted toward visually
  adjudicated cases. Measures in-domain generalisation to harder, differently-sourced
  labels.
- **generalization** — countries with *zero* training rows (Bangladesh, Albania,
  Nigeria, DR Congo). The out-of-domain slice, and the one closest to deployment.
- **qual_eval** — a large adjudicated pool used for false-positive and recall estimation
  at scale. Cross-version comparisons use `qual_eval_common`, the intersection across
  all versions, which no model in the comparison trained on.
- **predict** — unlabelled candidates, the deployment target.

**Table 3 — Split composition.** Pigs is absent from `generalization` entirely, and
Cattle support is in the single digits on every held-out slice; both facts bound what
can be claimed about those classes.

| Split | NotFarm | Poultry | Pigs | Cattle | Total | Countries |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,750 | 7,616 | 1,576 | 153 | 12,095 | 139 |
| val | 621 | 1,662 | 356 | 33 | 2,672 | 113 |
| test | 425 | 1,344 | 300 | 25 | 2,094 | 5 |
| eval | 135 | 319 | 62 | 7 | 523 | 5 |
| generalization | 171 | 251 | 0 | 4 | 426 | 4 |
| qual_eval | 9,676 | 1,852 | 282 | 6 | 11,816 | 132 |
| predict | — | — | — | — | 124,888 | 167 |

Training data are geographically concentrated: the United States contributes 6,760
training rows, Mexico 1,614, Thailand 575, Brazil 454, and Chile 330, with a long tail
of roughly 134 further countries. This concentration is why country-level rebalancing
is disabled ([§5.2](#52-class-imbalance-interventions)) and why the `generalization`
slice is weighted most heavily in deployment decisions.

A parallel column, `if_split_assigned`, defines splits for the geometric Isolation
Forest baseline ([§4.5](#45-geometric-baseline)); the convolutional pipeline passes it
through to outputs but never trains on it.

### 3.4 Imagery

For each candidate we extract a 128 × 128 pixel patch at 10 m resolution — a 1.28 km
square — centred on the cluster centroid. The patch grid is constructed in the local UTM
zone with an explicit affine transform, and pixels are requested directly from Google
Earth Engine (Gorelick et al., 2017) as arrays.

The source collection is `COPERNICUS/S2_SR_HARMONIZED`, harmonised Level-2A surface
reflectance (Main-Knorn et al., 2017). We take an **annual median composite** over
calendar year 2023, filtering scenes to `CLOUDY_PIXEL_PERCENTAGE` below 15% **†(E1.3)**.
Six bands are retained — B2, B3, B4 (visible), B8 (NIR), B11, B12 (SWIR) — and three
normalised-difference indices are computed server-side:

```
NDVI = (B8 − B4) / (B8 + B4)
NDBI = (B11 − B8) / (B11 + B8)
NDWI = (B3 − B8) / (B3 + B8)
```

giving a nine-channel `float32` tensor per candidate **†(E1.1)**. NDVI separates barn
roofs from surrounding vegetation, NDBI responds to built surfaces, and NDWI is intended
to expose the manure-lagoon signature. (§5.5 shows the three indices contribute nothing
measurable.)

The annual median is a deliberate trade. It suppresses cloud, haze, and transient
agricultural change, and yields one stable image per site; it also discards all
phenological and temporal signal, which we expect is part of the ceiling on
distinguishing facility types ([§7](#7-limitations-and-planned-work)). A per-pixel
scene-classification cloud mask and a least-cloudy compositing mode are implemented but
never evaluated: the one run configured to test them was lost to a storage-quota failure
**†(E1.3)**.

Patches are cached on disk keyed by candidate identifier and by an *imagery
configuration hash* — a digest over bands, indices, compositing mode, cloud threshold,
and date range. The hash is both the cache key and the inference-time filter, ensuring a
model is never scored on patches built under a different imagery configuration.

**Cloud-blocked candidates and imagery tiers.** Under the primary configuration, 2,197
candidates could not be imaged at all: in chronically cloudy tropical regions (Ecuador,
Malaysia, Indonesia, Brazil, the Philippines, Colombia, Panama, Costa Rica), *no*
Sentinel-2 scene in calendar 2023 passes a 15% scene-cloud filter, so the composite is
empty. Retrying under the identical configuration recovered three. We therefore relaxed
the imagery configuration in two stages and recorded the provenance of each recovered
patch in an `imagery_tier` field (Table 4). Predictions on the relaxed-cloud tier skew
heavily toward Poultry (≈79%), consistent with the out-of-distribution bias discussed in
[§5.8](#58-global-deployment), and are flagged as screening-quality.

**Table 4 — Imagery tiers in the full-world release.** Coverage is 157,099/157,102
candidates (99.998%).

| Tier | Imagery configuration | Candidates |
|---|---|---:|
| `primary_2023` | 2023 median, scene cloud < 15% | 154,905 |
| `clean_2022_2024` | 2022–2024 median, scene cloud < 15% | 1,810 |
| `relaxed_cloud60` | 2022–2024 median, scene cloud < 60% | 384 |
| **Total scored** | | **157,099** |

---

## 4. Methods

### 4.1 Architecture and channel adaptation

The production model is a ResNet-50 (He et al., 2016) initialised from SoftCon (Wang et
al., 2024), a Sentinel-2–native backbone pretrained with multi-label–guided soft
contrastive learning and distributed through TorchGeo (Stewart et al., 2022). We also
evaluate SSL4EO-S12 MoCo weights (Wang et al., 2023; He et al., 2020) and ImageNet
initialisation ([§5.1](#51-backbone-and-spatial-context)). Backbone selection was
shortlisted using GEO-Bench linear-probe results (Lacoste et al., 2023) **†(E1.6)**.

Adapting a pretrained backbone to nine channels is not a matter of widening the first
convolution. The pretrained weights are ordered by the Sentinel-2 13-band Level-1C
convention, `[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]`, whereas our
input is `[B2, B3, B4, B8, B11, B12, NDVI, NDBI, NDWI]`. Copying weights positionally
would place the pretrained B1 filter on our B2 input, and so on — a silent misalignment
across every channel. We therefore adapt **by band name**: each input channel with a
pretrained counterpart receives that counterpart's filter weights, and the three index
channels, which have none, are initialised to the mean of the pretrained first-layer
weights. The classification head is replaced with a fresh linear layer over the
2048-dimensional pooled feature.

### 4.2 Preprocessing and augmentation

Each sample is loaded, non-finite values are zero-filled, spectral bands are scaled by
1/10000 to surface reflectance, augmentation is applied, the patch is centre-cropped to
64 pixels (640 m) **†(E1.2)**, the channel subset is taken, and per-channel
standardisation is applied.

Two normalisation regimes exist because the backbones demand different ones. SSL4EO MoCo
weights expect raw scaled reflectance with no standardisation; SoftCon expects per-band
z-scoring, its official transform. Per-channel statistics are computed once over 512
randomly drawn training patches with augmentation disabled, persisted to disk, and
reloaded at inference — so a scoring shard uses the statistics of the run that trained
the checkpoint, never its own.

**Table 5 — Training augmentation recipe.** Cutout is disabled in the production model;
it was enabled only in the rejected regularisation bundle ([§5.4](#54-regularisation-bundle)).

| Augmentation | Probability | Parameters | Applies to |
|---|---|---|---|
| Horizontal flip | 0.5 | — | all channels |
| Vertical flip | 0.5 | — | all channels |
| Rotation 90°·k | 0.75 | k ∈ {1,2,3} | all channels |
| Continuous rotation | 0.3 | ±15°, reflect | all channels |
| Random resized crop | 0.3 | scale 0.8–1.0 | all channels |
| Brightness jitter | 0.5 | ×[0.85, 1.15] | spectral only |
| Per-band gain jitter | 0.3 | ×[0.95, 1.05] | spectral only |
| Gaussian noise | 0.3 | σ = 0.02 | spectral only |
| Channel dropout | 0.1 | ≤ 1 channel zeroed | spectral only |
| Cutout *(off)* | — | 2 × 16 px holes | all channels |

One inconsistency is worth flagging: per-band gain jitter perturbs the spectral bands,
but the index channels are *not* recomputed afterwards, so NDVI, NDBI, and NDWI are
mildly inconsistent with the bands they were derived from in roughly 30% of training
samples. A `recompute_indices` option exists and is disabled. (§5.5 shows this is
immaterial, because the indices carry no measurable signal.)

### 4.3 Training

The canonical recipe, held byte-identical across the label-round series so that data
effects are not confounded with recipe effects:

```yaml
model:
  architecture: resnet50_softcon
  hub_name: SENTINEL2_ALL_SOFTCON
  pretrained_band_order: [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]
  num_classes: 4
  input_channels: 9
  freeze_backbone_epochs: 5
  class_names: [NotFarm, Poultry, Pigs, Cattle]
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  scheduler: cosine
  early_stopping_patience: 10
  checkpoint_metric: val_f1
  normalization: per_channel
  crop_center_px: 64
  balanced_class_sampling: false
  upsample_minority_regions: false
  class_weight: null
  seed: 42
  mixed_precision: true
inference:
  threshold: 0.5
  confidence_tiers: {high: 0.9, medium: 0.7, low: 0.4}
```

Optimisation uses AdamW (Loshchilov and Hutter, 2019) with cosine annealing (Loshchilov
and Hutter, 2017) **†(E1.4)**. The backbone is frozen for the first five epochs while
the fresh head converges; at unfreezing, the optimiser is rebuilt with the learning rate
scaled by 0.1 and a fresh cosine schedule. Model selection maximises validation macro-F1
rather than minimising validation loss, on the grounds that loss is dominated by the
majority class in a 50:1-imbalanced problem **†(E1.7)**. Mixed precision is enabled for
training but deliberately *disabled* at inference, because half-precision perturbs
probabilities at the 10⁻³ level and can flip `argmax` on near-ties, breaking
bit-comparability with published releases.

The loss is cross-entropy. Optional variants — class weighting, focal loss (Lin et al.,
2017), and train-time logit adjustment (Menon et al., 2021) — are implemented and
evaluated in [§5.2](#52-class-imbalance-interventions); the production model uses none.

Every run is evaluated after training by reloading the best checkpoint and scoring each
held-out slice separately, with per-country breakdowns for `eval`, `generalization`, and
`qual_eval`.

### 4.4 Inference and deployment

Scoring loads the patch metadata table, filters to the current imagery hash, validates
patch locations (below), reconstructs the exact training transform, and runs the model.
For the four-class model the prediction is `argmax` and the full probability vector is
persisted, which allows any downstream binary farm/not-farm analysis to be recomputed as
1 − P(NotFarm) without rescoring. Confidence tiers at 0.9/0.7/0.4 support triage.
Test-time augmentation over the eight dihedral transforms is implemented but has never
been enabled in a published run **†(E1.8)**.

Full-world scoring is sharded by country across compute nodes using a greedy
longest-processing-time partition, with each shard writing to a distinct output path.

**Patch-location validation.** Before any training or scoring run, every patch is checked
against the *current* coordinates of its candidate: the haversine distance between the
coordinates at which the patch was extracted and the candidate's present centroid must
be under 250 m. Rows exceeding it are dropped, and if more than 5% of rows are stale the
run aborts. This guard exists because of the incident in [§6.1](#61-patch-store-key-misalignment),
and it is the single most important piece of defensive code in the pipeline.

### 4.5 Geometric baseline

A parallel, purely geometric model provides a baseline that uses no imagery. An Isolation
Forest (Liu et al., 2008; Pedregosa et al., 2011) (500 estimators, contamination 0.05) is
fitted on seven cluster-level morphometric features: median building area, median length,
number of unique orientations, pairwise morphological distance, water area, mean built
probability, and water proportion. It is fitted *only* on the 1,200-row Delmarva
reference set, making it a one-class template-similarity model.

**Table 6 — Geometric baseline versus the convolutional model**, binary farm-versus-not,
on 15,489 labelled clusters. The always-predict-farm row is the base rate implied by
candidate generation and is the correct reference for all precision figures.

| Model | ROC-AUC | Avg. precision | Precision @ recall 0.95 | Non-farms removed |
|---|---:|---:|---:|---:|
| Always predict farm | — | 0.767 | 0.767 | 0% |
| Isolation Forest (geometry) | 0.862 | 0.952 | 0.818 | 30% |
| CNN (imagery) | 0.963 | 0.979 | 0.959 | 86% |
| Blend (0.2 / 0.8) | — | 0.988 | — | — |

Using the Isolation Forest as a *sequential prefilter* before the convolutional model is
strictly dominated by simply thresholding the convolutional model. We therefore retain it
as a passthrough column and a diagnostic, not as a pipeline stage.

---

## 5. Experiments and results

We report two experimental series. The first varies the **model** (backbone, context,
imbalance treatment) on a three-class taxonomy (NotFarm/Poultry/OtherFarm). The second
holds the architecture byte-identical and varies the **labels**, on the four-class
taxonomy; the production model comes from this series. Unless noted, three-class numbers
are macro-F1 and four-class numbers are binary farm-versus-not ROC-AUC.

### 5.1 Backbone and spatial context

**Table 7 — Backbone and context ablation, three-class macro-F1.** Single-seed point
estimates on the contaminated patch store ([§6.1](#61-patch-store-key-misalignment)); see
Table 11 for corrected magnitudes.

| Run | Initialisation / change | test | eval | gen |
|---|---|---:|---:|---:|
| `v8_three_class` | ImageNet, 4 channels | 0.626 | 0.403 | 0.397 |
| `v9_imagenet9ch` | ImageNet, 9 channels | 0.617 | 0.393 | 0.382 |
| `v8_ssl4eo` | SSL4EO-S12 MoCo, 9 channels | 0.723 | 0.489 | 0.407 |
| `v9_softcon` | SoftCon, 9 channels | 0.724 | **0.504** | 0.402 |
| `v9_ctx128` | SSL4EO, 128 px context | 0.729 | 0.484 | **0.440** |
| `v9_softcon_ctx128` | SoftCon + 128 px context | **0.751** | 0.492 | 0.378 |
| `v8_crt` | SSL4EO + cRT head | 0.727 | 0.494 | 0.426 |
| `v8_logitadj` | SSL4EO + logit-adjusted CE | 0.572 | 0.470 | 0.330 |

**Pretraining, not band count, drives the win.** Sentinel-2–native pretraining improves
eval macro-F1 from 0.403 to 0.489–0.504. The obvious confound is that the ImageNet
baseline used four channels while the self-supervised runs used nine. The
`v9_imagenet9ch` run resolves this: holding the nine-channel input fixed and reverting
only the initialisation gives 0.393 eval — statistically indistinguishable from the
four-channel ImageNet run at 0.403, and far below every Sentinel-2–native run. The
additional bands are close to worthless to an ImageNet-initialised network and valuable
to a Sentinel-2–pretrained one. **We regard this as the single most robust experimental
result in the project.**

**Everything else is a tie.** A validity audit re-ran these comparisons after
de-duplicating candidates and attaching bootstrap confidence intervals (Efron, 1987).
De-duplication alone drops the SoftCon eval figure from 0.504 to 0.469. After correction,
the six leading Sentinel-2–native configurations fall in 0.466–0.470 with CI half-widths
of roughly ±0.05: a statistical tie. Only the Sentinel-2–native versus ImageNet contrast,
at roughly +0.10, exceeds twice the interval width. The production choice of SoftCon over
SSL4EO was therefore not justified by these measurements; it rested on external benchmark
evidence (Lacoste et al., 2023).

Re-testing the pair on the frozen benchmark confirms the production choice, but by a far
smaller margin than four-class macro-F1 suggests. SSL4EO is worse by 0.0053 ROC-AUC
(6.9σ) and by 0.037 macro-F1 on `qual_eval`, and is indistinguishable from SoftCon on
`test`, `eval`, and `generalization` once Cattle is excluded. Raw four-class macro-F1
instead reports SSL4EO as 0.143 worse on `test`, which is an artefact: per class the two
differ by 0.001, 0.006, and 0.010 on NotFarm, Poultry, and Pigs, and by **0.541 on
Cattle**, whose F1 varies by up to 0.2 between seeds of the same configuration
([§6.3](#63-measurement-limits)).

**Context does not compose.** Increasing the crop from 64 to 128 pixels improves
out-of-domain macro-F1 for SSL4EO (0.407 → 0.440). A larger blind slice was previously
reported to show a +0.047 macro-F1 advantage for the wider-context model, but that does
not survive re-testing: on the 1,072 blind rows we can pair exactly across the two
models, the difference is +0.002 pooled (95% CI [−0.044, +0.048], p = 0.93) and +0.017
averaged over countries.

A controlled ablation settles the question against larger context. Training the
production configuration at 48, 64, and 128-pixel crops and evaluating on the frozen
benchmark, 128 pixels is *worse* than the production 64 (−0.0044 ROC-AUC, 5.7σ), as is 48
(−0.0026, 3.4σ). The per-slice pattern is the informative part: 128 pixels gains on `test`
(+0.015 macro-F1) while losing heavily on `generalization` (−0.085, 8.6σ). **Additional
context helps the model memorise the regions it trained on and hurts transfer** — the
opposite of what a screening deployment needs. Combining SoftCon with 128-pixel context
does not stack either: gen falls to 0.378, below either lever alone.

### 5.2 Class-imbalance interventions

We evaluated five distinct mechanisms. None is used in the production model.

*Class-weighted cross-entropy* with weights [1.0, 0.7, 2.0] had no measurable effect —
though this run coincided with the split bug of [§6](#6-data-integrity-and-measurement-validity)
and is uninterpretable.

*Classifier retraining* (cRT) (Kang et al., 2020) improved out-of-domain macro-F1 by
+0.019 on the SSL4EO backbone but *reduced* it by −0.020 on SoftCon. A sign flip of this
size on n = 273 is inside sampling noise; the honest reading is that cRT's effect here is
unmeasured.

*Train-time logit adjustment* (Menon et al., 2021) reached eval 0.470 but at a 0.15 cost
in test macro-F1. A post-hoc τ sweep found the optimum at τ = 0 for the best models —
indicating the limiting factor is the representation, not the decision rule.

*Adaptive batch normalisation* (Li et al., 2018) **hurt**: out-of-domain macro-F1 fell
from 0.407 to 0.383. A useful negative: the out-of-domain gap is not radiometric domain
shift, which AdaBN addresses, but morphological and label-distributional difference.

*Class-balanced sampling* is the intervention we studied most carefully. Measured twice on
independent experiments, it improves eval by +0.027 and +0.046 while reducing
generalization by −0.016 and −0.054. The mechanism is legible: eval is drawn from the same
training countries whose composition the sampler reweights, whereas generalization
contains no training rows. The clearest diagnostic is Cattle: balanced sampling raises
test Cattle F1 from 0.556 to 0.636 while dropping qual_eval Cattle F1 from 0.250 to 0.000
— memorisation of 153 training examples rather than a learned category. Farm recall on
inference countries falls from 89.3% to 78.3%. Since deployment targets countries with no
training data, we keep balancing off.

We tested whether this trade is instead an artefact of spatial leakage
([§6.3](#63-measurement-limits)). It is not: restricting eval to clusters at least 1.28 km
from any training cluster *widens* the gain from +0.046 to +0.084 macro-F1, while on the
leaked rows the balanced model is 0.092 *worse*. The out-of-domain loss is undiminished by
filtering (−0.054 macro-F1; −0.019 ROC-AUC, p = 0.038). **The trade is a genuine property
of the intervention.**

Country-level rebalancing is disabled permanently for a structural reason: 26 training
countries have two or fewer rows, and inverse-frequency country weights would draw a
single-row country roughly 100 times per epoch.

### 5.3 Label acquisition

**Table 8 — Label-round series, binary farm-versus-not ROC-AUC.** Architecture and recipe
byte-identical across columns; only the label set differs. Bracketed figures are 95%
bootstrap CIs.

| Slice | n | v6 | v7 | v8 | v9 | v10 |
|---|---:|---:|---:|---:|---:|---:|
| test | 2,094 | 0.991 | 0.993 | 0.993 | **0.994** [.989,.997] | 0.993 |
| eval | 523 | 0.903 | 0.903 | **0.920** [.888,.949] | 0.911 | 0.903 |
| generalization | 426 | 0.824 [.747,.894] | 0.843 | 0.909 | **0.915** [.885,.942] | 0.903 |
| qual_eval_common | 11,437 | 0.966 | 0.954 | 0.981 | **0.983** [.980,.986] | 0.979 |
| ALB + COD (hardest OOD) | 179 | 0.777 | 0.772 | 0.912 | **0.916** | — |

**Round 2: negatives alone shift the prior without improving the model.** Round 2 added
962 adjudicated NotFarm corrections and nothing else. Superficially a success: the
false-positive rate on the hardest out-of-domain pair fell from 66.2% to 5.0%. But recall
collapsed in step, from 92.5% to 42.5%. Three diagnostics establish that no ranking
improvement occurred. ROC-AUC was flat or slightly worse (0.777 → 0.772). At a *matched*
false-positive rate, recall was identical (95.0% for both). And mean predicted farm
probability fell for true negatives (0.091 → 0.021) *and* for true positives (0.846 →
0.568) — a uniform downward shift, not sharpened separation. **The model learned
"unfamiliar country implies not a farm."** One-sided labels teach a prior, not a feature.
Any label campaign that adds only one class should be evaluated with threshold-free and
matched-operating-point metrics.

**Round 3: balanced promotion is what actually helped.** Adding 1,740 farm *positives*
improved ranking on every slice: ALB+COD 0.772 → 0.912, BGD+NGA 0.910 → 0.936, qual_eval
0.952 → 0.981. Against the round-1 baseline, the qual_eval false-positive rate fell from
4.9% to 3.6% *with* recall rising from 86.8% to 88.9% — a genuine Pareto improvement.
Pigs F1 gained +0.124.

**The per-country cap is not a sensitive knob.** Raising the cap from 50 to 70 promoted a
further 1,963 rows and changed essentially nothing. Re-testing on a purpose-built blind
benchmark of 11,365 rows across 131 countries converts that observation into a properly
powered null result: the paired difference is −0.0003 ROC-AUC with 95% CI [−0.0019,
+0.0012] and p = 0.715. On the same benchmark the two earlier rounds separate decisively
(−0.014 and −0.031, both p < 0.001), confirming the benchmark can detect real differences.
This lever is exhausted; the next data investment should change the label *distribution*
(new regions, new species) rather than its volume.

### 5.4 Regularisation bundle

A final experiment combined label smoothing at 0.05 (Szegedy et al., 2016), cutout with
two 16-pixel holes at probability 0.5 (DeVries and Taylor, 2017), and a longer schedule
(70 epochs, patience 15). It was run as a bundle, not as single levers — a design choice
we now regard as a mistake, since the result cannot be attributed.

The bundle was rejected. Label smoothing *worsened* measured calibration, the opposite of
its usual justification: expected calibration error (Guo et al., 2017) rose from 0.027 to
0.052 on `qual_eval_common` and from 0.006 to 0.020 on `test`. Cutout did not improve the
data-poor Cattle class — one-versus-rest AUC for Cattle on `test` fell from 0.936 to
0.856. Cutout is also applied to the full 128-pixel patch before the 64-pixel centre crop,
so a hole lands inside the visible crop only about a quarter of the time per hole, making
the effective augmentation rate roughly 22% of samples rather than the nominal 50%. The
one gain, +0.057 macro-F1 on eval, is on the slice deployment does not target, and an
ensemble with the production model performed identically to the production model alone.

### 5.5 Controlled ablation of the inherited defaults

The design choices in [§4](#4-methods) were, for most of this project's history,
inherited defaults rather than measured decisions. We tested them directly: twenty runs,
each the production configuration with exactly one change, evaluated on the frozen
benchmark of [§6.3](#63-measurement-limits) and judged against the seed noise measured on
the same benchmark (σ = 0.0008).

**Table 9 — Single-lever ablations**, farm-versus-not ROC-AUC on the frozen blind
benchmark (n = 11,365), as a difference from the production configuration. Δ/σ uses the
seed standard deviation measured on the same benchmark. **Exactly one change improves the
model.**

| Experiment | Change | ΔAUC | Δ/σ | Verdict |
|---|---|---:|---:|---|
| E1.4 | **no backbone freeze phase** | **+0.0045** | +5.9 | **better** |
| E1.5 | drop photometric augmentations | +0.0010 | +1.3 | tie |
| E1.1 | recompute indices after jitter | +0.0009 | +1.2 | tie |
| E1.4 | learning rate 3×10⁻⁴ | +0.0007 | +0.9 | tie |
| E1.5 | cutout only | +0.0004 | +0.5 | tie |
| E1.7 | select on validation loss | +0.0000 | +0.0 | tie |
| E1.9 | three-class taxonomy | +0.0000 | +0.0 | tie |
| E1.4 | learning rate 3×10⁻⁵ | −0.0004 | −0.5 | tie |
| E1.1 | **six bands, no indices** | −0.0010 | −1.3 | **tie** |
| E1.5 | geometric augmentations only | −0.0022 | −2.8 | worse |
| E1.2 | context 48 px | −0.0026 | −3.4 | worse |
| E1.2 | context 128 px | −0.0044 | −5.7 | worse |
| E1.6 | SSL4EO backbone | −0.0053 | −6.9 | worse |
| E1.1 | RGB + NIR | −0.0102 | −13.4 | worse |
| E1.1 | RGB + NDWI | −0.0115 | −15.2 | worse |
| E1.1 | RGB only | −0.0170 | −22.3 | worse |

**The staged freeze hurts — but not for the reason it appears to.** Disabling the
five-epoch freeze phase improves the frozen benchmark by +0.0045 (5.9σ, CI [+0.0025,
+0.0065]), test macro-F1 by +0.017 and qual_eval by +0.020, while shortening training.
This is the only change of the twenty that improved the model.

The *mechanism*, however, is not the one the lever's name implies. The unfreeze transition
does two things at once: it unfreezes the backbone **and** rebuilds the optimiser at 0.1×
the learning rate with a fresh cosine schedule. Setting `freeze_backbone_epochs: 0` skips
that branch entirely, so the "no freeze" arm also trains its backbone at 10× the learning
rate of the baseline (1e-4 against 1e-5) for every remaining epoch. This is confirmed in
the logged learning rate, which drops from 9.755e-05 at epoch 5 to 9.990e-06 at epoch 6 in
every freeze-enabled run. The ablation is a two-factor change reported as one, and the
second factor is very likely dominant.

This also qualifies the learning-rate result below. The sweep varied the *base* rate while
leaving the 0.1× coupling intact, so the configuration a staged warm-up would most
plausibly want — warm-up followed by fine-tuning at the full rate — was never expressible,
let alone tested. The round-four campaign adds an arm that decomposes it: warm-up retained,
unfreeze at full learning rate. Until it reports, the correct statement is that *the freeze
schedule's coupled learning-rate cut hurts*, not that freezing hurts. The base learning
rate is otherwise at a sensible optimum: 3×10⁻⁵ is clearly worse and 3×10⁻⁴ trades test
against qual_eval.

**The spectral indices contribute nothing; the SWIR bands do.** Six raw bands without
NDVI, NDBI, or NDWI are indistinguishable from the full nine-channel input on every slice
(−0.0010, 1.3σ), whereas reducing further to RGB+NIR costs 13σ. The information the model
uses is in the bands — including the two SWIR channels — and the three derived indices, a
third of the input, are dead weight. This is the expected result for quantities that are
deterministic functions of channels the network already sees, and it retires the
index/jitter inconsistency of [§4.2](#42-preprocessing-and-augmentation) as a concern.

**Augmentation earns its place, but only out of domain.** Reducing the recipe to flips and
rotations costs both the frozen benchmark (−0.0022) and generalization (−0.036 macro-F1,
3.6σ); dropping only the four photometric terms is a tie on the frozen benchmark but
−0.021 on generalization (2.1σ, inside the decision band). The augmentations buy transfer,
not in-domain accuracy. Cutout in isolation is neutral everywhere, which retrospectively
clears it of responsibility for the failure of the bundle in
[§5.4](#54-regularisation-bundle).

*Attribution caveat:* these three augmentation runs test *groups*, not individual terms,
and no augmentation hyperparameter (probability, magnitude) has been ablated. Per-term
leave-one-out remains outstanding.

**Two settings do not matter.** The checkpoint-selection metric and the class taxonomy are
both exactly 0.0000 on the frozen benchmark. The taxonomy result is the more
consequential: dropping to three classes costs nothing measurable, so the Cattle class can
be merged away at no cost — and doing so would remove the dominant source of metric noise
identified in [§6.3](#63-measurement-limits).

### 5.6 Calibration and thresholds

Miscalibration in this system is domain-conditional, which has a direct operational
consequence. Re-tuning a *global* decision threshold is a no-op: the validation-optimal
farm threshold is 0.481 against a default of 0.5, worth at most 2.5 percentage points of
out-of-domain recall. But applying a *lower threshold specifically out of domain* recovers
substantial recall at little false-positive cost, as the model is systematically
under-confident on unseen countries.

**Table 10 — Threshold sweep on the out-of-domain slice.**

| Threshold | Recall | False-positive rate | Precision |
|---|---:|---:|---:|
| 0.2 | 0.894 | 0.211 | 0.864 |
| 0.3 | 0.855 | 0.170 | 0.883 |
| **0.4** | **0.820** | **0.135** | **0.901** |
| 0.5 (default) | 0.765 | 0.117 | 0.907 |
| 0.6 | 0.718 | 0.094 | 0.920 |

Pre-registering the rule "adopt the lowest threshold buying at least five points of recall
for at most five points of false-positive rate" selects **0.4** (+0.055 recall for +0.018
FPR); 0.3 narrowly fails it at +0.053 FPR. We recommend a domain-conditional threshold:
the default in training countries, and 0.4 for screening in countries with no training
data.

We also tested *per-country* thresholds, fitted leave-one-country-out so no country's
threshold sees its own rows, and they are clearly worse than a single out-of-domain
constant: pooled, they buy +0.114 recall at a cost of +0.170 false-positive rate, against
+0.055 for +0.018 from the fixed threshold. With only four out-of-domain countries each
fit rests on three others and is unstable — one country's fitted threshold collapses to
0.05 and triples its false-positive rate. Per-country calibration is not viable until many
more labelled out-of-domain countries exist.

### 5.7 Per-class performance

**Table 11 — Per-class one-versus-rest AUC and F1 for the production model.**

| Class | test AUC | test F1 | eval AUC | eval F1 | gen AUC | gen F1 | qual AUC | qual F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NotFarm | 0.994 | 0.923 | 0.911 | 0.752 | 0.915 | 0.789 | 0.983 | 0.976 |
| Poultry | 0.975 | 0.939 | 0.855 | 0.836 | 0.906 | 0.787 | 0.975 | 0.817 |
| Pigs | 0.965 | 0.795 | 0.775 | 0.421 | — | — | 0.942 | 0.472 |
| Cattle | 0.936 | 0.556 | 0.837 | 0.400 | 0.485 | 0.000 | 0.908 | 0.286 |
| *Support* | *2,094* | | *523* | | *426* | | *11,437* | |

Two structural limits are visible. **Pigs is systematically confused with Poultry** — the
dominant error mode on eval, unmoved by every intervention we tried — which is unsurprising
at 10 m, where the two facility types differ mainly in barn dimensions and lagoon
configuration. **Cattle is not measurable**: 153 training rows, 25 in test, seven in eval,
four in generalization, six in qual_eval. Its out-of-domain one-versus-rest AUC of 0.485 is
indistinguishable from chance, and any Cattle number in this paper should be read as a
placeholder awaiting labels rather than as a measurement.

### 5.8 Global deployment

The production model scores 157,099 of 157,102 candidates across 167 countries (Table 4),
of which approximately 68% are predicted to be farms of some type.

Per-country out-of-domain AUC spans a wide range and is the most honest summary of where
the system can be trusted: South Africa 1.000, Argentina 0.996, Portugal 0.996, France
0.995, Germany 0.981, Australia 0.982, Poland 0.980, Romania 0.979, Hungary 0.973, Italy
0.969, Czechia 0.968, Bangladesh 0.947, Albania 0.942, Nigeria 0.917, DR Congo 0.814.
Performance degrades with distance from the training distribution in both the geographic
and construction-practice sense. Separate analyses identified near-total failures in
specific regions (Russia, Ukraine, Malaysia, India, Turkey) where farm-class F1 falls below
0.3; **these countries should be treated as unsupported by the current release.**

Label-round changes also move deployment-scale outputs substantially. Over the 154,905
commonly scored points, predicted farms went from 113,137 (73.0%) under round 1 to 87,265
(56.3%) under round 2 and 107,013 (69.1%) under round 3 — a reminder that the round-2 prior
shift was not a subtle statistical artefact but a change that would have removed roughly a
quarter of the world's predicted facilities had it shipped.

---

## 6. Data integrity and measurement validity

This section reports failures. We include it because two of them invalidated published
conclusions, and because the mechanisms are general enough to recur in any pipeline that
joins imagery to labels through an identifier.

### 6.1 Patch-store key misalignment

**Discovery.** A cluster in Mexico was scored as a farm with probability 0.963. Manual
inspection confirmed a real egg-production facility at the coordinates the *patch* was
extracted from — but the current label file placed that cluster identifier several hundred
kilometres away, on a NotFarm.

**Mechanism.** Patches were cached and joined by cluster identifier and imagery hash, with
no spatial check, and extraction skipped any identifier already present in the store. An
upstream re-run of the candidate-merge process renumbered 93.5% of cluster identifiers —
geometry and labels unchanged; only the identifier-to-cluster mapping churned. From that
moment, every join silently paired a candidate with another candidate's imagery, and the
skip-cache guaranteed the wrong patch was never re-extracted.

**Blast radius.** 34.3–34.6% of labelled-country rows were displaced by more than 250 m,
with median displacement in the hundreds of kilometres, against a 0.04% control rate in
unaffected regions. Contamination spread across *all* splits, including eval and
generalization, at country-level rates of 21% (Thailand, Bangladesh) to 48% (Chile). In one
binary run, 949 of 2,495 rows were mislocated, and of the 384 for which the original
location's label was recoverable, 84 (22%) were outright image/label contradictions. **All
metrics from that period are floors, not estimates.**

**Correction and consequences.** We re-keyed the store by location, re-extracted roughly
24,700 stale patches, and reduced residual staleness to 0.31%.

**Table 12 — Effect of correcting the misalignment**, three-class macro-F1. The correction
is worth more than any modelling intervention we tested.

| Run | test (contam.) | eval | gen | test (corrected) | eval | gen |
|---|---:|---:|---:|---:|---:|---:|
| `v9_softcon` | 0.712 | 0.462 | 0.400 | 0.890 | 0.631 | 0.486 |
| `v9_ctx128` | 0.710 | 0.469 | 0.415 | 0.902 | 0.641 | 0.443 |
| `v3_three_class` | 0.526 | 0.399 | 0.298 | 0.716 | 0.523 | 0.397 |

Two verdicts reversed under clean data. Class-balanced sampling had been recorded as "the
boundary moves but quality does not"; on corrected data it produced a genuine +0.027 eval
gain with Pigs/Poultry confusion falling from 52% to 38%. Larger context had been recorded
as non-additive with SoftCon; on corrected data it gave a partial +0.017. Neither reversal
would have been detectable without re-running.

**Invariants now enforced.** Three code-level rules resulted. First, patch locations are
validated against current candidate coordinates before every run, with a hard abort above
5% staleness. Second, the candidate-merge step asserts identifier stability — at least
99.9% overlap and centroid drift under 10⁻⁶ degrees — and refuses to join otherwise. Third,
**upstream cluster identifiers are treated as ephemeral**: usable within a single data pull
but never to join across pulls. Location, not identity, is the durable key.

### 6.2 Label-source shift

The gap between test and eval performance is not primarily a generalisation gap; it is a
change in what the labels mean.

For the minority farm class, training labels are 98% registry-derived and 56% United
States, while evaluation labels are 80% visually adjudicated and 3% United States. The two
are different measurement instruments applied to different populations. The same holds for
the negative class: most training NotFarm rows are pipeline-generated negatives with no
recorded label source, whereas nearly all evaluation NotFarm rows are visually confirmed
hard negatives.

Holding the split fixed and varying only the labelling instrument makes the effect directly
visible. Within eval, the production model scores ROC-AUC **1.000 on the 73
registry-labelled rows and 0.909 on the 449 visually-labelled ones**; the headline eval
figure of 0.911 is therefore essentially the visual-label figure, while test, at 84%
registry, is essentially the registry figure. The same ordering holds in every split (test
−0.043 ROC-AUC, qual_eval −0.019), and it is near-zero within train itself (−0.002): the
model fits both label sources equally well during training and generalises worse only on
visual ones. That pattern points to visually-adjudicated rows being intrinsically harder
cases rather than to registry labels being noisy, which argues for acquiring more visual
labels for training rather than down-weighting registry ones.

The consequence is that test-to-eval degradation should not be read as overfitting, and
interventions aimed at overfitting will not close it.

### 6.3 Measurement limits

**The benchmark is smaller than the effects.** At n = 523, the 95% confidence interval on
eval macro-F1 is approximately ±0.05. Detecting a difference Δ requires roughly n ≈ 524 ·
(0.05/Δ)²: about 1,455 rows for Δ = 0.03, 3,275 for Δ = 0.02, and 13,100 for Δ = 0.01.
Nearly every intervention in [§5](#5-experiments-and-results) falls in the 0.01–0.05 range.
Most of our experiments were, strictly, underpowered.

**Splits are not spatially blocked.** Measuring the distance from every held-out cluster to
the nearest of the 12,062 training clusters shows substantial exposure: **16.6% of eval
clusters, 38.0% of test, and 32.5% of val** lie within one patch width (1.28 km) of a
training cluster. The spatial cross-validation literature (Roberts et al., 2017; Ploton et
al., 2020) is unambiguous that this inflates apparent performance, and it does here:
splitting each slice at the 1.28 km boundary, the production model scores ROC-AUC 0.961 on
near eval rows against 0.905 on far ones (+0.056), and 0.993 against 0.964 on val (+0.028).
Two slices are exempt: **generalization has effectively no exposure** (0.5% within 1.28 km,
median distance 116 km) and qual_eval has 3.5%, which is why we weight those two most
heavily. This bounds evaluation contamination only; a spatially blocked *retrain* is still
required **†(E0.2)**.

**Duplicates inflated every split.** The patch store can hold multiple rows per candidate;
de-duplication reduces eval macro-F1 by approximately 0.03.

**Seed variance is larger than most measured effects.** We trained the production
configuration five times, varying only the random seed. On the frozen benchmark the model
is highly reproducible (farm ROC-AUC σ = 0.0008). On the 523-row evaluation slice it is
not: four-class macro-F1 ranges from 0.602 to 0.683, giving **σ = 0.033 and a decision band
of ±0.095** for a comparison between two single runs.

That band is wider than nearly every intervention this project has ranked on that metric:

| Historical verdict | Reported Δ (eval) | Δ/σ | Status |
|---|---:|---:|---|
| Class-balanced sampling helps in-domain | +0.046 | 1.4σ | within noise |
| Regularisation bundle helps eval | +0.057 | 1.7σ | within noise |
| SoftCon beats SSL4EO | +0.015 | 0.4σ | within noise |

Those comparisons were not measurable on the instrument used to measure them, and we report
them here as ties rather than as effects. Two conclusions survive on other evidence: the
balanced-sampling *out-of-domain* loss is −0.054 where σ = 0.0075, and the null result for
the third label round is confirmed independently on the frozen benchmark (p = 0.715).

**The instability is a metric artefact, not model behaviour.** Decomposing the seed spread
by class identifies its source:

| Slice | macro-F1 spread | NotFarm | Poultry | Pigs | **Cattle** | Cattle n |
|---|---:|---:|---:|---:|---:|---:|
| eval | 0.081 | 0.029 | 0.022 | 0.075 | **0.200** | 7 |
| qual_eval | 0.046 | 0.003 | 0.009 | 0.040 | **0.150** | 6 |
| test | 0.014 | 0.008 | 0.003 | 0.026 | 0.038 | 25 |

On qual_eval, NotFarm F1 (9,676 rows) varies by 0.003 across seeds while Cattle F1 (six
rows) varies by 0.150. Because macro-F1 weights all classes equally, **that single
near-empty class contributes roughly 80% of the variance in the headline number.** The
control is test, where Cattle has 25 rows and the spread falls to 0.014. Excluding Cattle
halves the evaluation-slice σ to 0.016 and reduces the qual_eval σ threefold to 0.0056.

Reporting four-class macro-F1 on slices where the rarest class has single-digit support is
therefore close to reporting a coin flip, and we recommend against it. This is also a
cautionary result for reading our own tables: raw macro-F1 made an SSL4EO run appear 0.143
worse than SoftCon on test — an apparent twenty-five-sigma collapse — when the three
measurable classes differed by 0.006 and the entire gap was Cattle falling from 0.541 to
zero in a single run.

**A blind benchmark closes most of this gap.** We froze a benchmark of **11,365 labelled
clusters spanning 131 countries**, drawn from the adjudicated qual_eval pool and further
filtered to clusters at least 1.28 km from any training cluster, so that it is blind in both
senses: no model trained on these rows, and none is a spatial near-duplicate of a training
row. Its 95% bootstrap confidence half-width is 0.0030 on farm ROC-AUC and 0.0061 on
macro-F1 — roughly **sixteen times tighter** than the ±0.05 of the 523-row evaluation slice,
and enough to make differences of 0.01 comfortably detectable. It should be the default
slice for future comparisons. Its own limitation is composition: dominated by NotFarm rows
and containing only six Cattle examples, it resolves farm-versus-not questions well and
minority-type questions not at all.

---

## 7. The round-four campaign: recipe levers under seed replication

After the analyses above were completed, a fourth labeling round restructured the
training data: everything not reserved for the generalization hold-out moved into
train/val, absorbing the qualitative-evaluation slice entirely. Training data grew
12,062 → 21,478 rows (+73%), validation 2,666 → 5,022; the test (2,094), eval
(523) and generalization (617 usable / 662 labeled) hold-outs kept their
membership. The frozen benchmark of Section 6 was thereby retired — its rows are
now training data — so the campaign relies on the three surviving hold-outs, for
which we verified **0.0% overlap** with the train/val sets of every archived model
(v6–v9), and that each archived run trained on its recorded explicit splits.

**Configuration.** All arms share the production recipe (byte-identical to v9):
SoftCon Sentinel-2 ResNet-50 (Wang et al., 2024) with a band-mapped first
convolution and a fresh 4-class linear head (NotFarm/Poultry/Pigs/Cattle); 9 input
channels (B2,B3,B4,B8,B11,B12 + NDVI,NDBI,NDWI) at 64×64 px, per-channel
train-set normalization; AdamW lr 1e-4, weight decay 0.01, cosine schedule
(T=50), batch 32; backbone frozen 5 epochs then unfrozen with the optimizer
rebuilt at 0.1× lr; early stopping patience 10 on validation macro-F1
(best-val checkpoint shipped); flips (p=0.5), 90° rotations, scale jitter and
cutout; no class balancing or weights; mixed precision. Each arm ran with seeds
42/43/44, which control head initialization, shuffle order and augmentation
draws; data, splits and imagery are byte-identical across all 18 runs.

| Arm | Delta vs. A | Question |
|---|---|---|
| A | none | label-round effect (vs. archived v9) |
| B | `freeze_backbone_epochs: 0` | does the freeze-phase result replicate? |
| C | 6 bands (drop NDVI/NDBI/NDWI) | do the indices earn their place? |
| D | B + C | do the levers compose? |
| E | D + DenseNet-121 (ImageNet init, 7.0M vs 23.5M params) | architecture family |
| F | `unfreeze_lr_scale: 1.0` (freeze kept) | freeze vs. learning rate, decomposed |

Arm F was a disclosed mid-campaign amendment: code inspection revealed the
unfreeze transition rebuilds the optimizer at 0.1× lr, so "no freeze" had always
changed two factors at once; F keeps the warm-up and removes only the lr cut, a
previously unrepresentable configuration. The amendment predates any arm-F result
and made the Holm correction stricter for the original contrasts.

**Protocol.** Recipe-level estimand (mean of per-seed AUCs, never a selected seed
or the seed-ensemble); every contrast carries a mandatory seed term,
SE² = SE²_boot + σ²_seed(1/n₁+1/n₂); Holm correction over the five confirmatory
contrasts (Holm, 1979); practical floor 0.005 AUC. Seed sensitivity of fine-tuned
pretrained models is well documented (Dodge et al., 2020; Picard, 2021), and the
campaign reproduced its canonical failure mode live: with one collected seed,
arm B "beat" A at +0.050 (p=0.003); with all three, +0.023 at p=0.29.

**Farm ROC-AUC by slice** (3-seed mean ± sd; archived models on identical rows;
Ambiguous excluded from the binary target, farm-type-unknown labels count as
positives):

| Model | Generalization (n=632) | Eval (n=572) | Test (n=2,162) |
|---|---|---|---|
| A (baseline) | 0.833 ± 0.009 | 0.915 ± 0.004 | 0.993 ± 0.000 |
| B (freeze0) | 0.857 ± 0.036 | 0.929 ± 0.004 | 0.995 ± 0.001 |
| C (6 bands) | 0.833 ± 0.005 | 0.917 ± 0.004 | 0.992 ± 0.001 |
| D (B+C) | 0.861 ± 0.033 | 0.933 ± 0.006 | 0.994 ± 0.001 |
| E (DenseNet) | 0.815 ± 0.012 | 0.934 ± 0.010 | 0.991 ± 0.001 |
| F (full-lr unfreeze) | 0.850 ± 0.027 | 0.929 ± 0.007 | 0.995 ± 0.001 |
| **archived v9 (round 3)** | **0.874** | 0.906 | 0.993 |

**Results.** No confirmatory contrast survives correction on the primary
(generalization) slice: b>a +0.024 (p_Holm=0.67), d>a +0.028 (0.60), f>a +0.017
(0.80), b>f +0.008 (0.80), e>d −0.046 (0.16). Three findings organize the
campaign:

1. **The label-round effect is a trade.** Arm A vs. archived v9: eval +0.009,
   test −0.000, generalization −0.041. The added rows are the absorbed
   qual_eval slice, concentrated in the focal countries; denser in-domain data
   specialized the model at the expense of transfer.
2. **The freeze-phase result decomposes into a learning-rate effect with a
   reliability cost.** F attributes ~two-thirds of the freeze0 delta to the
   backbone lr (f>a +0.017) and little to the warm-up (b>f +0.008). The higher
   rate quadruples seed variance (σ 0.005–0.009 at lr 1e-5 vs 0.027–0.036 at
   1e-4), with one seed in three collapsing ~0.06 AUC in each of B, D, F.
3. **The variance is predominantly cross-country calibration drift, not
   skill.** Collapsing seeds have normal within-country AUC (arm D's weakest:
   0.847 within vs 0.823 pooled); the collapsing seed's per-country offsets
   correlate 0.67–0.79 across B/D/F (the seed fixes the shuffle+augmentation
   stream — a shared treatment amplified by the higher lr). Within countries
   all six arms are nearly equivalent (0.832–0.850) while archived v9 reaches
   0.868: the regression vs v9 is genuine ranking loss. OOD calibration is
   uniformly poor (ECE 0.16–0.20 vs ~0.01 in-domain): scores are rankings, not
   probabilities.

Secondary: the six-band null replicates cleanly (c vs a −0.000, lowest σ of any
arm); DenseNet is confounded three ways but profiles as an in-domain specialist
(best on eval, last pooled OOD, ordinary within-country); Morocco and India
(AUC 0.65–0.76) are weakest under every recipe — between-country spread exceeds
every between-arm difference.

**Consequences.** v9 remains the production model (no arm matches it OOD, pooled
or within-country; the pre-registered retention rule held). The six-band
simplification is adopted; the freeze-phase change is not (gain inside seed
noise, reliability cost unacceptable for single runs). Future annotation should
target countries resembling the failure profile rather than the focal countries.

## 8. Limitations and planned work

Beyond the measurement issues above, four limits are structural.

**Resolution.** At 10 m, facility *typing* is information-limited. Pigs-versus-poultry
discrimination rests on barn dimensions and lagoon presence, both marginal at this scale,
and the confusion has not responded to any modelling intervention.

**Temporal collapse.** A single annual median discards phenology, seasonal lagoon dynamics,
and construction change — signals that are plausibly discriminative and are currently
unavailable to the model by construction.

**Cattle.** With 153 training examples, the class is unlearnable and unmeasurable, and
[§6.3](#63-measurement-limits) shows it is actively harmful to measurement: it supplies
roughly 80% of the seed variance in the headline metric. [§5.5](#55-controlled-ablation-of-the-inherited-defaults)
shows that a three-class taxonomy costs nothing measurable. Unless a targeted labelling
campaign is undertaken, **the class should be merged away**.

**Candidate-stage recall.** Everything downstream is bounded by the morphometric filters of
Table 1, which were calibrated on United States broiler complexes. Facilities that are
smaller, differently proportioned, or built with materials that fragment footprint
extraction are invisible to the entire pipeline, and this failure mode is invisible to our
metrics because such sites never become candidates. **We have no estimate of candidate-stage
recall outside the reference region, and consider obtaining one the most important
unmeasured quantity in the system.**

The companion document `paper/experiments_justification_plan.md` enumerates the experiments
required to convert the inherited defaults flagged with a dagger into justified choices, and
records which have since been run. The measurement upgrades (blind benchmark,
spatial-leakage quantification, seed variance) and the core ablations (band set, spatial
context, optimiser, augmentation, backbone, checkpoint metric, taxonomy) are complete. Three
deployment decisions are settled: a domain-conditional threshold is adopted, and both model
ensembling and geometry fusion are rejected as too small to justify their cost.

Four items remain open. Three require human annotation and have no compute substitute:
double-annotating evaluation rows to bound the label-noise ceiling, acquiring Cattle labels
(or merging the class, which is free), and measuring candidate-stage recall outside the
reference region. The fourth is a retrain on spatially blocked splits, for which the split
assignment is prepared but the training run not yet performed.

---

## 9. Conclusion

We have described a global pipeline for detecting and typing livestock facilities from
Sentinel-2 imagery, covering 157,099 candidate clusters across 167 countries, and reported
its experimental record in full.

The methodological findings we would carry to a similar project are: that sensor-native
self-supervised pretraining is worth substantially more than architectural or optimisation
tuning; that class-balanced sampling optimises the slice sharing the training distribution
at the expense of the slice that does not, and should be evaluated on out-of-domain data
before adoption; that one-sided label campaigns shift decision priors while leaving ranking
quality unchanged, and must be evaluated with threshold-free metrics; and that a silent
identifier-join failure cost us more accuracy than every modelling improvement we made,
which argues for validating spatial joins geometrically rather than by identifier as a
matter of routine.

Two further findings concern measurement rather than modelling, and we would apply them
earliest in any comparable project. **Estimate run-to-run variance before ranking anything**:
five seeds cost a few GPU-hours here and showed that most of our prior comparisons sat
inside noise. And **identify which class drives that variance before trusting a
macro-averaged metric**: a class with six held-out examples supplied roughly 80% of the
movement in the number this project had been optimising. With those two corrections in
place, twenty single-lever ablations reduce to one change worth making, a third of the input
channels shown to be redundant, and a set of defaults already close to a reasonable optimum.

The system is deployed and its outputs are published, with per-country performance reported
so that users can judge where it is trustworthy. Its principal remaining weakness is not the
model or the benchmark but the stage before both: we still have no estimate of what fraction
of real facilities the morphometric candidate generator finds outside its reference region,
and no metric we report can see a facility that never became a candidate.

---

## Appendix A — Reproducibility

**Artefacts.** Each release comprises a scored candidate table (GeoParquet, EPSG:4326
points, carrying per-class probabilities, confidence tier, split membership, label source,
and the imagery tier of Table 4), a GeoJSON and CSV export, a metrics summary, and a copy of
the exact training configuration. Releases are registered in a manifest consumed by an
interactive review map.

**Software.** PyTorch (Paszke et al., 2019) with TorchGeo (Stewart et al., 2022) for
pretrained Sentinel-2 backbones; scikit-learn (Pedregosa et al., 2011) for the Isolation
Forest baseline; Google Earth Engine (Gorelick et al., 2017) for imagery extraction.
Experiment tracking uses MLflow (<https://mlflow.org>). Candidate tiling uses H3
(<https://h3geo.org>).

**Compute.** Training runs on a single GPU; a full training run of the production
configuration completes in under an hour on an NVIDIA RTX 4090 (~55 min, ~60 s/epoch). The
ablation campaign of [§5.5](#55-controlled-ablation-of-the-inherited-defaults) comprised
twenty such runs. Full-world inference is sharded by country across nodes.

**Determinism.** Random seeds are fixed at 42 for PyTorch, NumPy, and the dataset sampler,
with per-worker seeding for augmentation. Inference runs in full precision to keep published
probabilities bit-reproducible. Note that seed fixing makes a run reproducible but does not
make the *metric* stable across seeds — see [§6.3](#63-measurement-limits).

**Non-paper data sources.** Registry supervision derives from public facility registries and
from the Farm Transparency Project (<https://www.farmtransparency.org>) facility database,
supplemented by visual annotation.

---

## References

Full BibTeX records in `paper/references.bib`.

- **Brown, C. F., et al. (2022).** Dynamic World, near real-time global 10 m land use land cover mapping. *Scientific Data* 9:251.
- **Chicco, D., and Jurman, G. (2020).** The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics* 21(6).
- **DeVries, T., and Taylor, G. W. (2017).** Improved regularization of convolutional neural networks with cutout. arXiv:1708.04552.
- **Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., and Smith, N. A. (2020).** Fine-tuning pretrained language models: weight initializations, data orders, and early stopping. arXiv:2002.06305.
- **Drusch, M., et al. (2012).** Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment* 120:25–36.
- **Efron, B. (1987).** Better bootstrap confidence intervals. *JASA* 82(397):171–185.
- **Fuller, A., Millard, K., and Green, J. R. (2023).** CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders. *NeurIPS*.
- **Gorelick, N., et al. (2017).** Google Earth Engine: Planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment* 202:18–27.
- **Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017).** On calibration of modern neural networks. *ICML*, 1321–1330.
- **Handan-Nader, C., and Ho, D. E. (2019).** Deep learning to map concentrated animal feeding operations. *Nature Sustainability* 2:298–306.
- **He, K., Zhang, X., Ren, S., and Sun, J. (2016).** Deep residual learning for image recognition. *CVPR*, 770–778.
- **He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. (2020).** Momentum contrast for unsupervised visual representation learning. *CVPR*, 9729–9738.
- **Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019).** EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE JSTARS* 12(7):2217–2226.
- **Holm, S. (1979).** A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2):65–70.
- **Kang, B., et al. (2020).** Decoupling representation and classifier for long-tailed recognition. *ICLR*.
- **Lacoste, A., et al. (2023).** GEO-Bench: Toward foundation models for Earth monitoring. *NeurIPS Datasets and Benchmarks*.
- **Li, Y., Wang, N., Shi, J., Hou, X., and Liu, J. (2018).** Adaptive batch normalization for practical domain adaptation. *Pattern Recognition* 80:109–117.
- **Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017).** Focal loss for dense object detection. *ICCV*, 2980–2988.
- **Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008).** Isolation forest. *ICDM*, 413–422.
- **Loshchilov, I., and Hutter, F. (2017).** SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
- **Loshchilov, I., and Hutter, F. (2019).** Decoupled weight decay regularization. *ICLR*.
- **Main-Knorn, M., et al. (2017).** Sen2Cor for Sentinel-2. *Proc. SPIE* 10427:1042704.
- **Menon, A. K., et al. (2021).** Long-tail learning via logit adjustment. *ICLR*.
- **Müller, R., Kornblith, S., and Hinton, G. E. (2019).** When does label smoothing help? *NeurIPS*.
- **Paszke, A., et al. (2019).** PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*.
- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. *JMLR* 12:2825–2830.
- **Picard, D. (2021).** Torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision. arXiv:2109.08203.
- **Ploton, P., et al. (2020).** Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11:4540.
- **Roberts, D. R., et al. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8):913–929.
- **Robinson, C., et al. (2022).** Mapping industrial poultry operations at scale with deep learning and aerial imagery. *IEEE JSTARS* 15:7458–7471.
- **Sirko, W., et al. (2021).** Continental-scale building detection from high resolution satellite imagery. arXiv:2107.12283.
- **Stewart, A. J., et al. (2022).** TorchGeo: Deep learning with geospatial data. *SIGSPATIAL*, 1–12.
- **Sumbul, G., Charfuelan, M., Demir, B., and Markl, V. (2019).** BigEarthNet: A large-scale benchmark archive for remote sensing image understanding. *IGARSS*, 5901–5904.
- **Szegedy, C., et al. (2016).** Rethinking the inception architecture for computer vision. *CVPR*, 2818–2826.
- **Wang, Y., et al. (2023).** SSL4EO-S12: A large-scale multimodal, multitemporal dataset for self-supervised learning in Earth observation. *IEEE GRSM* 11(3):98–106.
- **Wang, Y., Albrecht, C. M., and Zhu, X. X. (2024).** Multi-label guided soft contrastive learning for efficient Earth observation pretraining. *IEEE TGRS* 62:1–16. arXiv:2405.20462.
- **Xiong, Z., et al. (2024).** Neural plasticity-inspired multimodal foundation model for Earth observation. arXiv:2403.15356.
