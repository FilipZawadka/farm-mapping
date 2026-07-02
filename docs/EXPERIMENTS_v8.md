# v8 experiment plan — bugfixes + research-backed improvements

Written 2026-07-02. Context: world_v5–v7 three-class results were
unsatisfactory (OtherFarm F1 = 0 in v6/v7, Poultry↔OtherFarm confusion in
v5, weak OOD transfer to BGD/NGA). This document records (a) the root-cause
findings from the pipeline audit, (b) the research that informed the new
experiments, (c) the experiment matrix, ready to launch.

---

## 1. Audit findings (what was actually wrong)

### 1.1 CRITICAL — v6/v7 never trained on OtherFarm at all
`training/dataset.py` stratified train/val/test by `label == 1` vs
`label == 0` **only**. Any row with `label == 2` fell into neither list and
was silently dropped from train/val/test. v3–v5 dodged this because
`balanced_country_splits: true` routed through a class-agnostic splitter;
v6/v7 turned that off and hit the buggy path.

**Consequence: the "v6 natural-distribution experiment" and the "v7 light
class weight" experiment were invalid.** F1_class2 = 0.000 was not a class
imbalance phenomenon — the model had zero OtherFarm training samples, and
v7's `[1.0, 0.7, 2.0]` weight had nothing to act on. The v6/v7 conclusions
in `docs/pipeline/00_overview.md` are superseded.

Fixed: `_stratified_class_split()` stratifies over **all** classes;
`build_splits` now logs per-class counts per split and **raises** if any
class has 0 train rows.

### 1.2 Other defects fixed in this change-set
| Issue | Impact | Fix |
|---|---|---|
| No per-pixel cloud masking (only scene-level `CLOUDY_PIXEL_PERCENTAGE<15`) | Hazy/cloudy annual medians, worst in tropics (THA + both OOD countries) | `patches.cloud_mask: scl` option (masks SCL 3/8/9/10 before median); only hashes when enabled, so existing patch stores keep their hash |
| No input normalization (ImageNet backbone fed raw 0.05–0.3 reflectances) | Slower/worse fine-tuning | `training.normalization: per_channel` — train-split mean/std, persisted to `patches/splits/<stem>_norm_stats.json`, applied identically at inference |
| Checkpoint selection + early stopping on weighted val **loss** | Minority-class collapse invisible to model selection; val had only 250 rows in balanced-split configs | `training.checkpoint_metric: val_f1` (macro-F1) |
| `f1_class{i}` index shifting when a class was absent from a slice | Misread metrics | Fixed class range 0..max; per-epoch val per-class F1 now logged to MLflow; confusion matrix now in every metrics JSON |
| `num_workers=0` everywhere | GPU starvation (MooseFS volume) | `training.dataloader_workers` + per-worker RNG reseeding (workers previously would have shared identical augmentation streams) |
| Resume-from-checkpoint never unfroze the backbone if resumed before the unfreeze epoch | Frozen-forever backbone on resumed runs | Transition now keyed on `start_epoch`, not "was this a resume" |
| torchgeo pretrained weights adapted by copying the **first k** conv channels | Our `[B2,B3,B4,...]` would misalign against SSL4EO's 13-band `[B1,B2,B3,...]` order | Band-NAME-mapped conv adaptation (`model.pretrained_band_order`); indices (NDVI/NDWI/…) mean-initialised |

Checked and NOT broken (for the record): the S2_SR_HARMONIZED +1000 DN
offset does **not** apply (harmonized collection removes it; verified local
patch DNs min≈54), and `[B2,B3,B4,NDWI]`+64px channel/crop plumbing is
correct end-to-end including inference.

### 1.3 Known remaining gaps (deliberate, see §4)
- Sentinel-1 channels would be mangled by the `/10000` scaling and the
  `[-1,1]` augmentation clip (S1 GRD is in dB, ≈−30..0). Needs a small
  per-source scaling scheme in `PatchDataset` before any S1 fusion run.
- Random splits within countries → spatially close clusters can straddle
  train/test (optimistic test metrics). A spatial-block split option would
  make the headline test honest; eval/generalization slices are unaffected.
- `_country_balanced_split_indices` (v5-era) sizes val/test off the smallest
  country (250 rows total). The v8 recipe avoids that path entirely.

---

## 2. What the research says (condensed)

Full agent report available on request; highlights that shaped the matrix:

1. **10 m Sentinel-2 is proven viable for poultry-CAFO presence
   classification** — Earth Genome's Earth Index found 16k+ poultry CAFOs in
   6 US states with **SoftCon (S2-SSL) embeddings + a small head** (~94%
   precision-equivalent, 82% recall). Everyone who struggled used <1–5 m
   imagery or ImageNet features. The representation, not the sensor, is our
   most suspect component.
2. **S2-native pretrained backbones** (torchgeo): SSL4EO-S12 MoCo ResNet-50
   beats ImageNet massively on small S2 patches (EuroSAT 64px KNN: 94.1 vs
   65.6). Ships as `ResNet50_Weights.SENTINEL2_ALL_MOCO`; expects /10000
   scaling (= our legacy scaling — do NOT z-score for that run).
3. **Class imbalance at 4:1 is mild** — exotic losses (focal, CB, LDAM) buy
   ~nothing at this ratio; the literature-backed tools are **post-hoc logit
   adjustment** (zero retraining, tunable τ), **cRT** (retrain only the head
   with balanced sampling on top of naturally-trained features — the
   published fix for exactly our "balancing ruined the features" v5
   symptom), and **label audits** (oversampling a noisy minority replays its
   label errors).
4. **OOD (BGD/NGA)**: **AdaBN** (re-estimate BatchNorm stats on unlabeled
   target patches — mechanism-perfect for a BN ResNet, 0 labels, hours) ≫
   adversarial DA (DANN/CORAL barely beat ERM on geographic shift in WILDS).
   Also: ~50–200 in-country labels reliably beat every unsupervised trick —
   worth asking Rachel about a small BGD/NGA labeling budget as the single
   biggest OOD lever.
5. **Footprint geometry is a published poultry signature** (Robinson et al.
   production filter: rotated-rect aspect ratio 3.4–20.5, area 526–8107 m²)
   and is more country-invariant than radiometry. Our candidates already
   carry `num_bldgs, total_area_m2, median_area, template_score_if` —
   a LightGBM probe on those + CNN-feature fusion is a cheap high-ceiling
   follow-up.
6. **Temporal signal**: the annual median destroys the "temporally-invariant
   bright object amid seasonally cycling fields" signature that built-up
   mappers exploit. Seasonal medians or per-band temporal-std channels are
   the highest-ceiling *data* change (needs re-extraction).
7. **Skip list** (evidence-backed): focal/CB/LDAM losses as primary fix,
   SeCo weights (worse than ImageNet), single-image super-resolution,
   high-res basemap scraping (ToS-prohibited for Google/Esri/Bing; NICFI
   discontinued).

---

## 3. Experiment matrix (ready to run)

All configs in `configs/rachel_clusters/`. Launch pattern per run:

```bash
# candidates only if candidates_dir is new for that config
python -m training.runpod_launch --config configs/rachel_clusters/<name>.yaml --prep
# patches only for world_v8_cloudfree (new imagery hash)
python -m training.runpod_launch --config configs/rachel_clusters/<name>.yaml --patches
# train + inference + visualize
python -m training.runpod_launch --config configs/rachel_clusters/<name>.yaml
```

| # | Config | Change vs v8 base | Hypothesis | Cost |
|---|---|---|---|---|
| 1 | `world_v8_three_class` | Bugfix rerun of the natural-distribution recipe (+ val_f1 selection, per-channel norm, workers) | OtherFarm learnable at last; new honest baseline. Target: f1_class2 ≥ 0.43 (baseline_v2 level) with better Poultry than v5 | 1 GPU run, no re-extraction |
| 2 | *(post-hoc, no GPU)* `scripts/logit_adjust_sweep.py` on #1's scored parquet | τ-swept prior correction on saved probabilities | Recovers most of any remaining minority-class deficit for free; diagnostic: if OtherFarm F1 jumps, features were fine | ~1 min CPU |
| 3 | `world_v8_crt` | Stage-2: freeze #1's backbone, retrain head with class-balanced sampling | Beats #2 slightly if features are good (published cRT result) | ~15-epoch GPU run |
| 4 | `world_v8_logitadj` | Train-time logit-adjusted loss (τ=1) | Ties or slightly beats #2/#3; keep whichever wins on eval | 1 GPU run |
| 5 | `world_v8_ssl4eo` | SSL4EO-S12 MoCo ResNet-50, 9-ch band-mapped, no z-score | Biggest single jump, esp. OOD + OtherFarm (better features from S2-native pretraining) | 1 GPU run |
| 6 | *(post-hoc, no labels)* `scripts/adabn_adapt.py` on best of #1–#5 | Re-estimate BN stats on BGD/NGA patches | Generalization macro-F1 up several points; quantifies how much OOD gap is radiometric vs morphological | ~10 min GPU/CPU |
| 7 | `world_v8_cloudfree` | SCL per-pixel cloud mask + scene filter relaxed to 60% | Cleaner composites; THA per-country + BGD/NGA gen metrics up; also improves every future run's data | patch re-extraction (~2–4 h CPU) + 1 GPU run |

**Recommended order**: 1 → 2 (same day) → 5 → 6 → 3/4 (only if 2 leaves a
gap) → 7 (kick off extraction in parallel early since it's CPU-bound).

Success metric per run: eval-set macro-F1 and per-class F1 (Rachel's
representative sample — the deck's Eval 2) plus generalization macro-F1
(Eval 3), not just the friendly test split.

### Decision points after the matrix
- If #5 wins → try SoftCon RN50 (`wangyi111/softcon`, HF) and/or torchgeo
  `SENTINEL2_ALL_SOFTCON` next; it's the exact model behind Earth Index's
  poultry success.
- If #6 shows a big AdaBN jump → radiometric shift dominates; add MixStyle +
  heavier photometric augmentation to the winning recipe.
- If OtherFarm still lags after #2–#4 → suspect label noise; run a
  cleanlab-style audit of Poultry↔OtherFarm on out-of-fold predictions and
  send the top ~200 suspects to Rachel for re-review.

## 4. Next wave (needs modest code work, high ceiling)
1. **Footprint-geometry fusion**: LightGBM on `num_bldgs/total_area_m2/
   median_area/template_score_if` as an information-ceiling probe (hours),
   then concat a small MLP on those features with the CNN's pooled features.
2. **Seasonal / temporal channels**: 2–4 seasonal medians or per-band
   temporal std via a second imagery source; re-extraction required.
3. **Sentinel-1 VV/VH fusion**: provider exists (`earth_engine_s1`); first
   add per-source scaling in `PatchDataset` (dB → (db+30)/35), then a
   config with `imagery_sources: [s2, s1]`.
4. **Spatial-block splits**: honest headline test metrics.
5. **Ensemble / greedy soup** over the v8 family checkpoints — near-free
   robustness, best-evidenced OOD gain per unit effort.
6. **Small BGD/NGA labeled set** (50–200 clusters/country) — likely the
   single biggest OOD improvement available; labeling budget, not modeling.
