# Shortlist-country map review: Indonesia, Philippines, Vietnam

*2026-08-18 — visual review of the `world_v10_fourclass_v9` map against high-resolution
basemap imagery (Esri World Imagery). 148 candidates sampled across 6 strata
(manifest: `experiments/results/shortlist_review_manifest.csv`, sampling code:
`experiments/shortlist_review.py`, seed 42). ~60 sites individually inspected, plus
3×3 zoom-16 neighbourhood grids around one confirmed hotspot per country.*

Counts for context (scored release, per country): IDN 2,442 candidates / 64%
predicted farm; PHL 2,344 / 77%; VNM 1,910 / 55%. Labels are sparse everywhere
(41 / 59 / 101), and VNM's labels are 91% NotFarm — so quantitative per-country
metrics here are nearly meaningless, which is what makes visual review the right
instrument.

## Headline impressions

1. **The model's high-confidence predictions in all three countries are visually
   excellent.** Every top-confidence Poultry prediction inspected in VNM and IDN was
   an unmistakable industrial farm complex. This is far better than the "regional
   collapse" framing (RUS/UKR/MYS) led us to expect for Southeast Asia.

2. **Vietnam's Pigs head actually works in deployment** — a surprise, given the gen
   slice has zero Pigs support. Top pig predictions showed the classic signature:
   barn blocks plus dark or **reddish-brown anaerobic effluent lagoons**. The model
   has evidently learned the lagoon association. (VNM_cluster_4332, 5843, 6825.)

3. **The dominant model-vs-label disagreement in VNM is protected horticulture**:
   shade-net houses and greenhouse rows that mimic barn geometry and spectral
   signature (VNM_2386 @0.997, 8176 @0.964, 1035 @0.940 — all labeled NotFarm).
   Same failure family as the documented ALB/COD plastic-cover problem.

4. **The Philippines' problem is false negatives, not false positives.** Six of ten
   sampled disagreements are labeled-Poultry farms the model rejects — including
   PHL_4056 at p_farm **0.007** despite two obvious broiler houses (and it's a train
   row!). The shared trait: barns embedded in coconut grove/jungle or cluttered
   peri-urban mosaic, unlike the open-farmland context that dominates training data.
   The FP archetype is **schools** — long single-storey classroom blocks
   (PHL_1022 @0.79).

5. **The candidate stage misses real complexes.** In the Đồng Nai-area grid, two
   visible livestock complexes — a white-barn block and a green-roofed complex with
   a red lagoon — sit **930–1,300 m from any candidate**. First direct field
   evidence for the E2.7 concern (candidate-stage recall is unmeasured and bounds
   everything).

## Per-country detail

### Vietnam (n=1,910; 55% predicted farm)
- **Right:** all 6 inspected top-Poultry sites (e.g. VNM_33: eight-barn complex in
  Red River rice paddies; VNM_4311: two fenced integrator compounds, CP-style) and
  all 3 inspected top-Pigs sites (red/dark lagoons). NotFarm top picks were a
  coastal theme-park development and a Phú Quốc-style beach resort — correct, and
  revealing: **shophouse/rowhouse blocks are the main candidate-stage FP** (long
  narrow buildings pass the morphometric filter).
- **Contested:** every model-vs-label disagreement sampled is model=Poultry vs
  label=NotFarm on shade-net/greenhouse rows. Two readings: (a) the model is fooled
  by protected horticulture; (b) some "NotFarm" labels on dark-roofed rows may
  themselves be wrong — several look genuinely livestock-like. Worth double-blind
  re-annotation before treating all as model errors (feeds E0.4).
- **Landscape note:** Đồng Nai / Xuân Lộc is saturated — within one 2.7 km box:
  two huge covered pig complexes (both correctly Pigs @0.99+), several more barn
  blocks, shade-net arrays, and the two uncovered complexes above. VNM_4372
  (p_farm 0.13) is ~370 m from a dark-roofed complex of ambiguous type
  (shade-net vs barns) — unresolved.
- **Research context:** Đồng Nai is Vietnam's pig+chicken capital; the industry is
  consolidating from smallholder to integrator closed-house systems (C.P. Vietnam
  dominant; BaF building multi-storey "high-rise" pig farms — a form our template
  won't match). Biogas digesters + sedimentation/biological ponds are standard on
  large farms → the lagoon signature will stay reliable for pigs.

### Indonesia (n=2,442; 64% predicted farm)
- **Right:** top-Poultry picks are classic Javanese *kandang* rows in rice paddies —
  mixed open-house (dark/rusty roofs, bamboo-and-wood) and closed-house (bright
  white metal) types visible side by side (IDN_2662, 10062). NotFarm top picks
  include **lake aquaculture cage rafts** (IDN_1988) — right call; another
  candidate-stage FP family.
- **Wrong/suspect:** IDN_2189 (Pigs @0.93, Batam) shows **no structure at all** at
  the location in current basemap imagery — mislocation, stale imagery, or
  candidate artifact. IDN_5125 (pred Pigs @0.83) is in **Lampung's cattle-feedlot
  belt** and is labeled Cattle — the known unlearnable-Cattle problem visible in
  deployment: the model reassigns big-roofed non-poultry livestock to Pigs.
  IDN_7103 (Poultry @0.57 vs NotFarm) is a school/barracks compound — barracks-row
  FP archetype.
- **Priors to apply:** pigs are regionally confined (North Sumatra, Bali, NTT,
  Batam); Java pig predictions deserve skepticism. Cattle feedlots (Lampung,
  East Java) will systematically surface as "Pigs".
- **Research context:** broiler housing is split between open-house
  (*kandang terbuka*, wood/bamboo, cheap, often elevated) and the growing
  closed-house segment. Open-house kandang are exactly the low-contrast rusty-roof
  structures that are hard at 10 m; closed-house conversions look like Thai/US
  training farms and will be easy.

### Philippines (n=2,344; 77% predicted farm — highest of the three)
- **Right:** top predictions are real (PHL_1053 upland barns; PHL_285 five-barn
  block). 77% farm-rate looks high but was not contradicted by the sample.
- **Wrong — the FN cluster:** 6/10 disagreements are missed labeled farms, all
  vegetation-embedded (PHL_4056 @0.007!, 3929 @0.28: barns tucked into coconut
  grove/jungle clearings). Hypotheses: context mismatch with training farmland,
  plus chronic cloud degrading the 2023 S2 median in uplands (PHL was a major
  cloud-recovery country in full-world scoring). The E2.1 threshold change
  (0.5→0.4 OOD) would rescue some but not the confident misses.
- **FP archetype:** schools (PHL_1022) — the Philippines' long single-storey
  classroom blocks are near-perfect barn mimics.
- **Research context:** broiler production is integrator-driven (San Miguel/
  Magnolia, Bounty Fresh) via contract growing; the modern segment is
  tunnel-ventilated mega-farms (e.g. San Miguel's Mindanao complexes) — easy
  targets. The long tail of older open-sided grower houses in tree cover is the
  hard part, and it is exactly what the model currently misses.

## What "unseen" farms look like
Modern complexes visible in basemap imagery but absent from the candidate set are
NOT a new form: they are the same standardized parallel-barn template (the two
uncovered VNM complexes look like every covered one). The miss mechanism is
therefore upstream — Open Buildings footprint gaps / construction newer than the
footprint vintage — not a template mismatch. One genuinely new form to watch:
multi-storey pig buildings (BaF "high-rise" farms), which break the
long-low-barn template entirely.

## Implications
1. **E2.7 (candidate-stage recall) is confirmed as the binding constraint** — first
   direct examples of uncatalogued complexes at ~1 km from coverage. A grid-sweep
   recall audit in Đồng Nai would quantify it cheaply.
2. **Add hard-negative classes to labeling:** shade-net/greenhouse rows (VNM),
   schools/barracks (PHL, IDN), shophouse strips, aquaculture (cages + coastal
   ponds). These four families explain nearly every FP inspected.
3. **Vegetation-embedded farms need attention** (PHL FNs): candidates exist, so
   this is a model/imagery problem — plausibly helped by better cloud handling
   (E1.3) more than by thresholds.
4. **Type predictions:** Poultry-vs-Pigs is credible where lagoons are visible;
   Cattle predictions are non-functional (Lampung feedlots → "Pigs"), consistent
   with the merge-Cattle recommendation.
5. **VNM label quality:** several NotFarm labels on barn-like rows deserve
   re-annotation before being trusted as model errors (E0.4).

## Sources
- [Đồng Nai remains Vietnam's pig & chicken capital](https://www.vietnam.vn/en/dong-nai-tiep-tuc-la-thu-phu-chan-nuoi-heo-ga-cua-ca-nuoc-sau-sap-nhap-tinh)
- [Vietnam's pig farms an environmental nightmare (Mekong Eye)](https://www.mekongeye.com/2022/10/10/vietnams-pig-farms-an-environmental-nightmare)
- [Vietnam pig farming 2025: C.P. leads, BaF high-rise farms (Vietdata)](https://www.vietdata.vn/post/vietnam-s-pig-farming-in-2025-c-p-leads-the-value-chain-masan-expands-retail-baf-accelerates-hig)
- [BaF Hai Ha mega farm](https://baf.vn/en/baf-puts-into-operation-hai-ha-livestock-mega-farm-the-largest-complex-in-northern-viet-nam/)
- [Đồng Nai: environmentally friendly livestock (biogas, closed barns)](https://dongnai.gov.vn/en/news/Economic-Data-Social/businesses-are-at-the-forefront-of-developing-environmentally-friendly-livestock-farming-6314.html)
- [Open vs closed house broiler performance, East Java](https://ejournal.unib.ac.id/index.php/jspi/article/view/12257)
- [Kandang open house construction (Chickin)](https://chickin.id/blog/kandang-ayam-broiler-open-house/)
- [Philippine broiler market trends (USDA FAS)](https://apps.fas.usda.gov/newgainapi/api/Report/DownloadReportByFileName?fileName=Philippine+Broiler+Market+Trends+and+Prospects_Manila_Philippines_03-23-2020)
- [San Miguel broiler contract growing](https://www.sanmiguelfoods.com/business-opportunities/broiler-contract-growing)
- [Massive investment in Philippines' poultry production (WATTPoultry)](https://www.wattagnet.com/regions/asia/article/15636373/massive-investment-in-philippines-poultry-production)
