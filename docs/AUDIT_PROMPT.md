# Critical audit prompt (results, data, full pipeline)

Paste into a fresh session at the repo root. Best run in plan mode first, and
with explicit permission to spend a few hours reading data.

---

You are performing an adversarial audit of this farm-mapping project. Your job is
NOT to summarize it — it is to find what is wrong, weak, or unexamined, and to
propose the highest-leverage improvements. Assume undiscovered errors exist;
this project has repeatedly found that its own conclusions rested on silent
defects (a patch/ID misalignment that poisoned a year of runs, a label-coverage
bug that shipped to production, a hard-coded LR cut that misattributed its best
finding). Your default posture toward every artifact, including the reports, is
distrust-and-verify.

## Ground rules

1. VERIFY, don't trust. Every number you rely on must be recomputed from primary
   data (`experiments/gpu_results/*/scored_candidates.parquet`,
   `data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet`),
   not quoted from a markdown file. Where your recomputation disagrees with a
   documented number, that disagreement is itself a finding.
2. Judge every effect against seed variance. Nothing under the practical floor
   (0.005 AUC) or inside the seed band matters. Use recipe-level estimands
   (mean over seeds), never a best seed. Methodology: `experiments/EVAL_METHODS.md`.
3. Every claimed defect needs a minimal reproduction (a few lines of pandas) and
   a severity: does it change a decision, a published number, or nothing?
4. Distinguish CONFIRMED (you reproduced it) from PLAUSIBLE (you suspect it).
   Never present a suspicion as a finding.
5. Do not fix anything. Report; fixes are a separate decision.
6. Anchor methods in the published literature. For every methodological judgment
   (evaluation design, metrics, calibration, accuracy assessment, spatial CV)
   the reference point is what comparable peer-reviewed work does — not first
   principles alone. Comparable work: CAFO/livestock-facility detection from
   remote sensing (e.g. Handan-Nader & Ho 2019, Robinson et al. poultry-CAFO
   mapping), land-cover map accuracy assessment (Olofsson et al. 2014),
   spatial cross-validation (Roberts et al. 2017, Ploton et al. 2020),
   classifier comparison (Dietterich 1998), calibration (Guo et al. 2017).
   The project's own survey lives in `paper/main.tex` + `paper/references.bib`
   (34 entries) — start there, then search for anything newer. When you flag a
   weakness, say what the standard published approach is; when you propose an
   improvement, cite the precedent for it. A deviation from published practice
   is not automatically wrong, but every unjustified deviation is a finding.

## What is already known — do not rediscover it, dig past it

- Seed sigma dominates; freeze0's "win" is a 10x-LR confound with a 1-in-3
  cross-country calibration failure; pooled OOD AUC entangles calibration drift
  with skill (within-country is ~3x more stable). `experiments/R4_RUN_NOTES.md`.
- Round_4 absorbed qual_eval (+73% train data); leakage of the three eval slices
  vs archived v6–v9 train sets was verified 0%.
- Fixed incidents: unscorable-label rows dropped from scoring (fixed, rescored);
  Ambiguous counted as farm-positive (fixed); dataloader workers share one
  augmentation RNG (documented, unfixed); patch-store ID misalignment (fixed in
  v10 era).
- All 5 pre-registered round_4 contrasts are null; v9 remains production.

## Audit areas, in priority order

A. **The unmeasured front end.** Candidate-stage recall has never been measured:
   a facility the morphometric geometry filter never proposes is invisible to
   every downstream metric. Estimate it (e.g. against known-farm registries in
   `data/`, OSM extracts, or the labeled parquets' provenance columns — how many
   labeled farms have NO candidate within 250 m?). This caps the whole system.
B. **Label quality.** final_label provenance mixes registries, visual inspection
   and OSM. Quantify: inter-source disagreement on co-located clusters, label
   age, the Farm:Unknown share per country, and whether "NotFarm" in weakly
   labeled countries means "inspected and rejected" or "never inspected".
C. **Evaluation-set validity.** test AUC saturates at 0.993 — is `test` too easy
   (near-duplicate clusters of training rows? spatial proximity? the 16.6%
   leakage finding)? Recompute distance-to-nearest-train for every test/eval/gen
   row (`experiments/lib.py` has the BallTree helper). Check the 200 rows added
   to generalization in round_4 (predict->labeled): who labeled them, and does
   their difficulty differ from the original 426?
D. **Imagery/patch integrity.** Sample patches per country: cloud fraction, date
   spread vs config window, alignment with cluster centroids, the 448 rows
   dropped for >250 m drift (who are they, is the threshold right?).
E. **Failure analysis where it hurts.** MAR ~0.65–0.76 and IND ~0.68–0.75 AUC for
   every recipe. Pull the 30 worst false positives/negatives per country from a
   scored parquet; characterize them (SMOD_L2 class, size, imagery quality,
   label source). Is it labels, imagery, or model?
F. **Statistics.** Re-derive one full contrast (b vs a on generalization) from
   parquets independently; check the SE_total formula, the Holm family, the
   Ambiguous exclusion, dedup on candidate_id, and whether per-country n>=20
   cutoffs hide anything.
G. **Literature alignment.** Stage by stage (candidate generation, labeling
   protocol, split design, training, evaluation, uncertainty reporting, map
   publication), identify how comparable published systems do it and where this
   project deviates. Specific checks: is accuracy reported the way map-accuracy
   good practice requires (area-adjusted estimates per Olofsson et al., which we
   do NOT currently produce)? Do comparable CAFO papers validate candidate
   recall against external registries (they do — and we never have)? Is our
   spatial leakage handling up to Roberts/Ploton standards? Would our headline
   claims survive review at a venue where those papers appeared?
H. **Deployment surface.** ECE 0.16–0.20 OOD while the site shows confidence
   tiers thresholded at 0.9/0.7/0.4 — quantify how misleading tiers are per
   country; propose per-country calibration if warranted.

## Deliverable

1. Findings table: id, area, CONFIRMED/PLAUSIBLE, severity (changes-decision /
   changes-number / cosmetic), one-line repro pointer.
2. The three most decision-relevant findings, in prose, with the evidence.
3. Ranked improvement plan: expected impact, cost (GPU $ / annotation hours /
   code), and what measurement would prove each improvement worked. Separate
   "do now" from "needs new data".
4. Explicitly list what you checked and found CLEAN — an audit that only reports
   problems is unfalsifiable.
