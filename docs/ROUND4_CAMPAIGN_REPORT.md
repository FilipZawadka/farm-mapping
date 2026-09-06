# Round 4 Campaign Report (2026-08-21 → 08-27)

Six recipes × 3 seeds on the round-4 labels; 18 training runs + 18 full-world
scorings, ~$60. Full narrative and tables: the published report artifact; working
notes: `experiments/R4_RUN_NOTES.md`; pre-registration: `experiments/EVAL_METHODS.md`;
numbers: `experiments/results/r4_eval_full.txt` / `r4_evaluation.json`.

## Verdict
1. **No recipe change beat the baseline** — all 5 pre-registered contrasts null
   after Holm (gen: b>a p=.67, d>a p=.60, f>a p=.80, b>f p=.80, e>d p=.16).
2. **v9 keeps the production default**: gen AUC 0.8740 (within-country 0.8683)
   vs best arm D 0.8611 (0.8496). Pre-registered rule held.
3. **Round-4 labels (+73% train, absorbed qual_eval) traded OOD for in-domain**:
   arm A (byte-identical recipe) vs v9 = eval +0.0088, gen −0.0412.
4. **freeze0's historical "win" is a LR effect with a reliability cost**: the
   unfreeze branch hard-coded lr_scale=0.1, so freeze0 ≡ no-warm-up AND 10× LR.
   Arm F decomposed it: LR +0.0167, warm-up +0.0075. freeze5 arms sd .005–.009,
   freeze0/full-LR arms sd .027–.036 with a 1-in-3 collapsing seed.
5. **Seed variance is mostly cross-country calibration drift, not skill**:
   collapsed d_s44 has normal within-country AUC (0.847); seed-44 country offsets
   correlate 0.67–0.79 across B/D/F (shuffle+aug stream is a shared treatment,
   amplified by 10× LR). Within-country all arms ≈ 0.832–0.850; v9 0.8683.
6. **6 bands replicate as a clean null** (C≈A, lowest sd of any arm) → drop
   NDVI/NDBI/NDWI.
7. **DenseNet (E)**: 3-way confounded (arch × ImageNet init × 3.4× fewer params);
   best in-domain (eval 0.9342), last pooled OOD — in-domain-specialist signature.
8. Calibration poor OOD for every arm (ECE 0.16–0.20); MAR/IND weak for every
   recipe; Pigs never predicted OOD.

## Publication
All 18 published (154,908 pts each), v9 default. Incident: first publish missed
2,574 unscorable-label rows (training candidates dir vs keep_unscorable_labels);
also under-covered every eval slice and masked an Ambiguous→farm=1 evaluator bug.
Rescored, re-evaluated (no verdict changed; absolute eval AUC −0.005..−0.009
uniformly), republished, consumer-verified.

## Recommendations
Keep v9; adopt 6 bands; don't adopt freeze0/full-LR for single production runs;
label OOD-like countries next (MAR/IND profiles), not focal ones; report
within-country AUC + per-country thresholds; 3 seeds minimum; test per-worker
augmentation RNG seeding (current loader duplicates the stream across workers).
