# Rachel's evaluation metrics: inventory, port, and results for rounds 4 and 5

*2026-09-12. Source: Rachel Mason's CAFO-AI_v2 repository as pulled from Drive on
2026-09-12 (snapshot in `notebooks_rachel/`, gitignored; 460 commits, working-tree
files dated to 2026-09-10). Code: `training/country_metrics.py`,
`experiments/country_metric_plots.py`, `experiments/evaluate_country_metrics.py`.
Results: `experiments/results/country_metrics/summary.md` (+ per-run folders,
regenerable).*

Rachel's own summary of what she found most useful, which this work implements
verbatim:

> I found it most helpful to plot precision and recall vs threshold for each
> model, by country (excluding IDN, MOZ, PER), and calculate precision, recall,
> f1 score at a single, global threshold value. I used the threshold that
> maximised the mean, unweighted per-country f1 score for each model. I.e., I
> computed f1-score separately for each country at a range of candidate
> thresholds, took the average f1 over all countries at each t, then found the t
> that maximized that.

---

## 1. What her notebooks compute, and what was ported

Her evaluation lives in `evaluate.py` (dated 2026-09-06) and is driven by the
four `05_review-cnn-predictions-{1..4}.ipynb` notebooks (one per model delivery:
`model_v10_run6`, `run7`, `run9`, and a template for the next). Two further
notebooks compare models and summarise coverage.

| Where | What it computes | Status here |
|---|---|---|
| `evaluate.load_cnn_probs`, `setup_farm_eval` | Collapse the 11 labels to 6 truth classes (NotFarm / Poultry / Pigs / Cattle / GenericFarm / Ambiguous); `prob_Farm` = sum of the non-NotFarm class probabilities | **Ported** — `prepare()`, `task_spec()` |
| `evaluate.threshold_by_f1` | One global threshold: `mean` mode = argmax over a 200-point grid of the unweighted mean per-country F1; `pooled` mode = argmax F1 of pooled rows | **Ported** — `choose_threshold()`; verified identical to her loop (max curve difference 5e-5, from 4-decimal rounding) |
| `evaluate._pr_table` | Per-country precision / recall / F1 at the chosen threshold, on each country's reporting split (`eval` if it has one, else `generalization`) | **Ported** — `pr_table()`; P/R/F1 identical to sklearn |
| `evaluate.find_fp_fn` | Per-country FP, n_neg, FN, n_pos and the two rates over **all** labelled rows (train included), plus the rows to review | **Ported** — `fp_fn_summary()`; also a held-out-only variant [ours] |
| `evaluate._plot_pr_group` | Precision and recall vs threshold, one panel per country; `test` dashed, `eval`/`generalization` solid; guides at t and at 0.85 | **Ported** — `plot_pr_vs_threshold()` |
| `evaluate._plot_hist_group` | TP/TN filled, FP/FN outlined score histograms (counts + per-class density) | **Ported** — `plot_score_hist()` (held-out rows by default) |
| `evaluate._plot_fp_fn_rates` | FP rate sorted, FN rate overlaid, bars on < 5 labels faded, each bar annotated with its label count | **Ported** — `plot_fp_fn_rates()` |
| `evaluate.class_cm` | Poultry-gate confusion: truth class × (P(Poultry) ≥ t) per focal country on `eval` | **Ported** — `class_confusion()` |
| `evaluate.eval_set_table`, `cluster_fractions` | Composition of each country's reporting split (the two task universes; label mix) | **Ported** |
| Closing cell of each review notebook | Headline counts: candidates with P(farm) ≥ t and P(Poultry) ≥ t | **Ported** — `headline_counts()` (also unlabelled-only) |
| `06_compare_models.ipynb` | Model-vs-model at each model's own threshold: farms caught by both / lost / rescued, split into training+gen vs inference countries; lost fraction per country; paired score scatter | **Ported in part** — lost/rescued/Jaccard tables vs archived v9 (own thresholds and both at 0.4); scatter not drawn |
| `evaluate.country_overview`, `fp_rate_table` | Per-country cluster / label counts and FP rate grouped by continent and World-Bank income group | Not ported — needs `country_converter`, `wbgapi` and her frozen WB spreadsheet; a review aid, not a model metric. Easy to add with a static ISO→income table if wanted |
| `evaluate.sample_clusters_to_review`, `review_outcomes`, `06_results.unconfirmed_but_checkable`, `merge_viz_labels` | The FP-review loop: sample high-scoring unlabelled clusters per country, inspect them, fold verdicts back into labels | Not ported — label-collection workflow writing her `visual_labels.json` |
| `06_results.lic_farm_coverage`, `top_scored_labeling`, `farm_confirmation_map` | Label coverage by income group; rank of first unlabelled cluster among a country's top scorers; world map of confirmed-farm countries | Not ported — presentation of label coverage, not of a model |
| `evaluate.compare_rounds` | Split moves between deliveries (`for_analysis` → `round_2` → `round_3` → `round_4`) | Not ported as such — the cross-run script does the equivalent audit (which held-out rows each run trained on) and blanks them |
| `evaluate.record_missing_clusters` | Clusters with no Sentinel patch (157,102 − 154,908 = 2,194 here) | Not ported |
| `OLD_07_evaluate-models_new.ipynb` | Isolation-Forest vs CNN PR overlays per country; input/output label mismatch checks | Superseded by her own newer code |

Her recorded observations (notebook 1, on `model_v10_run6`) are worth keeping in
view because the new numbers reproduce every one of them: the curves are "really
quite flat"; a threshold of 0.75 gave precision ≥ 0.8 on every focal `eval`
split; scores are bimodal ("very high or very low"); the model generalizes well
to Bangladesh, less to Nigeria; poultry needs a high threshold ("I'd rather give
no farm type than one that has a good chance of being wrong"); poultry false
positives are a mixture of NotFarm and pig farms.

## 2. The method, precisely as implemented

* **Countries.** Focal USA, BRA, CHL, MEX, THA report on `eval` (their `test`
  split is drawn faintly alongside); generalization ALB, BGD, COD, IND, MAR, NGA
  report on `generalization`. IDN, MOZ, PER are her held-out countries and are
  excluded from threshold selection and every table (they have `generalization`
  rows in the round_5 file and are still excluded).
* **Farm task.** score = P(farm) = 1 − P(NotFarm). Positives are every
  `Farm: *` label, including Unknown / Mixed / Other / PigsOrPoultry
  (`GenericFarm`) — a high farm score on a farm the model cannot name is
  correct. `Ambiguous` and unlabelled rows are outside the universe.
* **Poultry task.** score = P(Poultry). Positives are the three poultry labels.
  Universe = {NotFarm, Poultry, Pigs, Cattle}, so a generic farm is neither a hit
  nor a miss.
* **Threshold.** `score ≥ t` for t on `np.linspace(0, 1, 200)`; F1 per country
  with `zero_division=0`; unweighted mean across the 11 countries; first argmax.
  The pooled variant is also reported.
* **Tables.** Per-country n, positive fraction, precision, recall, F1 at t;
  FP/FN counts and rates per country; poultry confusion on `eval`; headline
  counts on all 154,908 scored candidates.

Additions of ours are kept under separate keys and marked `[ours]` in code:

* `plateau` — the span of thresholds whose mean F1 is within 0.01 / 0.02 of the
  best; `mean_f1_at_fixed` — mean F1 at t = 0.4 (shipped OOD point), 0.5, 0.75.
* `per_country_loco` — each country scored at the threshold tuned on the other
  ten (the in-sample argmax is optimistic).
* `precision_floor` — her criterion restricted to thresholds where the mean
  per-country precision is ≥ 0.8 (the rule she applied by eye in notebook 1).
* `fp_fn_heldout` — her FP/FN table on held-out rows only (hers pools train).

## 3. Where it runs

**In the pipeline.** `training/inference.py` calls `country_metrics.write_report`
after saving `scored_candidates.parquet`, producing
`country_threshold_metrics.json` (~200–350 KB: thresholds, tables, the mean-F1
curves and the per-country P/R-vs-threshold curves on the grid, so every figure
can be redrawn from the JSON without the parquet) and logs the per-country
table. It uses the run's own labels and splits, so on a pod it is exactly
Rachel's number for that delivery. Failures are logged, never fatal.
`experiments/collect_results.py` fetches the file with the other metrics.

**Across runs.** `python3 experiments/evaluate_country_metrics.py` scores every
collected round run plus archived v6 and v9 (24 models, ~2 s each without
figures, ~5 min with) and writes `experiments/results/country_metrics/`:
`summary.md` / `summary.json`, four cross-run figures, and a folder per run with
the JSON and her eleven figures.

Because the label files differ between rounds (round_5 re-capped the splits:
train 21,888 → 15,187, eval 590 → 663, generalization 662 → 949, 272 label
fixes), and because any model evaluated on a later delivery has usually
trained on some of its held-out rows, the script scores all runs on **one**
reference file (default `all_clusters_v11.parquet`) and blanks the label of
every candidate that **any** evaluated run trained on. Every model is then
judged on identical rows none of them saw:

| reference split | blanked | kept |
|---|---|---|
| eval | 87 | 576 |
| generalization | 38 | 911 (incl. 272 IDN/MOZ/PER rows, excluded from tables) |
| test | 949 | 165 |
| qual_eval | 8,113 | 3,045 |

`--own-splits` scores each run on its own embedded labels instead (what the
pipeline file contains); `--labels v10` switches the reference.

### 3.1 Cross-check against her own saved outputs

Her four review notebooks were saved with outputs (2026-09-06). Matching their
headline counts against our scored parquets identifies which model each one
evaluated, and running our port on the same rows without blanking
(`--no-blank --labels v11`) reproduces her tables exactly:

| her notebook | model (her name) | our model | her farm t (mean / pooled) | ours | her poultry t | ours | her headline | ours |
|---|---|---|---|---|---|---|---|---|
| 05_review-1 | `model_v10_run6` | archived v6 | 0.623 / 0.905 | 0.623 / 0.905 | 0.090 | 0.090† | 116,485 farm ≥ t | 116,485 |
| 05_review-2 | `model_v10_run7` | archived v7 | 0.010 / 0.077 | — | 0.010 | — | 128,142 | — |
| 05_review-3 | `model_v10_run9` | archived v9 | 0.231 / 0.223 | 0.231 / 0.223 | 0.201 | 0.201 | 116,447 farm, 108,080 poultry | 116,447, 108,084 |
| 05_review-4 | (unnamed) | **round_4 arm A, seed 44** | 0.101 / 0.209 | 0.101 / 0.209 | 0.070 | 0.070 | 118,693 farm, 113,473 poultry | 118,699, 113,473 |

† v6 was checked on the blanked rows only; v7 is not in our archive. For v9 and
r4_a_s44 every one of the 22 per-country precision / recall / F1 rows in her
tables (farm and poultry) is identical to ours to the two decimals she prints;
the headline counts differ by 4–6 clusters of 154,908, which is float rounding
in the CSV she received. So the numbers in this document are her numbers.

Two things the cross-check makes visible:

* **Her most recent review (notebook 4) is of round_4 arm A seed 44**, one
  of the 18 round_4 runs published to the site, not v9 and not the arm the
  AUC campaign would pick. Its per-country farm table under her method is
  USA 0.94 · BRA 0.97 · CHL 0.93 · MEX 0.93 · THA 0.90 · BGD 0.93 · IND 0.82 ·
  MAR 0.89 · NGA 0.88 · COD 0.85 · ALB 0.44, against v9's 0.95 · 0.96 · 0.93 ·
  0.95 · 0.90 · 0.92 · 0.80 · 0.92 · 0.88 · 0.79 · 0.65.
* **Her eval rows are not all held out from the models she scores them on.**
  Her notebooks overlay the current split files onto an older model's CSV
  (`hack_gen_country_info`), and round_5 moved rows between splits. Of the
  663 `eval` rows in the current files, 87 were in v9's training set and the
  same 87 in every round_4 run's; of the 949 `generalization` rows, 32 (v9)
  and 38 (round_4) were. This is why the cross-run tables in §4 blank those
  rows; on the blanked rows v9's mean farm F1 is 0.879 vs 0.878 unblanked,
  i.e. the leak is small, but it is there.

### 3.2 What her precision/recall-vs-threshold panels show

Reading her saved figures for v9 and for r4_a_s44 (and ours, which are
identical):

* **Focal countries, farm.** Precision is flat at 0.85–0.95 across almost the
  whole threshold range and recall stays above 0.95 until t ≈ 0.7 (USA, CHL,
  MEX) or ≈ 0.5 (BRA, THA). This is the "quite flat" she noted, and it is why
  the argmax is nearly arbitrary: any t between 0.05 and 0.6 gives the same
  table to two decimals.
* **Generalization countries, farm.** BGD, MAR and NGA have precision ≥ 0.9
  everywhere with recall decaying roughly linearly with t, so a low threshold
  is right for them. IND has a precision *ceiling* of ~0.75 at every
  threshold for every model, which no threshold fixes (either the labels or
  the imagery there confuses farms with something else). COD is noisy (45
  rows). ALB's precision never exceeds 0.6 below t ≈ 0.85 for v9, and for
  r4_a_s44 it climbs steadily from 0.15 to 0.7 with no plateau at all, so ALB
  alone would want a threshold near 0.9.
* **Poultry.** The curves are not flat: precision and recall cross at t ≈ 0.4–0.6
  in BRA, CHL and MEX, and THA's poultry precision never reaches 0.85. The
  poultry gate therefore has to be high and costs recall, exactly her point
  that farm-type should be withheld rather than guessed.
* **Her FP/FN-rate bar chart** pools every labelled row, train included, so the
  trained-on countries sit at ~0 and the chart mostly ranks inference
  countries with a handful of labels. Ours restricts it to held-out rows
  (`fp_fn_heldout`), which is the honest version for judging a model; hers
  remains the right view for choosing which countries to review next.

## 4. Results

Eleven countries; reference rows after blanking (farm-task rows / farm
positives): USA 103/82, BRA 98/90, CHL 135/96, MEX 128/91, THA 94/64, ALB
136/18, BGD 169/154, COD 45/24, IND 98/65, MAR 94/87, NGA 82/70.

### 4.1 Farm vs not-farm, per arm (mean ± sd over seeds)

| arm | description | seeds | t (mean-F1) | mean F1 | loco F1 | F1 @ 0.4 | mean P | mean R | t (P ≥ 0.8) |
|---|---|---|---|---|---|---|---|---|---|
| v6 | archived v6 (round_1 labels) | 1 | 0.623 | 0.855 | 0.842 | 0.849 | 0.812 | 0.958 | 0.623 |
| v9 | archived v9 (production; round_3 labels) | 1 | 0.231 | **0.879** | **0.872** | **0.868** | 0.844 | 0.936 | 0.231 |
| r4_a | baseline (v9 recipe, round_4 labels) | 3 | 0.106 ± 0.073 | 0.858 ± 0.002 | 0.849 ± 0.001 | 0.839 ± 0.012 | 0.814 ± 0.020 | 0.945 ± 0.033 | 0.116 ± 0.059 |
| r4_b | freeze0 | 3 | 0.054 ± 0.010 | 0.868 ± 0.013 | 0.861 ± 0.011 | 0.843 ± 0.022 | 0.816 ± 0.011 | 0.953 ± 0.005 | 0.054 ± 0.010 |
| r4_c | 6 bands | 3 | 0.052 ± 0.039 | 0.860 ± 0.004 | 0.853 ± 0.003 | 0.821 ± 0.006 | 0.808 ± 0.014 | 0.957 ± 0.017 | 0.055 ± 0.035 |
| r4_d | freeze0 + 6 bands | 3 | 0.137 ± 0.041 | 0.876 ± 0.011 | 0.870 ± 0.011 | 0.850 ± 0.009 | 0.835 ± 0.007 | 0.945 ± 0.004 | 0.137 ± 0.041 |
| r4_e | DenseNet-121 | 3 | 0.055 ± 0.026 | 0.859 ± 0.007 | 0.835 ± 0.024 | 0.837 ± 0.015 | 0.812 ± 0.014 | 0.955 ± 0.018 | 0.060 ± 0.023 |
| r4_f | freeze5 + full-LR unfreeze | 3 | 0.069 ± 0.089 | 0.865 ± 0.004 | 0.860 ± 0.006 | 0.837 ± 0.005 | 0.820 ± 0.004 | 0.946 ± 0.005 | 0.069 ± 0.089 |
| r5_a | round_5 labels, no sampler | 1 | 0.111 | 0.865 | 0.864 | 0.843 | 0.815 | 0.957 | 0.111 |
| r5_g | grouped countries, class-cond. | 1 | 0.095 | 0.880 | 0.866 | 0.860 | 0.838 | 0.945 | 0.095 |
| r5_h | 3 buckets us/europe/rest | 1 | 0.141 | 0.878 | 0.868 | 0.849 | 0.845 | 0.928 | 0.141 |
| r5_i | per-country cap 20 % | 1 | 0.111 | **0.882** | 0.861 | 0.867 | 0.837 | 0.956 | 0.111 |

Per-country F1 at each run's own threshold:

| arm | USA | BRA | CHL | MEX | THA | ALB | BGD | COD | IND | MAR | NGA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 0.94 | 0.96 | 0.92 | 0.90 | 0.88 | 0.24 | 0.98 | 0.84 | 0.82 | 0.96 | 0.97 |
| v9 | 0.94 | 0.96 | 0.91 | 0.94 | 0.89 | **0.65** | 0.93 | 0.79 | 0.80 | 0.92 | 0.91 |
| r4_a | 0.92 ± 0.00 | 0.97 ± 0.00 | 0.91 ± 0.02 | 0.91 ± 0.01 | 0.90 ± 0.01 | 0.41 ± 0.03 | 0.94 ± 0.02 | 0.82 ± 0.03 | 0.82 ± 0.01 | 0.92 ± 0.03 | 0.91 ± 0.03 |
| r4_b | 0.93 ± 0.00 | 0.97 ± 0.01 | 0.91 ± 0.01 | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.49 ± 0.10 | 0.97 ± 0.01 | 0.80 ± 0.06 | 0.83 ± 0.01 | 0.91 ± 0.04 | 0.93 ± 0.01 |
| r4_c | 0.93 ± 0.01 | 0.97 ± 0.00 | 0.90 ± 0.01 | 0.91 ± 0.00 | 0.89 ± 0.03 | 0.43 ± 0.03 | 0.95 ± 0.02 | 0.82 ± 0.03 | 0.81 ± 0.03 | 0.94 ± 0.02 | 0.91 ± 0.01 |
| r4_d | 0.93 ± 0.01 | 0.97 ± 0.01 | 0.92 ± 0.00 | 0.93 ± 0.01 | 0.91 ± 0.03 | 0.54 ± 0.12 | 0.95 ± 0.02 | 0.82 ± 0.01 | 0.83 ± 0.01 | 0.91 ± 0.02 | 0.92 ± 0.02 |
| r4_e | 0.93 ± 0.00 | 0.97 ± 0.01 | 0.91 ± 0.02 | 0.93 ± 0.00 | 0.89 ± 0.01 | 0.42 ± 0.03 | 0.90 ± 0.04 | 0.80 ± 0.02 | 0.81 ± 0.01 | 0.94 ± 0.02 | 0.95 ± 0.01 |
| r4_f | 0.93 ± 0.00 | 0.97 ± 0.01 | 0.90 ± 0.01 | 0.93 ± 0.01 | 0.89 ± 0.02 | 0.45 ± 0.08 | 0.94 ± 0.02 | 0.83 ± 0.04 | 0.84 ± 0.01 | 0.93 ± 0.03 | 0.92 ± 0.02 |
| r5_a | 0.92 | 0.97 | 0.92 | 0.91 | 0.89 | 0.48 | 0.94 | 0.80 | 0.83 | 0.93 | 0.93 |
| r5_g | 0.92 | 0.96 | 0.93 | 0.93 | 0.90 | 0.61 | 0.93 | 0.80 | 0.83 | 0.92 | 0.95 |
| r5_h | 0.91 | 0.97 | 0.92 | 0.94 | 0.90 | 0.62 | 0.94 | 0.81 | 0.82 | 0.92 | 0.92 |
| r5_i | 0.92 | 0.96 | 0.93 | 0.94 | 0.88 | 0.62 | 0.94 | 0.84 | 0.84 | 0.90 | 0.93 |

### 4.2 Poultry vs rest, per arm

| arm | seeds | t (mean-F1) | mean F1 | loco F1 | F1 @ 0.4 | mean P | mean R | t (P ≥ 0.8) | F1 (P ≥ 0.8) | R (P ≥ 0.8) |
|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 0.035 | 0.775 | 0.746 | 0.753 | 0.688 | 0.955 | 0.568 | 0.738 | 0.713 |
| v9 | 1 | 0.201 | 0.819 | 0.785 | 0.782 | 0.793 | 0.874 | 0.276 | 0.804 | 0.828 |
| r4_a | 3 | 0.028 ± 0.010 | 0.811 ± 0.004 | 0.800 ± 0.012 | 0.762 ± 0.011 | 0.734 ± 0.011 | 0.960 ± 0.009 | 0.350 ± 0.085 | 0.772 ± 0.024 | 0.782 ± 0.049 |
| r4_b | 3 | 0.039 ± 0.020 | 0.820 ± 0.008 | 0.811 ± 0.012 | 0.770 ± 0.019 | 0.747 ± 0.011 | 0.947 ± 0.012 | 0.360 ± 0.196 | 0.774 ± 0.046 | 0.776 ± 0.083 |
| r4_c | 3 | 0.023 ± 0.013 | 0.812 ± 0.001 | 0.808 ± 0.005 | 0.754 ± 0.007 | 0.734 ± 0.008 | 0.958 ± 0.019 | 0.385 ± 0.096 | 0.758 ± 0.014 | 0.747 ± 0.030 |
| r4_d | 3 | 0.085 ± 0.061 | 0.826 ± 0.011 | 0.814 ± 0.011 | 0.780 ± 0.015 | 0.763 ± 0.015 | 0.935 ± 0.010 | 0.291 ± 0.142 | 0.806 ± 0.019 | 0.837 ± 0.035 |
| r4_e | 3 | 0.085 ± 0.075 | 0.817 ± 0.009 | 0.797 ± 0.016 | 0.782 ± 0.005 | 0.765 ± 0.017 | 0.924 ± 0.041 | 0.255 ± 0.044 | 0.802 ± 0.011 | 0.845 ± 0.017 |
| r4_f | 3 | 0.045 ± 0.039 | 0.822 ± 0.005 | 0.818 ± 0.008 | 0.781 ± 0.002 | 0.761 ± 0.011 | 0.933 ± 0.026 | 0.281 ± 0.143 | 0.792 ± 0.014 | 0.812 ± 0.022 |
| r5_a | 1 | 0.106 | 0.819 | 0.814 | 0.776 | 0.767 | 0.921 | 0.352 | 0.785 | 0.787 |
| r5_g | 1 | 0.080 | **0.833** | 0.823 | 0.806 | 0.772 | 0.928 | 0.231 | 0.821 | 0.858 |
| r5_h | 1 | 0.126 | 0.830 | 0.814 | 0.766 | 0.788 | 0.904 | 0.171 | **0.823** | 0.870 |
| r5_i | 1 | 0.055 | 0.830 | 0.809 | 0.785 | 0.758 | 0.951 | 0.246 | 0.820 | 0.836 |

### 4.3 What changes on the map (vs archived v9, both at t = 0.4)

On the 3,897 held-out labelled farms v9 catches 3,602; the round_4/5 models
catch 3,499–3,630, exchanging 78–213 farms each way. On the 612 held-out
NotFarm rows v9 flags 127; the others 92–142. On the 122,600 unlabelled
candidates each model flags 100,966–107,919 against v9's 106,921, with a
Jaccard overlap of 0.84–0.90 — i.e. **swapping any of these models for v9
changes roughly one flagged cluster in ten** regardless of which arm it is.
Full per-run tables (also at each model's own threshold) are in `summary.md`.

### 4.4 Figures

* `experiments/results/country_metrics/mean_f1_vs_threshold_{farm,poultry}.png`
  — mean per-country F1 vs threshold, one seed-averaged curve per arm.
* `experiments/results/country_metrics/country_f1_by_arm_{farm,poultry}.png`
  — per-country F1, one point per seed, dashed = v9.
* Per run (regenerable): `score_thresh_{task}_{eval,gen}.png` (her PR-vs-threshold
  panels), `prob_hist_{task}_{eval,gen}.png`, `fp_fn_rates_{task}_heldout.png`,
  `class_cm_poultry_eval.png`.

## 5. What the numbers say

1. **Her metric ranks the models the way the AUC campaign did.** v9 0.879,
   r4_d 0.876 ± 0.011, r5_g/h/i 0.878–0.882 (one seed each), r4_b 0.868 ± 0.013,
   r4_a 0.858 ± 0.002. The whole spread is 0.02 and the seed sd is ~0.01, so no
   round_4 lever is distinguishable from the baseline here either, and **v9
   stays the model to beat** — it is best or tied on mean F1, on the honest
   leave-one-country-out F1 (0.872) and at the shipped 0.4 threshold (0.868).
2. **The F1-argmax threshold is not an operating point.** For every model but
   v6 it lands between 0.01 and 0.23, moves by up to 0.15 between seeds of the
   same recipe (r4_a: 0.035 / 0.181 / 0.101), and at that value 75–81 % of all
   154,908 candidates are flagged as farms. The reason is structural: the focal
   `eval` sets are 70–90 % farms and the scores are bimodal, so F1 is maximised
   by accepting nearly everything and the mean-F1 curve is flat from ~0.03 to
   ~0.3 (see the plateau column in `summary.md` and the curve figure). A mean
   precision floor of 0.8 does not bind on the farm task (it is already met at
   the argmax), so if the threshold is to be chosen from these data the
   constraint must be per country (e.g. every country ≥ 0.8) or the choice
   must be made on a lower-prevalence slice. Her own by-eye choice of 0.75, and
   our shipped 0.4, both sit on the flat part with far fewer world flags.
3. **Albania is the hard case for every model**, and it is the only country
   that separates them. 13 % farm prevalence turns the models' FP rate into
   precision 0.27–0.49 and F1 0.24 (v6) to 0.65 (v9); the round_5 balanced arms
   reach 0.61–0.62 and r4_d 0.54 ± 0.12. Focal countries are saturated
   (F1 0.88–0.97, spread ≤ 0.02 between arms) and cannot tell recipes apart.
4. **Poultry needs a higher gate than farm**, exactly as she wrote: under the
   precision floor the poultry threshold is 0.17–0.39 and recall drops to
   0.71–0.87. r5_g/h/i and r4_d are best (F1 0.82–0.83 at the argmax, 0.80–0.82
   under the floor); Thailand is the weakest focal poultry country in all 24
   models (F1 0.64–0.78 at their own threshold, against 0.81–0.91 for the other
   four); the poultry gate's false positives on `eval`,
   summed over the focal countries, are NotFarm 46–93 and Pigs 37–47 per model
   (confusion figures) — the NotFarm + Pigs mixture she observed, with NotFarm
   somewhat ahead.
5. **Round_5 is still one seed per arm.** Its 0.878–0.882 are within one
   round_4 seed sd of r4_d, and the round_4 record shows what a single seed is
   worth (arm B: p = 0.003 → 0.29 once seeds two and three landed). The seed
   replication of H and I remains the next step.

## 6. Using it

```bash
# every collected run + archived v6/v9, with her figures (≈5 min)
python3 experiments/evaluate_country_metrics.py
# tables only; or one run; or each run on its own labels
python3 experiments/evaluate_country_metrics.py --no-plots --runs r5_
python3 experiments/evaluate_country_metrics.py --own-splits
# reproduce one of her notebooks exactly (current labels, nothing blanked; not leakage-safe)
python3 experiments/evaluate_country_metrics.py --no-blank --runs v9 --out /tmp/repro
```

On a pod every scoring run now leaves `country_threshold_metrics.json` beside
`scored_candidates.parquet`; the collector brings it back. Rachel's figures are
reproduced one-for-one (same panels, styles and guides), so a per-run folder can
be sent to her and compared directly with her Colab output for the same model.

*Not done here:* the paper (`paper/main.*`) does not yet describe this
secondary endpoint; the World-Bank income-group breakdowns and her FP-review
sampling loop were deliberately left in her repository.
