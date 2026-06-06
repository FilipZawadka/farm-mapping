# Selection pipeline analysis — IF vs CNN vs combinations

All numbers below are reproduced from [`notebooks/selection_pipeline_analysis.ipynb`](../notebooks/selection_pipeline_analysis.ipynb), which is executed end-to-end with embedded plots. This markdown is the readable distillation — the notebook is the source of truth.

## Setup

| | |
|---|---|
| Source candidates | `data/rachel_geometry_candidates/all_countries/all_clusters.parquet` |
| IF scores | `template_score_if` column in the candidates parquet (Isolation Forest on building-cluster geometric features) |
| CNN scores | `data/output/baseline_v2_all_clusters/scored_candidates.parquet`, column `predicted_score` (baseline_v2 binary farm-vs-not-farm) |
| Evaluation set | 15,489 labeled clusters present in both — joined on `cluster_id` |
| Binary ground truth | `is_farm = (final_label != 'NotFarm')` |
| 3-class label | `Poultry` if `final_label` contains "Poultry", `NotFarm` if NotFarm, else `OtherFarm` |

---

## Q1 — How clean are Rachel's initial selection criteria?

The geometric clustering is already heavily enriched for farms before any modeling runs.

| Class | n | Share |
|---|---:|---:|
| Poultry | 9,247 | **59.7 %** |
| OtherFarm (Pigs + Cattle + Mixed/Other) | 2,626 | **17.0 %** |
| NotFarm | 3,616 | **23.3 %** |
| **Total** | **15,489** | 100 % |

**Implication.** A trivial "always-predict-farm" classifier already achieves **precision 0.767 at recall 1.0** on this pool. That's the bar any downstream re-ranker has to beat.

The 23 % NotFarm fraction is what every model below is trying to identify and remove.

---

## Q2 — Does the Isolation Forest help?

### Score distribution by class

`template_score_if` is signed; higher values indicate "more farm-like" buildings according to IF's anomaly model. There's clean separation in the means but the distributions overlap heavily.

| Class | n | mean | std | 25 % | median | 75 % |
|---|---:|---:|---:|---:|---:|---:|
| NotFarm | 3,616 | **−0.023** | 0.087 | −0.080 | −0.021 | 0.036 |
| OtherFarm | 2,626 | **+0.055** | 0.091 | −0.005 | 0.065 | 0.134 |
| Poultry | 9,247 | **+0.123** | 0.071 | 0.089 | 0.142 | 0.177 |

The poultry/not-farm mean gap is about 1.7 standard deviations — meaningful but far from separable.

### Headline metrics

| Metric | Value |
|---|---:|
| ROC AUC | **0.862** |
| Average Precision (PR AUC) | **0.952** |

### Precision at fixed recall

| Target recall | Threshold | Precision | Δ vs always-farm (0.767) |
|---:|---:|---:|---:|
| 0.99 | — | 0.783 | +1.6 pp |
| 0.95 | −0.0644 | 0.818 | +5.1 pp |
| 0.90 | — | 0.865 | +9.8 pp |
| 0.80 | — | 0.923 | +15.6 pp |
| 0.70 | — | 0.953 | +18.6 pp |

### Confusion matrices at operating points

**IF at threshold = −0.064 (recall ≈ 0.95)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 1,100 | 2,516 |
| True Farm | 594 | 11,279 |

Precision = 0.818, Recall = 0.950, F1 = 0.879. To keep 95 % of farms, IF lets through ~70 % of the not-farms.

**IF at threshold = −0.020 (max F1 = 0.884)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 1,815 | 1,801 |
| True Farm | 1,043 | 10,830 |

Precision = 0.857, Recall = 0.912, F1 = 0.884.

**Reading.** IF gives a real lift, but at near-perfect recall the gain is small (~1-5 pp over the always-farm baseline). It only meaningfully helps if you're willing to drop 5-20 % of true farms.

---

## Q3 — Does the CNN help?

### Score distribution by class

CNN binary probability — much sharper bimodality than IF.

| Class | n | mean | std | 25 % | median | 75 % |
|---|---:|---:|---:|---:|---:|---:|
| NotFarm | 3,616 | **0.253** | 0.300 | 0.025 | 0.107 | 0.405 |
| OtherFarm | 2,626 | **0.924** | 0.155 | 0.939 | 0.986 | 0.996 |
| Poultry | 9,247 | **0.944** | 0.120 | 0.953 | 0.989 | 0.998 |

NotFarms are pushed strongly toward 0; farms (both poultry and other) are pushed strongly toward 1 with very long left tails — the model is confident on most patches and uncertain on a small minority.

### Headline metrics

| Metric | Value |
|---|---:|
| ROC AUC | **0.963** |
| Average Precision (PR AUC) | **0.979** |

### Precision at fixed recall

| Target recall | Threshold | Precision | Δ vs always-farm |
|---:|---:|---:|---:|
| 0.99 | — | **0.913** | +14.6 pp |
| 0.95 | 0.6900 | **0.959** | +19.2 pp |
| 0.90 | — | 0.971 | +20.4 pp |
| 0.80 | — | 0.981 | +21.4 pp |
| 0.70 | — | 0.987 | +22.0 pp |

### Confusion matrices at operating points

**CNN at threshold = 0.690 (recall ≈ 0.95)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 3,129 | 487 |
| True Farm | 594 | 11,279 |

Precision = 0.959, Recall = 0.950, F1 = 0.954. The CNN keeps the same farm recall as IF (95 %) but removes **86 % of the not-farms** instead of IF's 30 %.

**CNN at threshold = 0.544 (max F1 = 0.959)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 2,947 | 669 |
| True Farm | 321 | 11,552 |

Precision = 0.945, Recall = 0.973, F1 = 0.959.

**Reading.** At near-perfect recall (0.99) the CNN still keeps **91 % precision** — you can find almost every farm and only ~9 % of selections will be wrong.

---

## Q4 — Is one model clearly better than the other?

**Yes — the CNN dominates IF at every operating point.** The CNN's PR curve sits strictly above IF's; there is no regime in which IF is preferable.

### Same-recall precision comparison

| Recall | IF precision | CNN precision | CNN advantage |
|---:|---:|---:|---:|
| 0.99 | 0.783 | **0.913** | **+13.0 pp** |
| 0.95 | 0.818 | **0.959** | **+14.1 pp** |
| 0.90 | 0.865 | **0.971** | **+10.6 pp** |
| 0.80 | 0.923 | **0.981** | **+5.8 pp** |
| 0.70 | 0.953 | 0.987 | +3.4 pp |

### Headline metric comparison

| | IF | CNN | CNN − IF |
|---|---:|---:|---:|
| ROC AUC | 0.862 | **0.963** | +0.101 |
| Average Precision | 0.952 | **0.979** | +0.027 |
| F1 at the model's optimum threshold | 0.884 | **0.959** | +0.075 |

CNN improves on IF by ≈10 ROC AUC points, and **the precision advantage is biggest where you want it most — at the high-recall end**.

---

## Q5 — Combinations: averaging or sequential IF→CNN?

### Score averaging — small but real gain

Convex combination `score = w · IF_norm + (1 − w) · CNN`, where IF is min-max normalized to [0, 1]:

| `w_IF` | Average Precision |
|---:|---:|
| 0.0 (CNN only) | 0.9794 |
| 0.1 | 0.9882 |
| **0.2 (best)** | **0.9884** |
| 0.3 | 0.9880 |
| 0.4 | 0.9873 |
| 0.5 | 0.9862 |
| 0.6 | 0.9843 |
| 0.7 | 0.9810 |
| 0.8 | 0.9751 |
| 0.9 | 0.9661 |
| 1.0 (IF only) | 0.9524 |

**Best combination: ~80 % CNN + ~20 % IF → AP 0.9884, an improvement of +0.9 pp over CNN alone.**

Confusion matrices for the best combination:

**Combo at threshold = 0.689 (recall ≈ 0.95)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 3,160 | 456 |
| True Farm | 594 | 11,279 |

Precision = 0.961, Recall = 0.950, F1 = 0.956. **31 fewer false positives** than CNN alone at the same recall.

**Combo at threshold = 0.572 (max F1 = 0.960)**

| | Predicted NotFarm | Predicted Farm |
|---|---:|---:|
| True NotFarm | 2,989 | 627 |
| True Farm | 337 | 11,536 |

Precision = 0.948, Recall = 0.972, F1 = 0.960.

### Sequential IF → CNN — does NOT help

The "use IF as a pre-filter, then re-rank with CNN" approach: drop the bottom `q %` by IF score, then evaluate CNN AP on the survivors.

| IF drop | Kept | Pool precision after IF | Farm recall after IF | CNN AP on survivors |
|---:|---:|---:|---:|---:|
| 0 % | 15,489 | 0.767 | 1.000 | 0.979 |
| 5 % | — | — | — | 0.981 |
| 10 % | 13,940 | 0.813 | 0.955 | 0.982 |
| 20 % | 12,391 | 0.864 | 0.902 | 0.984 |
| 30 % | 10,842 | 0.906 | 0.828 | 0.987 |
| 40 % | 9,294 | — | — | 0.989 |
| 50 % | 7,745 | 0.963 | 0.628 | 0.992 |
| 60 % | 6,196 | — | — | 0.994 |
| 70 % | 4,648 | — | — | 0.996 |

The "CNN AP on survivors" rises only because the easy negatives are being thrown away — but IF also throws away farms at roughly the same rate. The `farm_recall_after_IF` column is the real cost.

At any **overall** recall, a directly-thresholded CNN beats the two-stage IF→CNN pipeline. The sequential filter is **strictly dominated** by simple thresholding.

---

## Per-country breakdown

Filtered to countries with ≥20 labeled samples and both classes (farm + not-farm) present:

| | Country | n | Farm rate | IF AUC | CNN AUC | In training |
|---:|---|---:|---:|---:|---:|---:|
| 1 | USA | 10,469 | 0.756 | 0.895 | **0.978** | ✓ |
| 2 | THA | 943 | 0.609 | 0.704 | **0.946** | ✓ |
| 3 | CHL | 505 | 0.867 | 0.725 | **0.939** | ✓ |
| 4 | BRA | 705 | 0.658 | 0.803 | **0.932** | ✓ |
| 5 | MEX | 2,508 | 0.885 | 0.821 | **0.907** | ✓ |
| 6 | AFG | 23 | 0.348 | 0.825 | 0.667 | ✗ (zero-shot) |

### Train vs zero-shot summary

|  | IF AUC mean | IF AUC median | CNN AUC mean | CNN AUC median |
|---|---:|---:|---:|---:|
| In training (5 countries) | 0.790 | 0.803 | **0.940** | 0.939 |
| Zero-shot (1 country with enough data) | 0.825 | 0.825 | 0.667 | 0.667 |

**Caveat.** Most long-tail countries have only ~10 inspected samples and overwhelmingly one class (mostly farms), so they don't pass the n≥20 + both-classes filter. The Afghanistan data point is suggestive but not conclusive on its own:

- IF appears to generalize about the same to unseen countries (it depends on geometric features that aren't country-specific).
- CNN appears to generalize less cleanly — its 0.67 AUC on Afghanistan is well below the 0.94 train-country average. This matches the earlier finding from the multi-class run that CNN performance degrades in countries it never saw, while IF (which doesn't depend on satellite imagery features specific to a country) holds up.

This is a hint that on truly novel geography, IF + CNN combination may be more robust than CNN alone — but the sample is too thin to conclude.

---

## Bottom line

1. **Use the CNN with a threshold tuned to your recall budget.** This is the single strongest recommendation.
   * Want every farm? Threshold tuned for recall=0.99 → 91 % precision.
   * Want a cleaner set? Recall=0.95 → 96 % precision.
   * Max F1 operating point: threshold ≈ 0.54, gets F1 ≈ 0.96.
2. **Mix in a small IF contribution if you want the last fraction of a point.** `0.2 · IF_norm + 0.8 · CNN` → AP 0.9884 vs 0.9794 (+0.9 pp). The combination shaves off about 30 false positives at the recall-0.95 operating point. Marginal — only worth it for paper-grade results or downstream tasks that are extremely precision-sensitive.
3. **Don't do sequential IF → CNN.** It throws away farm recall for no precision gain over a properly-thresholded CNN. The "use IF as a coarse filter" intuition doesn't survive empirical evaluation.
4. **IF on its own is still a real signal** (AUC 0.86) — useful as a cheap pre-ranker if compute is constrained, e.g. for triaging which clusters to extract patches for. But it's strictly worse than the CNN for the prediction task itself.
5. **Watch country generalization.** The CNN's huge advantage over IF holds up clearly on training countries; on a single zero-shot country with enough data (Afghanistan, n=23) IF actually beats CNN. The sample is too thin to draw a strong conclusion, but it's worth re-checking once more long-tail countries get fully-labeled candidate pools.

---

## Notes on scope

- The 15,489 evaluated here are the labeled subset of Rachel's clusters. The wider candidate pool (~101 k) is mostly unlabeled, so this analysis assumes the labeled subset is representative of class proportions in the wider pool. The labeled set's geographic skew (mostly USA + BRA + MEX + THA + CHL, plus ~10 inspected per long-tail country) carries through to the AUC/AP numbers — the global picture may shift if/when the long-tail gets fully labeled.
- The CNN used here is `baseline_v2` (binary farm-vs-not-farm). Comparable per-class numbers for the multi-class model are in [`notebooks/baseline_v2_multiclass_analysis.ipynb`](../notebooks/baseline_v2_multiclass_analysis.ipynb).
- All numbers are reproduced from the notebook by re-executing on the current data. Re-run cells to regenerate after any data update.
