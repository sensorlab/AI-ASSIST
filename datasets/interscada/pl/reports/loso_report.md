# interscada/pl: leave-one-state-out evaluation (data drop v2, 2026-09-08)

## Scope

Leave-one-state-out evaluation of retrieval-based CCT estimation on the 780-state `interscada/pl` archive, run through the production service path (`build_estimation_service()`, embedded Qdrant, `DATASET_NAME=interscada/pl`). Two passes: `scripts/evaluation/eles_benchmark.py` for clearing-time accuracy and support diagnostics, `scripts/evaluation/crit_gen_confusion.py` for critical-generator identification. Every query excludes its own state from retrieval, and both scripts assert that exclusion held on every query; neither assertion fired across 2 x 780 queries.

## Data

The v2 drop is a pure superset of the previous one. All 761 earlier states are present with byte-identical load-flow rows, and all 47,943 shared (state, experiment) records carry identical CCT, fault location and critical generator. One new batch, `SET_0925`, adds 19 states, `Point_no. 2` through `20`. Totals: **780 states, 63 fault locations, 49,140 records**. Column structure is unchanged: 216 load-flow features, 47 topology columns, critical generators on buses 30-39.

Two facts about the archive shape matter for reading everything below. All 780 states fall in a **single topology group**, so the exact-topology filter never restricts a query and never causes an abstention; this dataset cannot exhibit the coverage behaviour that dominates the ELES results. And **11.4% of records (5,613) are stable**, reaching the 500 ms simulation ceiling with no critical generator.

## Clearing-time accuracy

| Quantity | Value |
|---|---|
| Records scored | 48,861 of 49,140 (**99.43%**) |
| MAE | **0.01281 s** |
| RMSE | 0.02167 s |
| Median absolute error | 0.00708 s |
| q90 / q95 / q99 | 0.03200 / 0.04452 / 0.07801 s |
| Maximum absolute error | 0.40085 s |
| Mean neighbours / n_eff | 44.9 / 10.3 |

Mean recorded CCT is 0.2965 s, so the MAE is **4.3% of the mean clearing time**. The 279 uncovered records (0.57%) are not topology abstentions, which are impossible here: they are locations where no retrieved state carried a candidate generator.

Conditioning on the recorded fault location is doing most of the work. Falling back to the single most likely location instead answers 100% of records but at MAE 0.07220 s, **5.6 times worse**.

### Error distribution

A mean absolute error hides the shape that decides whether an estimator is usable, so the full ladder, in seconds:

| Estimate | n | MAE | RMSE | q0 | q25 | q50 | q75 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recorded location | 48,861 | 0.01281 | 0.02167 | 0 | 0.00196 | 0.00708 | 0.01727 | 0.03200 | 0.04452 | 0.07801 | **0.40085** |
| most likely location | 49,140 | 0.07220 | 0.10961 | 0 | 0.01015 | 0.04181 | 0.10551 | 0.19372 | 0.25806 | 0.34182 | 0.44800 |
| recorded, n_eff below median | 24,430 | 0.01375 | 0.02362 | 0 | 0.00227 | 0.00715 | 0.01850 | 0.03471 | 0.04720 | 0.08617 | 0.40085 |
| recorded, n_eff at or above median | 24,431 | 0.01188 | 0.01952 | 0 | 0.00159 | 0.00701 | 0.01612 | 0.02924 | 0.04158 | 0.07170 | 0.35275 |

The distribution is heavily skewed: the median error of 7.1 ms is a little over half the mean, three quarters of records fall inside 17 ms, and the maximum is 0.401 s, **31 times the MAE**. This is a tight estimator with a thin tail of severe misses rather than a uniformly mediocre one, and the MAE alone conveys neither fact.

Splitting on effective sample size, the one support measure an operator could check before trusting an estimate, helps but does not explain the tail: thin support raises the MAE by 16% and the 90th percentile by 19%, while the worst case barely moves (0.401 s against 0.353 s).

## Support diagnostics

Each diagnostic sorts records from most to least trusted; the table gives MAE over all records and over the best-ranked half.

| Diagnostic | MAE, full | MAE, best 50% | Change |
|---|---|---|---|
| `cct_weighted_std` | 0.012705 | 0.007748 | **-39.0%** |
| `distance_norm` | 0.012812 | 0.009171 | -28.4% |
| `n_neighbors` | 0.012812 | 0.009980 | -22.1% |
| `distance_min` | 0.012812 | 0.010749 | -16.1% |
| `location_weight_mass` | 0.012812 | 0.011106 | -13.3% |
| `n_eff` | 0.012812 | 0.011879 | -7.3% |
| `distance_median` | 0.012812 | 0.013357 | +4.3% |
| `distance_mean` | 0.012812 | 0.013483 | +5.2% |
| `neighborhood_compactness` | 0.012705 | 0.014538 | **+14.4%** |
| `distance_spread` | 0.012812 | 0.015355 | +19.8% |

Disagreement among retrieved outcomes (`cct_weighted_std`) is the only strong signal. Four diagnostics rank estimates **worse than not sorting at all**, `neighborhood_compactness` and `distance_spread` most clearly, so neither should be shown to an operator as a confidence measure on this archive.

## Critical-generator identification

At each record's recorded fault location, candidate generators are ranked by `crit_gen_likelihood` (raw, never-renormalised kernel mass, the only quantity comparable across generator groups) and the top-ranked candidate is compared with the recorded outcome. Stable records are a genuine eleventh class rather than an absence: the archive stores a stable outcome as a row with a null generator, the service groups those rows like any other, and the ranking can therefore name "stable" as the most likely result.

| Quantity | Value |
|---|---|
| Top-1 accuracy | **78.22%** |
| Top-3 accuracy | **94.85%** |
| Top-1, unstable records only | 76.82% |
| Top-1, stable records only | 89.04% |
| Recorded generator absent from candidates | 0.57% |
| Mean candidates per record | 5.82 |

### Confusion matrix (rows recorded, columns predicted)

|  | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **30** | 663 | 309 | 51 | 23 | 149 | 93 | 43 | 169 | 142 | 19 | 159 |
| **31** | 67 | 7749 | 385 | 21 | 389 | 157 | 76 | 155 | 136 | 23 | 126 |
| **32** | 27 | 793 | 1471 | 10 | 152 | 56 | 30 | 48 | 24 | 5 | 29 |
| **33** | 23 | 28 | 13 | 520 | 209 | 165 | 36 | 25 | 7 | 6 | 0 |
| **34** | 93 | 329 | 85 | 186 | 11941 | 299 | 195 | 173 | 63 | 13 | 137 |
| **35** | 48 | 172 | 45 | 172 | 245 | 2021 | 330 | 94 | 37 | 12 | 40 |
| **36** | 41 | 85 | 17 | 52 | 246 | 329 | 1897 | 35 | 23 | 5 | 30 |
| **37** | 63 | 264 | 75 | 25 | 141 | 190 | 42 | 5170 | 109 | 23 | 194 |
| **38** | 80 | 287 | 51 | 1 | 125 | 25 | 32 | 50 | 1738 | 7 | 124 |
| **39** | 7 | 9 | 16 | 9 | 22 | 40 | 15 | 28 | 18 | 269 | 7 |
| **stable** | 23 | 159 | 74 | 8 | 42 | 64 | 19 | 139 | 77 | 10 | 4998 |

### Confusion matrix, row-normalised (% of each recorded class)

Counts are hard to read when class support spans an order of magnitude, so the same matrix with each row summing to 100. The diagonal is that class's recall.

|  | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **30** | **36.4** | 17.0 | 2.8 | 1.3 | 8.2 | 5.1 | 2.4 | 9.3 | 7.8 | 1.0 | 8.7 |
| **31** | 0.7 | **83.5** | 4.1 | 0.2 | 4.2 | 1.7 | 0.8 | 1.7 | 1.5 | 0.2 | 1.4 |
| **32** | 1.0 | 30.0 | **55.6** | 0.4 | 5.7 | 2.1 | 1.1 | 1.8 | 0.9 | 0.2 | 1.1 |
| **33** | 2.2 | 2.7 | 1.3 | **50.4** | 20.3 | 16.0 | 3.5 | 2.4 | 0.7 | 0.6 | 0.0 |
| **34** | 0.7 | 2.4 | 0.6 | 1.4 | **88.4** | 2.2 | 1.4 | 1.3 | 0.5 | 0.1 | 1.0 |
| **35** | 1.5 | 5.3 | 1.4 | 5.3 | 7.6 | **62.8** | 10.3 | 2.9 | 1.2 | 0.4 | 1.2 |
| **36** | 1.5 | 3.1 | 0.6 | 1.9 | 8.9 | 11.9 | **68.7** | 1.3 | 0.8 | 0.2 | 1.1 |
| **37** | 1.0 | 4.2 | 1.2 | 0.4 | 2.2 | 3.0 | 0.7 | **82.1** | 1.7 | 0.4 | 3.1 |
| **38** | 3.2 | 11.4 | 2.0 | 0.0 | 5.0 | 1.0 | 1.3 | 2.0 | **69.0** | 0.3 | 4.9 |
| **39** | 1.6 | 2.0 | 3.6 | 2.0 | 5.0 | 9.1 | 3.4 | 6.4 | 4.1 | **61.1** | 1.6 |
| **stable** | 0.4 | 2.8 | 1.3 | 0.1 | 0.7 | 1.1 | 0.3 | 2.5 | 1.4 | 0.2 | **89.0** |

Normalising exposes structure the counts hide. Bus 32 is misread as bus 31 in **30.0%** of its records, a specific confusion rather than diffuse error, and bus 33 leaks **20.3%** to bus 34 and **16.0%** to bus 35. Bus 30 stays the weak class at 36.4%, spread across nearly every other label rather than concentrated anywhere.

### Per class

| Class | Support | Recall | Precision | **F1** | Top-3 |
|---|---|---|---|---|---|
| 30 | 1,820 | 0.364 | 0.584 | **0.449** | 0.822 |
| 31 | 9,284 | 0.835 | 0.761 | 0.796 | 0.970 |
| 32 | 2,645 | 0.556 | 0.644 | 0.597 | 0.894 |
| 33 | 1,032 | 0.504 | 0.506 | **0.505** | 0.815 |
| 34 | 13,514 | 0.884 | 0.874 | **0.879** | 0.984 |
| 35 | 3,216 | 0.628 | 0.588 | 0.607 | 0.909 |
| 36 | 2,760 | 0.687 | 0.699 | 0.693 | 0.908 |
| 37 | 6,296 | 0.821 | 0.850 | 0.835 | 0.957 |
| 38 | 2,520 | 0.690 | 0.732 | 0.710 | 0.898 |
| 39 | 440 | 0.611 | 0.686 | 0.647 | 0.793 |
| stable | 5,613 | 0.890 | 0.855 | 0.873 | 0.987 |

**Macro F1 0.690, weighted F1 0.779**, against a top-1 accuracy of 0.782. The gap between the two averages is the imbalance: accuracy and weighted F1 are carried by the three largest classes, while macro F1 counts bus 30 (0.449) and bus 33 (0.505) as equal partners and drops nine points. Report macro when the question is whether every generator is identifiable, weighted when it is how often the answer is right.

Accuracy tracks support: the three largest classes (34, 31, 37) reach 0.82 to 0.88 recall while bus 30, with 1,820 records, reaches 0.364 and is confused with almost every other generator. Bus 33 is the weakest by precision (0.506), and its errors run overwhelmingly to buses 34 and 35. The confusion is not symmetric: bus 32 is mistaken for 31 in 793 records while 31 is mistaken for 32 in 385, so the ranking has a standing bias toward the larger class.

The top-1 versus top-3 gap (78.2% against 94.9%) says the recorded generator is nearly always among the leading candidates even when it is not first, which matters if the output is a shortlist for an operator rather than a single answer.

## Most critical location

The operational question a screening pass asks is not "how accurate is each estimate" but "which contingency do I simulate first". The most critical location is the one with the lowest recorded CCT. This is a different quantity from `location_likelihood`, which ranks locations by retrieved weight mass, that is by which location the query most resembles rather than which is most severe; `cct_weighted_global` is built from the most likely location and cannot stand in for this.

| Measure | Value |
|---|---|
| Exact location match | 65.64% |
| Chosen location attains the minimum recorded CCT | **83.59%** |
| Regret within one 10 ms label step | 87.69% |
| Mean / median regret | 0.00572 s / **0.00000 s** |
| 95th percentile / maximum regret | 0.03500 s / 0.29500 s |
| MAE on the state's minimum CCT | 0.01075 s |
| States with a tied minimum | 168 of 780 |

Exact name matching understates performance. CCT labels sit on a 10 ms grid and 168 states have several locations tied at the minimum, so picking either of two equally severe locations is scored as a miss. By the measure that reflects the decision, whether the chosen location actually carries the worst clearing time, retrieval is right in **83.6%** of states and the median regret is exactly zero.

Estimating the minimum itself is easier than the average record: MAE 0.01075 s against 0.01281 s overall. The tail is real, though. One state, `SET_0221__Point_no.___10`, gives up 0.295 s of clearing time, meaning screening was pointed at a location far from the true worst.

### Top-1 and top-3 screening

Ordering a state's 63 locations by estimated severity and simulating the first k gives the classification-style view of the same question. Regret is the clearing time given up by stopping at k, in seconds.

| k | Worst location found | Regret mean | q50 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|
| 1 | 83.72% | 0.00571 | 0 | 0.019 | 0.035 | 0.10121 | 0.295 |
| **3** | **97.56%** | 0.00054 | 0 | **0** | **0** | 0.02821 | 0.041 |

Three simulations out of 63, **4.8% of a full contingency screen**, reach a genuinely worst-case contingency in 97.6% of states. The regret quantiles are the stronger argument: at k=3 regret is exactly zero through the 95th percentile and the worst case falls from 0.295 s to 0.041 s, so the third simulation buys tail protection rather than average accuracy.

Larger k is not reported because it stops discriminating. A state holds 63 locations, but a recorded generator's candidate set averages under six, so a top-5 rate would mostly measure that a candidate list exists.

Top-k answers "is the worst one found", not "are all the unsafe ones found", and the two diverge sharply:

| Threshold | recall@1 | recall@3 | States affected |
|---|---|---|---|
| CCT < 0.10 s | 0.491 | 0.718 | 409 |
| CCT < 0.15 s | 0.278 | 0.492 | 533 |
| CCT < 0.20 s | 0.140 | 0.359 | 630 |

At 0.20 s, 630 of 780 states hold at least one unsafe location and usually several, so three picks recover 35.9% of them. That is arithmetic rather than a ranking failure: if the requirement is to screen out every unsafe contingency, the risk-coverage and referral-load view applies, not top-k.

### Location likelihood is not a fault-location prediction

The service also ranks locations by `location_likelihood`, that is by retrieved weight mass within a generator group. Measured against the recorded location, it is weak:

| Measure | Value |
|---|---|
| Recorded location ranks first | 9.94% |
| Recorded location in top 3 | 24.62% |
| Mean locations competing | 48.11 |
| Mean winning score | 0.0972 |
| Most likely location equals the estimated most critical location | **5.90%** |

Against a uniform baseline of about 2% over 48 competing locations, ranking first 9.9% of the time is better than chance and still far too weak to present as a fault-location prediction. The last row is the one that matters operationally: the most likely location and the most critical location coincide in under 6% of states, because similarity says which locations have well-supported neighbours, not which fault is severe. This is why the fallback estimate built on the most likely location carries 5.6 times the error, and why the fault location has to be supplied as an input rather than inferred.

## Are the new states different from the rest?

Yes, decisively, in the input space.

Nearest-neighbour distance to the 761 existing states, over 169 standardised continuous features:

| | median | mean | max |
|---|---|---|---|
| new (19) | **10.31** | 8.60 | 12.17 |
| existing (761) | 1.40 | 1.40 | 6.87 |

Mann-Whitney U, one-sided, **p = 1.1e-9**, and **17 of 19 new states lie beyond the 95th percentile** of the existing states' own nearest-neighbour distances. Per feature, **57 of 137 continuous features (41.6%) shift significantly** after Holm correction, several at Cohen's d near 2.7, concentrated in loads 15 and 16 and the 345 kV bus voltage angles.

The batch is a **trajectory, not a cloud**. Distance to the nearest existing state grows monotonically with the point number: Point 2 at 0.000, Point 3 at 0.671, Point 4 at 3.69, rising to 11.86 at Point 20. It begins exactly on an existing operating point and walks away from the archive. With the higher mean loading (3,679 MW against 3,293 MW) this reads as a deliberate stress sweep rather than a fresh random sample.

**One new state duplicates an existing one exactly.** `SET_0925__Point_no.____2` matches `SET_0025__Point_no.____2` on all 216 load-flow columns and all 63 CCT and critical-generator outcomes. Either the sweep re-simulated its starting point deliberately, or a state was exported twice; worth confirming with the data provider.

Retrieval support on the new states is unambiguously thinner, while the accuracy penalty is directionally consistent but underpowered at n = 19:

| | new | existing | p |
|---|---|---|---|
| Per-state MAE | 0.01587 | 0.01276 | 0.117 |
| `distance_min` | 3.16 | 1.86 | 0.0001 |
| `n_neighbors` | 22.2 | 45.6 | <0.0001 |
| `n_eff` | 4.42 | 10.49 | <0.0001 |
| Coverage | 98.83% | 99.45% | - |

Roughly half the neighbours at nearly double the distance, for 24% higher error that does not reach significance. Support degrades before conditional accuracy does, which is the same ordering the ELES analysis reports.

A useful negative detail: the duplicated state's own MAE is 0.0135 s, marginally **worse** than the dataset mean, despite having a distance-zero twin with identical outcomes in the reference pool. The estimate is a kernel-weighted mean over about 45 neighbours, so a perfect match contributes one unit of weight and is then diluted. Retrieval does not behave like a 1-nearest-neighbour lookup here.

## Limitations

- **Single topology group.** The exact-topology filter is inert on this archive, so nothing here tests abstention behaviour, and coverage figures are not comparable with ELES.
- **Leave-one-state-out excludes only the query state.** Sweep points within a `SET` family are plausibly adjacent operating points and remain retrievable. There are no timestamps, so no temporal exclusion control is possible; the 0.01281 s MAE should be read as an upper bound on accuracy under near-duplicate support.
- **Preprocessing is transductive.** The scaler is fitted and the index populated over all 780 states including the query. Exclusion applies to retrieval, not to preprocessing.
- **Class imbalance.** Generator classes span 440 to 13,514 records, so the aggregate top-1 rate is dominated by the largest classes; read the per-class table alongside it.
- **The variant tag `lines_only` in artifact names is an artefact** of the `TOPOLOGY_VARIANT` default. This dataset has one topology column set and it was used correctly; only the filename is misleading.

## Reproduction

```bash
uv run ai-assist-prepare interscada/pl

DATASET_NAME=interscada/pl QDRANT_URL=":memory:" \
    nice -n 19 uv run python scripts/evaluation/eles_benchmark.py

DATASET_NAME=interscada/pl QDRANT_URL=":memory:" \
    nice -n 19 uv run python scripts/evaluation/crit_gen_confusion.py
```

Runtimes on the shared workstation: 1,239 s and 1,286 s respectively, service build 1.7 s.

Artifacts, all regenerated by the commands above:

| Path | Content |
|---|---|
| `datasets/interscada/pl/reports/risk_coverage_interscada-pl_lines_only.csv` | risk-coverage curve per diagnostic |
| `datasets/interscada/pl/reports/critical_location_interscada-pl_lines_only.csv` | most-critical-location selection |
| `datasets/interscada/pl/reports/error_quantiles_interscada-pl_lines_only.csv` | absolute-error quantiles per estimate and support split |
| `datasets/interscada/pl/reports/topk_location_interscada-pl_lines_only.csv` | top-1 and top-3 severity ranking with regret quantiles |
| `datasets/interscada/pl/reports/topk_screening_recall_interscada-pl_lines_only.csv` | share of unsafe locations captured at top-1 and top-3 |
| `datasets/interscada/pl/reports/crit_gen_confusion_interscada-pl.csv` | confusion matrix, counts |
| `datasets/interscada/pl/reports/crit_gen_confusion_relative_interscada-pl.csv` | confusion matrix, row-normalised |
| `datasets/interscada/pl/reports/crit_gen_per_class_interscada-pl.csv` | per-class recall, precision, top-3 |
| `datasets/interscada/pl/reports/crit_gen_summary_interscada-pl.csv` | headline identification rates |
| `tmp/report-interscada-pl-lines_only.joblib` | per-record CCT predictions and diagnostics |
| `tmp/crit-gen-confusion-interscada-pl.joblib` | per-record generator rankings |
