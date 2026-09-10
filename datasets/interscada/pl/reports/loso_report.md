# interscada/pl: leave-one-state-out evaluation (data drop v2, 2026-09-08)

## Scope

Leave-one-state-out evaluation of retrieval-based CCT estimation on the 761-state `interscada/pl` archive, run through the production service path (`build_estimation_service()`, embedded Qdrant, `DATASET_NAME=interscada/pl`). Two passes: `scripts/evaluation/eles_benchmark.py` for clearing-time accuracy and support diagnostics, `scripts/evaluation/crit_gen_confusion.py` for critical-generator identification. Every query excludes its own state from retrieval, and both scripts assert that exclusion held on every query; neither assertion fired across 2 x 761 queries.

## Data

The v2 drop is a pure superset of the previous one: all 761 raw states of the earlier drop are present with byte-identical load-flow rows, and every shared (state, experiment) record carries identical CCT, fault location and critical generator. One new batch, `SET_0925`, adds 19 states, `Point_no. 2` through `20`. Preparation then removes 19 duplicate states, 1,197 records, so the analysed archive is 761 states of which 18 come from the new batch. Totals: **761 states, 63 fault locations, 47,943 records**. Column structure is unchanged: 216 load-flow features, 47 topology columns, critical generators on buses 30-39.

Two facts about the archive shape matter for reading everything below. All 761 states fall in a **single topology group**, so the exact-topology filter never restricts a query and never causes an abstention; this dataset cannot exhibit the coverage behaviour that dominates the ELES results. And **11.5% of records (5,496) are stable**, reaching the 500 ms simulation ceiling with no critical generator.

## Methodology

### Protocol

Every state in the archive is used once as a query and scored against a reference library holding **all other states**. The query's own state is passed to the service as an excluded id, and both scripts assert on every query that no excluded state contributed to the returned estimate; the assertion never fired. Exact duplicates are removed at preparation, so a query cannot meet a copy of itself under another name.

Retrieval runs through the production path, `build_estimation_service()` with an embedded Qdrant, not a separate research harness. A query is scaled by the dataset's fitted `ColumnTransformer`, then matched against reference states **filtered to exact equality of the switching-status vector**, and the nearest **K = 100** of those by Euclidean distance in the scaled feature space are returned. Neighbours are grouped by recorded critical generator and fault location, and within a group each contributes weight `exp(-d)`, `d` being its distance to the query, normalised to sum to one. The estimate is that weighted mean of the neighbours' recorded CCT.

Preprocessing is transductive: the scaler is fitted and the index populated over the whole archive including the query. Exclusion applies to retrieval, not to preprocessing.


On this dataset a fault names a single critical generator, so a record is one `(state, location)` contingency and record-level and event-level figures coincide.

### Metrics

**Coverage** is the share of records receiving an estimate at their own recorded fault location. A record is uncovered when no retrieved state carries a candidate generator at that location. Where a dataset's topology key partitions the archive, a query with no compatible state abstains and is likewise uncovered.

**Absolute error** is `|recorded CCT - estimated CCT|` in seconds. Reported as MAE and RMSE, and as a quantile ladder q25 through q100, because a mean hides the shape: these estimators are tight in the bulk with a thin tail of severe misses, and the two are worth separating. The 0th quantile is omitted, being exactly zero everywhere measured.

Two estimates are scored separately. The **recorded-location** estimate conditions on the fault location, which a contingency screen enumerates and therefore knows. The **most-likely-location** estimate is the fallback that uses retrieval's own top-ranked location instead, and is reported to show what conditioning is worth.

**Support diagnostics** are scored by risk-coverage. Records are ranked from most to least trusted by each diagnostic, then MAE is recomputed over the best-ranked half. A diagnostic is informative if that MAE falls below the full-coverage MAE, and actively misleading if it rises, which means it ranks estimates worse than not sorting at all.

**Critical-generator identification** ranks the candidate generators at a record's recorded fault location by `crit_gen_likelihood`, the raw never-renormalised kernel mass, which is the only quantity comparable across generator groups. **Top-1** is the share of records whose recorded generator is ranked first, **top-3** the share ranked in the first three. Stable outcomes are a class rather than an absence: the archive stores a stable record with a null generator, the service groups those rows like any other, and the ranking can name "stable" as the most likely result.

**Per class**, recall is the share of a class's records identified correctly, precision the share of records predicted as that class that belong to it, and **F1** their harmonic mean. A class never predicted scores F1 0 rather than undefined. **Macro F1** averages classes equally and **weighted F1** by support; the gap between them measures how much the aggregate leans on the largest classes.

**Most critical location** is the fault location with the lowest recorded CCT, the contingency a screen should simulate first. Retrieval's choice is the location with the lowest estimated CCT. A hit credits **any** location tied at the recorded minimum, since CCT sits on a coarse label grid and choosing between two equally severe contingencies is not an error. **Regret** is the same question in seconds: the recorded CCT at the chosen location minus the true minimum, zero on a hit.

**Top-k screening** generalises that: order the state's locations by estimated severity, simulate the first k, and ask whether a worst-case contingency is among them. Only **k = 1 and k = 3** are reported; larger k stops discriminating, since a candidate set averages under six generators. **Screening recall** answers the different and harder question of what share of *all* locations below a clearing-time threshold the same k picks capture.

**Location likelihood** is reported separately from severity. `location_likelihood` ranks locations by retrieved weight mass, that is by which location a query resembles, not by which is most severe, and the two are compared explicitly rather than assumed to agree.

### What is not measured

No calibrated prediction intervals, no live-deployment performance (the protocol is retrospective and, absent timestamps on these archives, temporally non-causal), and no computational speed. The reported quality of a diagnostic is a ranking property, not a probability of error.

## Coverage and accuracy

| Quantity | Value |
|---|---|
| Records scored | 47,635 of 47,943 (**99.36%**) |
| MAE | **0.01358 s** |
| RMSE | 0.02278 s |
| Median absolute error | 0.00762 s |
| q90 / q95 / q99 | 0.03387 / 0.04676 / 0.08097 s |
| Maximum absolute error | 0.39634 s |
| Mean neighbours / n_eff | 45.2 / 10.7 |

Mean recorded CCT is 0.2960 s, so the MAE is **4.6% of the mean clearing time**. The 308 uncovered records (0.64%) are not topology abstentions, which are impossible here: they are locations where no retrieved state carried a candidate generator.

Conditioning on the recorded fault location is doing most of the work. Falling back to the single most likely location instead answers 100% of records but at MAE 0.07265 s, **5.3 times worse**.

### Error distribution

A mean absolute error hides the shape that decides whether an estimator is usable, so the full ladder, in seconds:

| Estimate | n | MAE | RMSE | q25 | q50 | q75 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|---|---|---|
| recorded location | 47,635 | 0.01358 | 0.02278 | 0.00218 | 0.00762 | 0.01834 | 0.03387 | 0.04676 | 0.08097 | **0.39634** |
| most likely location | 47,943 | 0.07265 | 0.11013 | 0.01047 | 0.04200 | 0.10623 | 0.19549 | 0.25935 | 0.34204 | 0.44800 |
| recorded, n_eff below median (6.0) | 23,817 | 0.01506 | 0.02521 | 0.00274 | 0.00825 | 0.02031 | 0.03721 | 0.04990 | 0.09350 | 0.32186 |
| recorded, n_eff at or above median | 23,818 | 0.01211 | 0.02005 | 0.00154 | 0.00709 | 0.01646 | 0.03007 | 0.04261 | 0.07261 | 0.39634 |

The distribution is heavily skewed: the median error of 7.6 ms is a little over half the mean, three quarters of records fall inside 18 ms, and the maximum is 0.396 s, **29 times the MAE**. This is a tight estimator with a thin tail of severe misses rather than a uniformly mediocre one, and the MAE alone conveys neither fact.

Splitting on effective sample size, the one support measure an operator could check before trusting an estimate, helps but does not explain the tail: thin support raises the MAE by 24% and the 90th percentile by 24%, and here the worst case moves the other way (0.322 s against 0.396 s), so the tail is not where thin support hurts.

## Support diagnostics

Each diagnostic sorts records from most to least trusted; the table gives MAE over all records and over the best-ranked half.

| Diagnostic | MAE, full | MAE, best 50% | Change |
|---|---|---|---|
| `cct_weighted_std` | 0.013418 | 0.008445 | **-37.1%** |
| `distance_norm` | 0.013584 | 0.009863 | -27.4% |
| `n_neighbors` | 0.013584 | 0.010243 | -24.6% |
| `distance_min` | 0.013584 | 0.011467 | -15.6% |
| `location_weight_mass` | 0.013584 | 0.012072 | -11.1% |
| `n_eff` | 0.013584 | 0.012109 | -10.9% |
| `distance_median` | 0.013584 | 0.013863 | +2.1% |
| `distance_mean` | 0.013584 | 0.013974 | +2.9% |
| `neighborhood_compactness` | 0.013418 | 0.015545 | **+15.9%** |
| `distance_spread` | 0.013584 | 0.016082 | +18.4% |

Disagreement among retrieved outcomes (`cct_weighted_std`) is the only strong signal. Four diagnostics rank estimates **worse than not sorting at all**, `neighborhood_compactness` and `distance_spread` most clearly, so neither should be shown to an operator as a confidence measure on this archive.

## Critical-generator identification

At each record's recorded fault location, candidate generators are ranked by `crit_gen_likelihood` (raw, never-renormalised kernel mass, the only quantity comparable across generator groups) and the top-ranked candidate is compared with the recorded outcome. Stable records are a genuine eleventh class rather than an absence: the archive stores a stable outcome as a row with a null generator, the service groups those rows like any other, and the ranking can therefore name "stable" as the most likely result.

| Quantity | Value |
|---|---|
| Top-1 accuracy | **76.66%** |
| Top-3 accuracy | **94.31%** |
| Top-1, unstable records only | 75.09% |
| Top-1, stable records only | 88.81% |
| Recorded generator absent from candidates | 0.64% |
| Mean candidates per record | 5.83 |

### Confusion matrix (rows recorded, columns predicted)

|  | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **30** | 447 | 319 | 60 | 26 | 164 | 107 | 50 | 175 | 150 | 21 | 168 |
| **31** | 71 | 7552 | 410 | 21 | 391 | 167 | 82 | 156 | 138 | 24 | 128 |
| **32** | 36 | 824 | 1264 | 12 | 156 | 71 | 32 | 50 | 24 | 5 | 31 |
| **33** | 25 | 33 | 15 | 449 | 214 | 179 | 38 | 24 | 7 | 6 | 1 |
| **34** | 104 | 333 | 87 | 196 | 11766 | 309 | 201 | 177 | 62 | 14 | 145 |
| **35** | 58 | 178 | 61 | 173 | 258 | 1749 | 353 | 97 | 41 | 13 | 39 |
| **36** | 44 | 86 | 22 | 54 | 252 | 360 | 1725 | 37 | 24 | 7 | 35 |
| **37** | 68 | 268 | 78 | 25 | 149 | 191 | 42 | 5074 | 108 | 25 | 197 |
| **38** | 88 | 288 | 52 | 1 | 128 | 27 | 32 | 51 | 1612 | 8 | 135 |
| **39** | 9 | 9 | 16 | 8 | 23 | 43 | 15 | 33 | 18 | 235 | 8 |
| **stable** | 24 | 162 | 76 | 5 | 43 | 66 | 20 | 134 | 75 | 10 | 4881 |

### Confusion matrix, row-normalised (% of each recorded class)

Counts are hard to read when class support spans an order of magnitude, so the same matrix with each row summing to 100. The diagonal is that class's recall.

|  | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **30** | **26.5** | 18.9 | 3.6 | 1.5 | 9.7 | 6.3 | 3.0 | 10.4 | 8.9 | 1.2 | 10.0 |
| **31** | 0.8 | **82.6** | 4.5 | 0.2 | 4.3 | 1.8 | 0.9 | 1.7 | 1.5 | 0.3 | 1.4 |
| **32** | 1.4 | 32.9 | **50.5** | 0.5 | 6.2 | 2.8 | 1.3 | 2.0 | 1.0 | 0.2 | 1.2 |
| **33** | 2.5 | 3.3 | 1.5 | **45.3** | 21.6 | 18.1 | 3.8 | 2.4 | 0.7 | 0.6 | 0.1 |
| **34** | 0.8 | 2.5 | 0.6 | 1.5 | **87.8** | 2.3 | 1.5 | 1.3 | 0.5 | 0.1 | 1.1 |
| **35** | 1.9 | 5.9 | 2.0 | 5.7 | 8.5 | **57.9** | 11.7 | 3.2 | 1.4 | 0.4 | 1.3 |
| **36** | 1.7 | 3.2 | 0.8 | 2.0 | 9.5 | 13.6 | **65.2** | 1.4 | 0.9 | 0.3 | 1.3 |
| **37** | 1.1 | 4.3 | 1.2 | 0.4 | 2.4 | 3.1 | 0.7 | **81.5** | 1.7 | 0.4 | 3.2 |
| **38** | 3.6 | 11.9 | 2.2 | 0.0 | 5.3 | 1.1 | 1.3 | 2.1 | **66.6** | 0.3 | 5.6 |
| **39** | 2.2 | 2.2 | 3.8 | 1.9 | 5.5 | 10.3 | 3.6 | 7.9 | 4.3 | **56.4** | 1.9 |
| **stable** | 0.4 | 2.9 | 1.4 | 0.1 | 0.8 | 1.2 | 0.4 | 2.4 | 1.4 | 0.2 | **88.8** |

Normalising exposes structure the counts hide. Bus 32 is misread as bus 31 in **32.9%** of its records, a specific confusion rather than diffuse error, and bus 33 leaks **21.6%** to bus 34 and **18.1%** to bus 35. Bus 30 stays the weak class at 26.5%, spread across nearly every other label rather than concentrated anywhere.

### Per class

| Class | Support | Recall | Precision | **F1** | Top-3 |
|---|---|---|---|---|---|
| 30 | 1,687 | 0.265 | 0.459 | **0.336** | 0.794 |
| 31 | 9,140 | 0.826 | 0.751 | 0.787 | 0.967 |
| 32 | 2,505 | 0.505 | 0.590 | 0.544 | 0.879 |
| 33 | 991 | 0.453 | 0.463 | **0.458** | 0.790 |
| 34 | 13,394 | 0.879 | 0.869 | **0.874** | 0.983 |
| 35 | 3,020 | 0.579 | 0.535 | 0.556 | 0.894 |
| 36 | 2,646 | 0.652 | 0.666 | 0.659 | 0.896 |
| 37 | 6,225 | 0.815 | 0.845 | 0.830 | 0.952 |
| 38 | 2,422 | 0.666 | 0.714 | 0.689 | 0.890 |
| 39 | 417 | 0.564 | 0.639 | 0.599 | 0.758 |
| stable | 5,496 | 0.888 | 0.846 | 0.867 | 0.987 |

**Macro F1 0.654, weighted F1 0.762**, against a top-1 accuracy of 0.767. The gap between the two averages is the imbalance: accuracy and weighted F1 are carried by the three largest classes, while macro F1 counts bus 30 (0.336) and bus 33 (0.458) as equal partners and drops eleven points. Report macro when the question is whether every generator is identifiable, weighted when it is how often the answer is right.

Accuracy tracks support: the three largest classes (34, 31, 37) reach 0.82 to 0.88 recall while bus 30, with 1,687 records, reaches 0.265 and is confused with almost every other generator. Bus 33 is the weakest by precision (0.463), and its errors run overwhelmingly to buses 34 and 35. The confusion is not symmetric: bus 32 is mistaken for 31 in 824 records while 31 is mistaken for 32 in 410, so the ranking has a standing bias toward the larger class.

The top-1 versus top-3 gap (76.7% against 94.3%) says the recorded generator is nearly always among the leading candidates even when it is not first, which matters if the output is a shortlist for an operator rather than a single answer.

### Event level, one row per fault

Records are events on this dataset: a fault names a single critical generator, so `(state, location)` appears exactly once and the ratio is 1.00 rows per event. The event-level and per-record figures are therefore the same number, and set-membership accuracy, which asks whether the predicted machine is among those that lost synchronism, equals top-1 accuracy at **76.66%**.

This is stated only because interscada/fr differs: a fault there takes a group of machines out of step, the transform explodes the group into one row per machine, and its per-record and event-level figures come apart by a factor of 2.57.

## Most critical location

The operational question a screening pass asks is not "how accurate is each estimate" but "which contingency do I simulate first". The most critical location is the one with the lowest recorded CCT. This is a different quantity from `location_likelihood`, which ranks locations by retrieved weight mass, that is by which location the query most resembles rather than which is most severe; `cct_weighted_global` is built from the most likely location and cannot stand in for this.

| Measure | Value |
|---|---|
| Exact location match | 64.52% |
| Chosen location attains the minimum recorded CCT | **82.13%** |
| Regret within one 10 ms label step | 86.07% |
| Mean / median regret | 0.00663 s / **0.00000 s** |
| 95th percentile / maximum regret | 0.03600 s / 0.29500 s |
| MAE on the state's minimum CCT | 0.01107 s |
| States with a tied minimum | 162 of 761 |

Exact name matching understates performance. CCT labels sit on a 10 ms grid and 162 states have several locations tied at the minimum, so picking either of two equally severe locations is scored as a miss. By the measure that reflects the decision, whether the chosen location actually carries the worst clearing time, retrieval is right in **82.1%** of states and the median regret is exactly zero.

Estimating the minimum itself is easier than the average record: MAE 0.01107 s against 0.01358 s overall. The tail is real, though. One state, `SET_0221__Point_no.___10`, gives up 0.295 s of clearing time, meaning screening was pointed at a location far from the true worst.

### Top-1 and top-3 screening

Ordering a state's 63 locations by estimated severity and simulating the first k gives the classification-style view of the same question. Regret is the clearing time given up by stopping at k, in seconds.

| k | Worst location found | Regret mean | q50 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|
| 1 | 82.13% | 0.00663 | 0 | 0.022 | 0.036 | 0.10320 | 0.295 |
| **3** | **96.85%** | 0.00066 | 0 | **0** | **0** | 0.02840 | 0.041 |

Three simulations out of 63, **4.8% of a full contingency screen**, reach a genuinely worst-case contingency in 96.8% of states. The regret quantiles are the stronger argument: at k=3 regret is exactly zero through the 95th percentile and the worst case falls from 0.295 s to 0.041 s, so the third simulation buys tail protection rather than average accuracy.

Larger k is not reported because it stops discriminating. A state holds 63 locations, but a recorded generator's candidate set averages under six, so a top-5 rate would mostly measure that a candidate list exists.

Top-k answers "is the worst one found", not "are all the unsafe ones found", and the two diverge sharply:

| Threshold | recall@1 | recall@3 | States affected |
|---|---|---|---|
| CCT < 0.10 s | 0.490 | 0.719 | 401 |
| CCT < 0.15 s | 0.276 | 0.490 | 523 |
| CCT < 0.20 s | 0.138 | 0.356 | 617 |

At 0.20 s, 617 of 761 states hold at least one unsafe location and usually several, so three picks recover 35.6% of them. That is arithmetic rather than a ranking failure: if the requirement is to screen out every unsafe contingency, the risk-coverage and referral-load view applies, not top-k.

### Location likelihood is not a fault-location prediction

The service also ranks locations by `location_likelihood`, that is by retrieved weight mass within a generator group. Measured against the recorded location, it is weak:

| Measure | Value |
|---|---|
| Recorded location ranks first | 9.68% |
| Recorded location in top 3 | 23.84% |
| Mean locations competing | 48.36 |
| Mean winning score | 0.0958 |
| Most likely location equals the estimated most critical location | **5.91%** |

Against a uniform baseline of about 2% over 48 competing locations, ranking first 9.7% of the time is better than chance and still far too weak to present as a fault-location prediction. The last row is the one that matters operationally: the most likely location and the most critical location coincide in under 6% of states, because similarity says which locations have well-supported neighbours, not which fault is severe. This is why the fallback estimate built on the most likely location carries 5.6 times the error, and why the fault location has to be supplied as an input rather than inferred.

## Are the new states different from the rest?

Yes, decisively, in the input space.

Nearest-neighbour distance to the 743 existing states, over 169 standardised continuous features:

| | median | mean | max |
|---|---|---|---|
| new (18) | **10.38** | 9.07 | 12.18 |
| existing (743) | 1.43 | 1.49 | 6.82 |

Mann-Whitney U, one-sided, **p = 4.1e-11**, and **17 of 18 new states lie beyond the 95th percentile** of the existing states' own nearest-neighbour distances. Per feature, **57 of 137 continuous features (41.6%) shift significantly** after Holm correction, several at Cohen's d near 2.7, concentrated in loads 15 and 16 and the 345 kV bus voltage angles.

The batch is a **trajectory, not a cloud**. Distance to the nearest existing state grows monotonically with the point number: Point 3 at 0.671, Point 4 at 3.69, rising to 11.86 at Point 20. Point 2 sat at exactly 0.000 and has been removed as a duplicate. It begins exactly on an existing operating point and walks away from the archive. With the higher mean loading (3,679 MW against 3,293 MW) this reads as a deliberate stress sweep rather than a fresh random sample.

**One new state duplicates an existing one exactly.** `SET_0925__Point_no.____2` matches `SET_0025__Point_no.____2` on all 216 load-flow columns and all 63 CCT and critical-generator outcomes. Either the sweep re-simulated its starting point deliberately, or a state was exported twice; worth confirming with the data provider.

Retrieval support on the new states is unambiguously thinner, while the accuracy penalty is directionally consistent but underpowered at n = 18:

| | new | existing | p |
|---|---|---|---|
| Per-state MAE | 0.01604 | 0.01355 | 0.249 |
| `distance_min` | 3.32 | 1.98 | <0.0001 |
| `n_neighbors` | 20.4 | 45.9 | <0.0001 |
| `n_eff` | 3.91 | 10.87 | <0.0001 |
| Coverage | 98.77% | 99.37% | - |

Under half the neighbours at nearly double the distance, for 18% higher error that does not reach significance. Support degrades before conditional accuracy does, which is the same ordering the ELES analysis reports.

### Do they form their own cluster?

No. Clustering the 761 states by k-means on the same standardised features, the new batch always lands inside existing clusters and never gets one of its own, at any k from 2 to 6.

| Cluster (k=6) | sweep 0 | sweep 1 | sweep 2 | `SET_0925` |
|---|---|---|---|---|
| 0 | 93 | 105 | 115 | 3 |
| **2** | 38 | 33 | 24 | **15** |
| 1, 3, 4, 5 | 115 | 110 | 110 | 0 |

They do concentrate, 15 of 18 in one cluster, but reach at most **13.6%** of any cluster's membership. Within the clusters they join they sit further out than typical members, at a mean distance to centroid of 9.86 against 7.55 in cluster 2 and 7.33 against 4.67 in cluster 0, both inside the existing 95th percentile.

The archive's actual structure is the three sweeps, and they are interleaved rather than separated: `SET_00NN`, `SET_01NN` and `SET_02NN` hold 246, 248 and 249 states and distribute across all six clusters in near-equal thirds. They are three repetitions of one operating-point grid, not three regions of it.

### What distinguishes them physically

| Quantity | new (18) | existing (743) | new range | existing range | Cohen d | p |
|---|---|---|---|---|---|---|
| **generation minus load (losses)** | **81.9 MW** | 304.2 MW | 70.7-89.8 | 69.4-606.8 | **-1.70** | 2.5e-07 |
| total load | 3,728 MW | 3,305 MW | 2,883-4,500 | 2,331-5,583 | 0.75 | 1.3e-03 |
| total load, reactive | 787 Mvar | 710 Mvar | 614-944 | 497-1,189 | 0.65 | 4.4e-03 |
| total generation | 3,810 MW | 3,609 MW | 2,954-4,590 | 2,851-5,707 | 0.36 | 0.11 |
| mean / minimum voltage | 1.05 / 0.98 pu | 1.04 / 0.98 pu | | | 0.29 / 0.22 | ns |
| angle spread | 25.5 deg | 25.3 deg | 15.7-32.0 | 13.9-55.2 | 0.04 | ns |
| generators online | 10 | 10 | | | | |

The sharpest separator is **network losses**: the new batch occupies a narrow 71 to 90 MW band where the existing archive spans 69 to 607 MW. Everything else overlaps heavily. Voltages, angle spread and unit commitment are indistinguishable, and all ten machines are online in every state of both groups.

So the new states are not a new operating regime. They are more heavily loaded points inside the same envelope, drawn from a much narrower loss band, sitting at the periphery of existing clusters rather than beyond them.

That also explains the retrieval result without appealing to novelty. The existing states arrive in tight sweeps, with a mean within-family nearest-neighbour distance of **2.05** and as little as 1.03 for `SET_0007`, so a typical query has near-identical siblings to retrieve. The new batch has no sweep of its own, and its nearest existing neighbours fall in just two families: `SET_0001` for 10 of the 18 and `SET_0023` for 7, with one at `SET_0025`. Thin support here is a sampling artefact of how the archive was built, not evidence that the states are hard to estimate.

## Duplicate states

**19 duplicate pairs, 38 states, were removed at preparation**, leaving 761 of the drop's 780. Each pair is byte-identical across all 216 load-flow columns, verified on the raw CSV before any parsing, and their outcomes match too: the same CCT and critical generator at every one of the 63 fault locations. The pairs are `Point_no. 2` through `7` of `SET_0013` repeating `SET_0021`, `SET_0113` repeating `SET_0121`, `SET_0213` repeating `SET_0221`, and `SET_0925__Point_no.____2` repeating `SET_0025__Point_no.____2` from the earlier drop. One pair's raw rows differ only in whitespace inside the generator label, `37 [BUS37   13.8]` against `37 [BUS37 13.8]`, which the regex parse discards.

Joining the topology columns changes nothing: all 780 raw states share one switching configuration, so the whole-row and grid-values-only duplicate counts are both 19.

Removing them at preparation rather than at evaluation is deliberate, and matches interscada/fr. A duplicate defeats leave-one-state-out, which excludes the query by identity and so leaves its twin, holding the same answer, in the reference library. It also distorts deployment, because each copy votes separately in the kernel-weighted mean.

An earlier version of this report was computed with the duplicates in place. Removing them moved the headline figures by roughly the amount their share of the archive suggests, 2.4% of states: MAE 0.01281 to 0.01358 s, top-1 identification 78.22% to 76.66%, macro F1 0.690 to 0.654, and the worst-location hit rate 83.59% to 82.13%. Every figure moved in the same direction, which is what a removed leak looks like.

## Limitations

- **Single topology group.** The exact-topology filter is inert on this archive, so nothing here tests abstention behaviour, and coverage figures are not comparable with ELES.
- **Leave-one-state-out excludes only the query state.** Sweep points within a `SET` family are plausibly adjacent operating points and remain retrievable. There are no timestamps, so no temporal exclusion control is possible; the 0.01358 s MAE should be read as an upper bound on accuracy under near-duplicate support. Exact duplicates are removed at preparation, so what remains is genuinely similar rather than identical states.
- **Preprocessing is transductive.** The scaler is fitted and the index populated over all 761 states including the query. Exclusion applies to retrieval, not to preprocessing.
- **Class imbalance.** Generator classes span 417 to 13,394 records, so the aggregate top-1 rate is dominated by the largest classes; read the per-class table alongside it.
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

Artifacts, all regenerated by the commands above. Unless noted, they live in `datasets/interscada/pl/reports/`:

| File | Content |
|---|---|
| `risk_coverage_interscada-pl_lines_only.csv` | risk-coverage curve per diagnostic |
| `critical_location_interscada-pl_lines_only.csv` | most-critical-location selection |
| `error_quantiles_interscada-pl_lines_only.csv` | absolute-error quantiles per estimate and support split |
| `topk_location_interscada-pl_lines_only.csv` | top-1 and top-3 severity ranking with regret quantiles |
| `topk_screening_recall_interscada-pl_lines_only.csv` | share of unsafe locations captured at top-1 and top-3 |
| `crit_gen_confusion_interscada-pl.csv` | confusion matrix, counts |
| `crit_gen_confusion_relative_interscada-pl.csv` | confusion matrix, row-normalised |
| `crit_gen_per_class_interscada-pl.csv` | per-class recall, precision, top-3 |
| `crit_gen_summary_interscada-pl.csv` | headline identification rates |
| `tmp/report-interscada-pl-lines_only.joblib` | per-record CCT predictions and diagnostics |
| `tmp/crit-gen-confusion-interscada-pl.joblib` | per-record generator rankings |
