# interscada/fr: leave-one-state-out evaluation (July 2026 drop)

## Scope

Leave-one-state-out evaluation of retrieval-based CCT estimation on the July 2026 French drop, run through the production service path (`build_estimation_service()`, embedded Qdrant, `DATASET_NAME=interscada/fr`) with `scripts/evaluation/eles_benchmark.py`. Every query excludes its own state, and the script asserts that exclusion held on every query; the assertion never fired across 1,304 queries.

This supersedes an earlier pass on the same raw files, before deduplication. That run reported MAE 0.00948 s and is not comparable: a quarter of its queries retrieved a byte-identical copy of their own state (see "What deduplication changed").

## Data

1,571 raw scenarios reduce to **1,304 states, 11,653 fault events, 29,895 records** through three preparation filters, each logged by `transform.py` and documented in the dataset README: 88 `CALC PB` cells, 264 duplicate scenarios, and 3 states with no CCT row.

Two structural facts govern how everything below reads.

**A fault takes a group of machines out of synchronism.** The raw `Critical gen` cell lists several generators, and the transform explodes them into one row per machine, so `Crit_gen` means a single generator as it does in every other dataset. The consequence is that **records are not events**: 29,895 rows describe 11,653 `(state, location)` events, 2.57 rows per event, with `CCT` replicated across a group's rows. Row-level averages therefore weight a 5-machine event five times a single-machine one. The location analyses below collapse to event level first; the error table does not.

**Nearly a third of events are right-censored.** 3,655 events (31.4%) sit at the 1.0 s simulation ceiling with no critical generator, meaning the fault did not destabilise the system within the tested window. Their true CCT is at least 1.0 s, not equal to it.

The 431 line-status flags partition the archive into **10 topology groups** of 242, 121 (eight of them) and 94 states. Only 29 flags vary; all are French circuits, and every interconnector flag added in this drop is constant.

## Coverage and accuracy

Coverage is effectively complete: **29,892 of 29,895 records (99.99%)** receive a location-specific estimate, and every record receives some estimate. The topology filter never abstains here because no group is a singleton, unlike the 55-state pilot where it answered 3.66% of records.

### Error distribution

| Estimate | n | MAE | RMSE | q25 | q50 | q75 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|---|---|---|
| recorded location | 29,892 | **0.01407** | 0.02652 | 0.00167 | 0.00633 | 0.01614 | 0.03625 | 0.05555 | 0.11442 | **0.26306** |
| most likely location | 29,895 | 0.12297 | 0.18804 | 0.00414 | 0.04636 | 0.23985 | 0.35141 | 0.40903 | 0.45985 | 0.60167 |
| recorded, n_eff below median (8.7) | 14,944 | 0.01384 | 0.02636 | 0.00169 | 0.00607 | 0.01477 | 0.03512 | 0.05667 | 0.11501 | 0.24025 |
| recorded, n_eff at or above median | 14,948 | 0.01430 | 0.02667 | 0.00165 | 0.00651 | 0.01704 | 0.03689 | 0.05496 | 0.11245 | 0.26306 |

Against a mean recorded CCT of 0.658 s that is a 2.1% relative MAE, and the distribution is skewed: the median error is 6.3 ms, three quarters of records fall inside 16 ms, and the maximum is 0.263 s, nearly 19 times the mean.

**The headline MAE is deflated by the censored records.** Retrieval predicts every one of the 3,655 stable records exactly, MAE 0.00000, because within a topology group all neighbours at those locations also sit at the ceiling. Excluding them, MAE on genuinely observed records is **0.01603 s**. Any comparison against another dataset should use that figure, since no other archive in this project is a third censored.

Effective sample size does not separate good estimates from bad here: the two halves differ by 3% and the thin-support half is marginally *better*, so `n_eff` carries no usable signal on this archive.

## Support diagnostics

MAE over all records, and over the best-ranked half by each diagnostic.

| Diagnostic | MAE, full | MAE, best 50% | Change |
|---|---|---|---|
| `cct_weighted_std` | 0.014049 | 0.004463 | **-68.2%** |
| `location_weight_mass` | 0.014073 | 0.009202 | -34.6% |
| `n_neighbors` | 0.014073 | 0.010155 | -27.8% |
| `distance_norm` | 0.014073 | 0.012933 | -8.1% |
| `neighborhood_compactness` | 0.014049 | 0.014122 | +0.5% |
| `n_eff` | 0.014073 | 0.014302 | +1.6% |
| `distance_min` | 0.014073 | 0.014491 | +3.0% |
| `distance_spread` | 0.014073 | 0.015520 | +10.3% |
| `distance_mean` | 0.014073 | 0.016204 | +15.1% |
| `distance_median` | 0.014073 | 0.016432 | +16.8% |

Disagreement among retrieved outcomes is again the only strong signal, as on interscada/pl and BUS39. Five diagnostics rank estimates worse than not sorting at all, so none of them should be shown as a confidence measure on this archive.

## Critical-generator identification

At each record's recorded fault location, candidate generators are ranked by `crit_gen_likelihood` and the top-ranked one is compared with the recorded outcome. On this dataset the per-record form of that question is close to meaningless, for two structural reasons, so the event-level result below is the one to read.

**Per-record accuracy is capped at 39.0% by construction.** One prediction is made per `(state, location)`, but the explode gives that event up to five rows, so at most one row can be scored correct. The ceiling is the event count over the row count, 11,653 / 29,895 = 38.98%. Measured per-record top-1 is **38.31%**, which is 98.3% of the achievable maximum, and per-record top-3 is 83.10%.

**Machines within a group are indistinguishable by construction.** The exploded rows of one event carry identical retrieval support, so the ranking among `PALUE7PALUET1`, `T3` and `T4` is a tie broken by insertion order. The model therefore only ever predicts `PALUE7PALUET1`, `FLAMA7FLAMAT1` or `stable`, and `T3`, `T4`, `PENLY7PENLYT1` and `PENLY7PENLYT2` are never named. Per-class F1 inherits that: macro 0.337 and weighted 0.253, with four classes at exactly 0. Those numbers measure the tie-break, not the estimator, and should not be compared with interscada/pl's macro 0.690.

### Confusion matrix (rows recorded, columns predicted)

Reported for symmetry with the interscada/pl report, but read the event-level table below instead: this one is shaped by the tie-break, not by the estimator.

|  | FLAMA7FLAMAT1 | PALUE7PALUET1 | stable |
|---|---|---|---|
| **FLAMA7FLAMAT1** | **1,301** | 0 | 0 |
| **PALUE7PALUET1** | 0 | **6,589** | 108 |
| **PALUE7PALUET3** | 0 | 6,589 | 108 |
| **PALUE7PALUET4** | 0 | 6,589 | 108 |
| **PENLY7PENLYT1** | 0 | 2,412 | 12 |
| **PENLY7PENLYT2** | 0 | 2,412 | 12 |
| **stable** | 0 | 93 | **3,562** |

### Confusion matrix, row-normalised (% of each recorded class)

|  | FLAMA7FLAMAT1 | PALUE7PALUET1 | stable |
|---|---|---|---|
| **FLAMA7FLAMAT1** | **100.0** | 0.0 | 0.0 |
| **PALUE7PALUET1** | 0.0 | **98.4** | 1.6 |
| **PALUE7PALUET3** | 0.0 | 98.4 | 1.6 |
| **PALUE7PALUET4** | 0.0 | 98.4 | 1.6 |
| **PENLY7PENLYT1** | 0.0 | 99.5 | 0.5 |
| **PENLY7PENLYT2** | 0.0 | 99.5 | 0.5 |
| **stable** | 0.0 | 2.5 | **97.5** |

Three of the seven rows are identical, which is the tie-break made visible: the exploded rows of one event carry the same support, so whichever machine sorts first absorbs the whole group.

### Per class

| Class | Support | Recall | Precision | F1 |
|---|---|---|---|---|
| FLAMA7FLAMAT1 | 1,301 | 1.000 | 1.000 | **1.000** |
| PALUE7PALUET1 | 6,697 | 0.984 | 0.267 | 0.420 |
| PALUE7PALUET3 | 6,697 | 0.000 | undefined | **0.000** |
| PALUE7PALUET4 | 6,697 | 0.000 | undefined | **0.000** |
| PENLY7PENLYT1 | 2,424 | 0.000 | undefined | **0.000** |
| PENLY7PENLYT2 | 2,424 | 0.000 | undefined | **0.000** |
| stable | 3,655 | 0.975 | 0.911 | 0.942 |

Macro F1 0.337, weighted F1 0.253. Four classes score exactly 0 because they are never predicted, and `PALUE7PALUET1` pays for absorbing them with a precision of 0.267. Neither number describes the estimator; both describe the explode. The interscada/pl report's macro 0.690 is not comparable.

### Event level, one row per fault

| Recorded out-of-step group | Events | → FLAMA T1 | → PALUE T1 | → stable |
|---|---|---|---|---|
| `FLAMA7FLAMAT1` | 1,301 | **1,301** | 0 | 0 |
| `PALUE7PALUET1, T3, T4` | 4,273 | 0 | **4,177** | 96 |
| `PENLY7PENLYT1, T2, PALUE7PALUET1, T3, T4` | 2,424 | 0 | **2,412** | 12 |
| stable | 3,655 | 0 | 93 | **3,562** |

Row-normalised: **100.0%**, **97.8%**, **99.5%** and **97.5%**.

**Set-membership accuracy is 98.28%** over the 11,653 events - the share where the predicted machine is one of those that actually lost synchronism, or where a stable event was called stable. By group: FLAMA 100.0%, the 3-machine group 97.8%, the 5-machine group 99.5%, stable 97.5%.

The recorded generator was absent from the candidate set in 0.01% of records, with 4.03 candidates per record on average.

So retrieval names the correct out-of-step *group* in 98.3% of events, and the errors are almost entirely confusions with the stable class rather than between generators. What it cannot do on this data is tell one machine of a group from another, because nothing in the archive distinguishes them.

## Most critical location

The most critical location is the one with the lowest recorded CCT, the contingency a screening pass should simulate first. Ranking a state's locations by estimated severity:

### Top-1 and top-3 screening

| k | Worst location found | Regret mean | q50 | q90 | q95 | q99 | q100 |
|---|---|---|---|---|---|---|---|
| 1 | **99.92%** | 0.00002 | 0 | 0 | 0 | 0 | 0.02344 |
| 3 | **100.00%** | 0.00000 | 0 | 0 | 0 | 0 | 0.00000 |

Exact location match is 99.77%, and the two figures differ only because two states have several locations tied at the minimum. MAE on the state's minimum CCT is 0.00631 s, better than the overall 0.01407 s. At k=3 the worst case over all 1,304 states is zero regret.

This is the strongest result in the report and it is not an artefact of duplication, which was removed at preparation.

Screening recall is nearly vacuous on this archive: only **27 of 1,304 states** hold any location below 0.20 s, so recall is 1.0 at k=1 over those 27. With a mean CCT of 0.658 s and a 1.0 s ceiling, this drop contains few severe contingencies at all.

### Location likelihood is not a fault-location prediction

`location_likelihood` ranks locations by retrieved weight mass within a generator group, that is by which location the query resembles, not which is severe.

| Measure | Value |
|---|---|
| Recorded location ranks first | 30.29% |
| Recorded location in top 3 | 70.19% |
| Mean locations competing | 5.94 |
| Mean winning score | 0.3024 |
| **Most likely location equals the estimated most critical location** | **0.00%** |

Against a uniform baseline near 17% over 5.94 competing locations, ranking first 30% of the time beats chance but is far too weak to present as a prediction. The last row is categorical: across all 1,304 states the most likely location is *never* the estimated most critical one. That is why the fallback estimate built on the most likely location carries nearly nine times the error, and why fault location must be supplied as an input.

## Duplicate states

**264 duplicate scenarios, 240 pairs plus one group of 25, were removed at preparation** (see the dataset README). Every figure in this report is computed without them. The first pass on this drop, before they were removed, produced numbers that looked much better and were not trustworthy.

| | before dedup | after dedup and explode |
|---|---|---|
| States / scored rows | 1,568 / 14,024 | 1,304 / 29,892 |
| MAE | 0.00948 s | 0.01407 s |
| Records predicted exactly right | 11.82% | 4.98% |
| Records with a neighbour closer than 1e-3 | 32.3% | 0.17% |
| `distance_min` q25 | 0.0000 | 1.3119 |
| `distance_min` as a diagnostic | -14.2% | +3.0% |
| `n_eff` as a diagnostic | +17.4% | +1.6% |

A quarter of queries used to retrieve a byte-identical copy of their own state, which leave-one-state-out does not prevent: it excludes the query by identity, not the copies of it stored under other names. Two diagnostics were measuring that duplication rather than support quality, and both collapsed once it was removed - minimum distance stopped helping entirely, and the inverted `n_eff` relationship (thin support 42% *better*) flattened to a 3% difference.

Error rose about 48%, but that figure mixes two changes: removing the twins, and the explode now weighting multi-machine events more heavily. It should not be read as the cost of deduplication alone.

## Limitations

- **Right censoring, 31.4% of events.** Stable records are bounds, not measurements, and retrieval scores them at zero error by construction. Use the 0.01603 s observed-only MAE for cross-dataset comparison.
- **Records are not events.** Row-level metrics weight a 5-machine event five times a single-machine one. Location analyses collapse to event level; the error table does not.
- **Few severe contingencies.** 27 of 1,304 states hold a location below 0.20 s, so screening behaviour is barely exercised.
- **Preprocessing is transductive.** The scaler is fitted and the index populated over all 1,304 states including the query; exclusion applies to retrieval, not preprocessing.
- **Leave-one-state-out excludes only the query state.** Exact duplicates are now removed at preparation, but the control does not withhold merely-similar states. `ELES_BENCHMARK_DUPLICATE_RADIUS` exists for that and was not needed here: after deduplication the nearest-neighbour distribution has a clean gap, with nothing between 1e-6 and 1.25.
- **The variant tag `lines_only` in artifact names is an artefact** of the `TOPOLOGY_VARIANT` default. This dataset has one topology column set and it was used correctly.

## Reproduction

```bash
uv run ai-assist-prepare interscada/fr

DATASET_NAME=interscada/fr QDRANT_URL=":memory:" \
    nice -n 19 uv run python scripts/evaluation/eles_benchmark.py
```

Retrieval took 1,771 s for 1,304 states; the service builds in 3.6 s.

| File | Content |
|---|---|
| `risk_coverage_interscada-fr_lines_only.csv` | risk-coverage curve per diagnostic |
| `error_quantiles_interscada-fr_lines_only.csv` | absolute-error quantiles per estimate and support split |
| `critical_location_interscada-fr_lines_only.csv` | most-critical-location selection |
| `topk_location_interscada-fr_lines_only.csv` | top-1 and top-3 severity ranking with regret quantiles |
| `topk_screening_recall_interscada-fr_lines_only.csv` | share of unsafe locations captured at top-1 and top-3 |
| `crit_gen_event_confusion_interscada-fr.csv` | event-level confusion matrix, counts |
| `crit_gen_event_confusion_relative_interscada-fr.csv` | event-level confusion matrix, row-normalised |
| `crit_gen_event_summary_interscada-fr.csv` | set-membership accuracy and the per-record ceiling |
| `crit_gen_confusion_interscada-fr.csv` | per-record confusion matrix, counts (see caveats above) |
| `crit_gen_confusion_relative_interscada-fr.csv` | per-record confusion matrix, row-normalised |
| `crit_gen_per_class_interscada-fr.csv` | per-class recall, precision, F1 |
| `tmp/report-interscada-fr-lines_only.joblib` | per-record predictions and diagnostics |
| `tmp/crit-gen-confusion-interscada-fr.joblib` | per-record generator rankings |
