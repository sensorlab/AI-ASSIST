# Interscada FR Dataset

Power system simulation data for the French transmission grid. The **September 2026 drop** carries 1,571 raw snapshots and transient stability results for 9 fault experiments per scenario, of which **1,304 states and 11,653 records** survive preparation (see Preparation Filters below). The earlier **55-snapshot May 2024 pilot** is kept under `raw/v1/`; it is not a subset of this drop, which uses a different bus set, a different fault list and a different simulation ceiling.

## Source Files

| File | Role | Shape | Sep |
|---|---|---|---|
| `raw/cct_global_results_clean.csv` | TSA targets (CCT + critical generators) | 1,568 × 19 | `,` |
| `raw/line_global_results_clean.csv` | Topology features (line statuses) | 1,571 × 432 | `;` |
| `raw/static_global_results_clean.csv` | LF features (bus voltages, P, Q) | 1,571 × 1934 | `;` |
| `raw/hvdc_global_results_clean.csv` | HVDC link flows and statuses | 1,571 × 15 | `;` |
| `raw/v1/*.csv` | The superseded 55-snapshot May 2024 pilot | 55 rows | |

**`hvdc_global_results_clean.csv` is not consumed.** `transform.py` reads only the static and line files. Its two `status_*` flags are constant across every scenario and would add nothing to the topology key; its per-link P/Q columns are unused continuous features. Its rows also partition the scenarios exactly as the static file does, so it carries no discriminating information beyond what is already read.

## Transform Script

`transform.py` produces ML-ready pickles in `interim/`.

## Design Decisions

### LF dataset

`static_global_results_clean.csv` and `line_global_results_clean.csv` are concatenated into a single LF dataframe. The static file contains per-bus groups of `V_{bus}`, `angle_{bus}`, `Pgen_{bus}`, `Qgen_{bus}`, `Pload_{bus}`, `Qload_{bus}` (voltages in kV, not per-unit). The line file contains binary line statuses (0/1).

**Sparse bus data**: The number of active buses varies per scenario (`n_buses` column, range 302-319 in the September 2026 drop, 303-315 in the pilot) because scenarios represent real grid snapshots with different topologies. Bus columns that don't exist in a given scenario are NaN (confirmed by partner: "empty columns indicate that the bus doesn't exist in the current scenario"). No null-completeness assertion is made on the LF columns.

### CCT file (`cct_global_results_clean.csv`)

- **Wide-to-long reshape**: Columns alternate `(CCT, Critical gen, CCT, Critical gen, ...)`. The fault name is embedded in the CCT column header (`Fault – NAME` for the first fault, bare `NAME` for subsequent ones). `pd.wide_to_long` cannot be used (no numeric suffix) — the reshape is done manually by iterating over column pairs.
- **CCT offset**: The file stores `t_clearance`, not CCT directly. The fault is inserted at t=1s, so **CCT = file_value − 1.0 s** (e.g. a file entry of 1.5s → CCT = 0.5s). The transform applies this subtraction. The ceiling moved between drops: the pilot stopped at 1.8s (CCT 800ms), this drop stops at 2.0s (**CCT 1.0s**).
- **`CALC PB`**: Marks a simulation failure for a reason other than loss of synchronism (not a stable fault). These rows are **dropped** during transformation (88 cells in this drop; 5 of 660 in the pilot).
- **Stable faults**: When CCT reaches the ceiling, no critical generator is identified - `Crit_gen` is NaN. These rows are kept: **4,425 of 14,024 records (31.6%)** before deduplication, against 40 of 655 in the pilot. They are right-censored rather than observed - the true CCT is at least the ceiling, not equal to it - and at nearly a third of the archive that censoring is large enough to affect any aggregate computed from them.
- **A fault takes a *group* of machines out of synchronism, and the group is exploded into one row per machine.** Unlike every other dataset in this project, the raw `Critical gen` cell lists several generators, comma-separated. This drop holds exactly three distinct values: `FLAMA7FLAMAT1`, `PALUE7PALUET1, PALUE7PALUET3, PALUE7PALUET4`, and `PENLY7PENLYT1, PENLY7PENLYT2, PALUE7PALUET1, PALUE7PALUET3, PALUE7PALUET4` - and the two multi-machine values are nested, the 3-set being a subset of the 5-set, so as whole strings they describe how far the out-of-step group extends rather than three disjoint outcomes.

  `transform.py` splits them, so `Crit_gen` means what it means everywhere else in the project: **one generator per row**, six distinct machines, matching the pilot's six.

  | | |
  |---|---|
  | `PALUE7PALUET1` / `T3` / `T4` | 6,697 rows each |
  | `PENLY7PENLYT1` / `T2` | 2,424 each |
  | `FLAMA7FLAMAT1` | 1,301 |
  | stable, no critical generator | 3,655 |

- **The original group is kept in `crit_gen_group`, and record counts are no longer event counts.** The explosion is lossy: from per-machine rows alone there is no way to recover which machines lost synchronism together in one event, so the full string is retained alongside. Its consequence must be carried into any analysis: **29,895 rows describe 11,653 distinct `(state, location)` events, 2.57 rows per event**, and `CCT` is replicated across a group's rows because it is a property of the fault, not of the individual machine. Any metric averaged over rows therefore weights a 5-machine event five times as heavily as a single-machine one. Group by `(state, Location)` first where that matters.

### Preparation Filters

`transform.py` removes three classes of row, each logged with its count:

1. **`CALC PB` cells** (88 in this drop) — a simulation failure for a reason other than loss of synchronism. Coerced to NaN by the numeric parse and dropped.
2. **Duplicate scenarios** (264 of 1,571) — the drop stores 240 pairs plus one group of 25 as byte-identical copies of another scenario, and their outcomes are identical too: all 453 duplicate pairs agree on CCT at every shared fault location. They are detected by exact row equality, not by a distance threshold, which needs no tolerance and does not shift with float precision (the same data yields 276 or 446 "duplicates" at radius 1e-6 depending only on float32 versus float64).

   Removing them at preparation rather than at evaluation is deliberate. A duplicate defeats leave-one-state-out, since excluding the query by id leaves a verbatim copy of its answer in the reference library — but it also distorts *deployment*, because each copy votes separately in the kernel-weighted mean, so an operating point stored 25 times would carry 25 times the weight of one stored once. That is a property of the archive, not of how it is scored.

   The group of 25 identical scenarios (`scenario1048`, `1080`, `1121`, `1163`, `1234`, `1297`, `1376`, `1389`, …) is worth raising with the data provider: 25 identical exports is unlikely to be intentional sampling.
3. **States with no TSA record** (3: `scenario1569`, `1570`, `1571`) — the static, line and hvdc files cover 1,571 scenarios while the CCT file covers 1,568. A state with no outcome cannot support an estimate, and `EstimationService` treats a retrieved state with no TSA record as an error rather than a silent gap, so these are dropped rather than indexed and left to fail at query time.

After all three, `tsa` is restricted to the retained states so record counts and coverage denominators refer to one set of states: **1,304 states, 11,653 records**.

### Topology columns

All 431 columns from `line_global_results_clean.csv` are topology features (binary line statuses). Unlike the PL dataset, no renaming is needed - column names already encode the line identity (e.g. `ALBERL71BATHI`).

Only **29 of the 431** flags vary across the drop; the rest are constant and cannot split anything, so the archive's **10 topology groups** (sizes 245, 145, 144, 142, 133, 125, 121, 121, 121, 94 after deduplication) come entirely from those 29. Every varying flag is a French circuit: the four columns added in this drop are all constant, including `BIPOLL71MANDA`, which is out of service in every scenario.

**Selecting them in the scaler**: `make_scaler_interscada_fr()` takes this exact list rather than matching line names by shape. The previous `^[A-Z][^_]*$` pattern silently dropped `.ANGL 7 .CHER 1` and `.ANGL 7 .CHIL 1` when this drop introduced names with a leading dot, and the build then failed later with an unrelated-looking error. These dotted names are the England-side interconnector nodes: `.CHER` (Cheriton, ElecLink), `.CHIL` (Chilling, IFA2) and `.ANGL` (Angleterre, a GB equivalent node), with `BIPOL` the IFA bipole to Les Mandarins.
