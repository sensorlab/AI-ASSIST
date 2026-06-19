# Interscada FR Dataset

Power system simulation data for the French transmission grid. Contains 55 snapshots of the real French grid during **May 2024** and transient stability results for 12 fault experiments per scenario. This is a pilot dataset — a larger set is expected with more scenarios covering additional grid topologies.

## Source Files

| File | Role | Shape | Sep |
|---|---|---|---|
| `raw/cct_global_results_clean.csv` | TSA targets (CCT + critical generator) | 55 × 24 | `,` |
| `raw/line_global_results_clean.csv` | Topology features (line statuses) | 55 × 427 | `;` |
| `raw/static_global_results_clean.csv` | LF features (bus voltages, P, Q) | 55 × 2083 | `;` |

## Transform Script

`scripts/interscada/transform_fr.py` — produces ML-ready pickles in `data/interscada/fr/interim/`.

## Design Decisions

### LF dataset

`static_global_results_clean.csv` and `line_global_results_clean.csv` are concatenated into a single LF dataframe. The static file contains per-bus groups of `V_{bus}`, `angle_{bus}`, `Pgen_{bus}`, `Qgen_{bus}`, `Pload_{bus}`, `Qload_{bus}` (voltages in kV, not per-unit). The line file contains binary line statuses (0/1).

**Sparse bus data**: The number of active buses varies per scenario (`n_buses` column, range 303–315) because scenarios represent real grid snapshots with different topologies. Bus columns that don't exist in a given scenario are NaN (confirmed by partner: "empty columns indicate that the bus doesn't exist in the current scenario"). No null-completeness assertion is made on the LF columns.

### CCT file (`cct_global_results_clean.csv`)

- **Wide-to-long reshape**: Columns alternate `(CCT, Critical gen, CCT, Critical gen, ...)`. The fault name is embedded in the CCT column header (`Fault – NAME` for the first fault, bare `NAME` for subsequent ones). `pd.wide_to_long` cannot be used (no numeric suffix) — the reshape is done manually by iterating over column pairs.
- **CCT offset**: The file stores `t_clearance`, not CCT directly. The fault is inserted at t=1s, so **CCT = file_value − 1.0 s** (e.g. a file entry of 1.5s → CCT = 0.5s). The transform applies this subtraction. Maximum considered CCT is 800ms (file value 1.8s).
- **`CALC PB`**: Marks a simulation failure for a reason other than loss of synchronism (not a stable fault). These rows are **dropped** during transformation (5 out of 660 rows, 0.8% loss).
- **Stable faults**: When CCT reaches the 800ms ceiling, no critical generator is identified — `Crit_gen` is NaN. These rows are kept (40 occurrences).
- **`Crit_gen`**: Generator names are already meaningful strings (e.g. `PENLY7PENLYT1`) — no parsing needed, used directly as `Crit_gen` for cross-dataset compatibility.

### Topology columns

All 427 columns from `line_global_results_clean.csv` are topology features (binary line statuses). Unlike the PL dataset, no renaming is needed — column names already encode the line identity (e.g. `ALBERL71BATHI`).
