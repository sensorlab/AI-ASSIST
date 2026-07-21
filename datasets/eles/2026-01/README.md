# ELES Dataset (2026-01)

Real-world power system data from ELES (Slovenian transmission system operator). Contains 4,402 hourly operating states sampled June 2024 – January 2025, with transient stability results for up to 172 line contingency scenarios per state (~324K total TSA records).

## Source Files

| File | Role | Notes |
|---|---|---|
| `raw/SLOJun2024_Jan2025_only_lne_1h.zip` | Archive of all raw CSVs | 89 batches × 6 file types (534 CSVs) plus 2 summary files (`info_main.csv`, `info_main_final.csv`) = 536 CSVs total |
| `raw/ignorelist.txt` | Rows to skip during loading | Corrupted or invalid measurements |
| `raw/powerfactory_dictionary.xlsx` | PowerFactory element dictionary | Used to identify topology columns via `Lines` and `Loads` sheets |

### Raw archive structure

Each batch `N` (0–88) contains:

| File | Role |
|---|---|
| `LF_main_N.csv` | Load-flow features for one time batch |
| `TSA_main_N.csv` | Transient stability results (wide format) |
| `FSA_main_N.csv` | Frequency stability indices |
| `SSSA_main_N.csv` | Small-signal stability indices — not used |
| `PQdiff_main_N.csv` | P/Q differences — not used |
| `Date_main_N.csv` | Timestamps for each row — not used |

## Transform Script

`transform.py` processes all 89 batches in parallel, produces ML-ready pickles in `interim/`.

## LF Feature Groups

| Group | Columns | Range | Notes |
|---|---|---|---|
| Bus voltage | `U_Bus*_[pu]` × 2226 | 0–3.117 pu | 0 if bus disconnected |
| Phase angle | `phi_Bus*_[deg]` × 2226 | −180–180° | Relative to reference bus |
| Line active power | `P1_Lne*_[MW]`, `P2_Lne*_[MW]` × 1683 | −7423–14050 MW | From/to end of each line |
| Line reactive power | `Q1_Lne*_[MW]`, `Q2_Lne*_[MW]` × 1683 | −8654–3376 MVAr | Header says `[MW]` but unit is MVAr |
| Short-circuit power | `Sk_*_[MVA]` × 1408 | — | Per-element short-circuit capacity |
| Topology | `oserv_Gen*`, `oserv_Lne*` × 1072 | True/False | See below |

## Design Decisions

### LF files (`LF_main_N.csv`)

- **Separator / decimal**: `;` separator, `,` decimal (European locale).
- **State ID**: Each row's state key is `"{batch_index}_{row_index}"` (e.g. `"3_47"`), combining the file batch number and the within-file row index. This ensures uniqueness across the full dataset.
- **`oserv_*` columns**: Arrive as string-encoded floats (`"1.0"`, `"0.0"`) with NaN for missing entries. Cast chain: `astype("string") → fillna("1.0") → astype(float) → astype(int) → astype(bool)`. Encoding: `True` (~= out of service), `False` (~= in service) — per the `make_scaler_eles` comment in `service.py`. NaN fills as `"1.0"` → `True` (out of service).
- **Topology column selection**: Not all `oserv_*` columns are used as topology features. The `powerfactory_dictionary.xlsx` (sheets: `Lines`, `Loads`) provides a list of elements to include. The `Generators` sheet is excluded — too many topology changes across operating states make generator status unreliable as a feature.

### TSA files (`TSA_main_N.csv`)

- **Wide-to-long**: Same structure as bus39 — `CCT_N / Terminal_N / Crit_gen_N / Type_N / Location_N` groups reshaped with `pd.wide_to_long`.
- **CSV quirks**: Two pre-processing fixes applied before parsing:
  1. Missing separator before decimal values: `re.sub(r"(?<!;)(\d{1},\d+)(?=;)", r";\1", text)`
  2. Trailing semicolons stripped per line: `re.sub(r";+$", "", text, flags=re.MULTILINE)`
- **CCT units**: Seconds, range 0.11–1.95 s.
- **`Type`**: Always `1` (line contingency) — consistent with `only_lne` in the filename. Bus faults are not included in this dataset.
- **`Crit_gen`**: ENTSO-E EIC codes, e.g. `28W-G-0000000009`. 73 unique generators.
- **`Location`**: PowerFactory element IDs for the faulted line, e.g. `10T-AT-SI-00003P`. 121 unique locations.

### FSA files (`FSA_main_N.csv`)

- **No CSV quirks**: unlike TSA, FSA parses fine as-is with plain `pd.read_csv` - **do not** apply TSA's missing-separator regex fix here. An earlier attempt did, and it corrupted FSA's multi-digit values (e.g. the `M1`/`M2`/`M3` margin metrics' `100,0`), because that regex only expects a single digit before the decimal comma - it silently split `100,0` into `10` and `0,0`, breaking the CSV's field count.
- **Wide-to-long**: unlike TSA's numeric `_N` suffix (reshapable via `pd.wide_to_long`), FSA's per-contingency columns are suffixed with a compound `<failed_gen>_<measured_gen>` id (e.g. `minF_28W-G-0000000017_28W-G-000000121Y`) - reshaped manually (stack per metric, concat) rather than via `pd.wide_to_long`, then split into two columns on the single `_` separator (every id has exactly one, confirmed empirically - generator EIC codes use dashes internally, never underscores). See `melt_fsa()` in `transform.py`.
- **`failed_gen`/`measured_gen`**: intentionally *not* named `Crit_gen` like TSA - TSA's `Crit_gen` is a simulation *outcome* (the generator whose rotor angle diverges, determining CCT), whereas `failed_gen` is the *initiating* contingency (the generator deliberately disconnected) - conceptually closer to TSA's `Location` than to `Crit_gen`. `measured_gen` avoids `obs_gen`/`Obs*` since that prefix already means something else in this dataset family (SSSA's observability factor).
- **`MThreshold1/2/3`**: three global scalar columns (0.99/0.97/0.95 - 99%/97%/95% of nominal frequency), confirmed identical across all 89 batches. Dropped before reshaping rather than carried as per-row data.
- **Metrics per (failed_gen, measured_gen) pair**: `minF`, `maxF`, `maxRoCoF` (frequency in pu, RoCoF in pu/s), plus `M1`/`M2`/`M3` margin-violation metrics relative to the three thresholds above.
- **Missing results**: about half of all `(state, failed_gen, measured_gen)` triples have no result at all - `measured_gen` was itself out of service in that state, so there was nothing to measure. All 6 metrics are null together for those triples (confirmed empirically); such rows are dropped rather than treated as invalid data.
- **Row count**: 6,127,620 valid rows across all batches; 12,566 unique `(failed_gen, measured_gen)` pairs (not all present in every batch).

### Ignorelist (`ignorelist.txt`)

Lists specific rows to skip per batch file, format: `{date}_x_main_{N}.csv, ID = {row};`. Identified as corrupted or physically implausible operating states. Applied via `skiprows` when reading LF, TSA, and FSA CSVs.
