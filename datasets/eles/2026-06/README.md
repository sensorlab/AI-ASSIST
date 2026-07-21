# ELES Dataset (2026-06)

Real-world power system data from ELES (Slovenian transmission system operator), a newer drop of the same kind of data as `eles/2026-01` but in a different archive layout and already pre-cleaned. Contains 4,393 hourly operating states, with transient stability results for up to 172 line contingency scenarios per state (~323K total TSA records). LF column count (10,298) and grid model scale (Crit_gen: 73 unique generators, Location: 120 unique lines, CCT range 0.11-1.95s) match `eles/2026-01` closely, consistent with the same underlying grid model.

Prepared by: Matjaž Škrlec, matjaz.skrlec@fe.uni-lj.si (per the bundled data dictionaries)

## Source Files

| File | Role | Notes |
|---|---|---|
| `raw/Podatki_DSA.zip` | Archive of all raw CSVs + data dictionaries | See layout below |

### Raw archive structure

Unlike `eles/2026-01`'s flat per-batch layout, this archive nests files under `clean_files/<Type>/`:

| Path | Role |
|---|---|
| `clean_files/LF/LF_main_N.csv` | Load-flow features for batch `N` (0-88) |
| `clean_files/TSA/TSA_main_N.csv` | Transient stability results for batch `N` (wide format) |
| `clean_files/FSA/FSA_main_N.csv` | Frequency stability indices for batch `N` - **not yet parsed** |
| `clean_files/SSSA/<YYYYMMDD>_<HHMM>_SSSA.csv` | Small-signal stability indices, **one file per state** (not per batch) - **not yet parsed** |
| `clean_files/Dates/Date_main_N.csv` | Maps each row of batch `N` to its `<YYYYMMDD>_<HHMM>` timestamp, which is exactly the SSSA filename key for that state. Contains a stray extra `LF_main_9.csv` file that doesn't belong here (harmless, unused). |
| `Explenation_of_samples.docx` / `Razlaga_zapisa_vzorcev.docx` | English/Slovenian data dictionaries (same author/format as `datasets/bus39/README.md`'s dictionary) |

No `PQdiff_main_N.csv` in this drop (present in `2026-01`, was always unused). No `ignorelist.txt` and no `powerfactory_dictionary.xlsx` this time - see Design Decisions.

## Transform Script

`prepare.py` extracts the ZIP preserving its `clean_files/<Type>/` structure (not flattened - a flattened extract would let the stray `Dates/LF_main_9.csv` silently overwrite the real `LF/LF_main_9.csv`). `transform.py` processes all 89 batches in parallel for LF + TSA + FSA, producing `lf.pkl`/`tsa.pkl`/`fsa.pkl`/`topology_cols.joblib.z` in `interim/`. SSSA parsing is not implemented yet (tracked in the repo's `TODO.md`).

## LF Feature Groups

| Group | Columns | Notes |
|---|---|---|
| Bus voltage | `U_Bus*_[pu]` × 2226 | 0 if bus disconnected |
| Phase angle | `phi_Bus*_[deg]` × 2226 | Relative to reference bus |
| Line power (dual-sided) | `P1_Lne*_[MW]`, `P2_Lne*_[MW]`, `Q1_Lne*`, `Q2_Lne*` × 260 each | Measured on both ends of a line, per the PowerFactory model side |
| Generator/load power (single-sided) | `P_*_[MW]`, `Q_*_[Mvar]` × 1163 each | Covers both generators (`P_G01`) and loads (`P_Load03`); 0 if the element is out of service |
| Short-circuit power | `Sk_*_[MVA]` × 1408 | Per-element short-circuit capacity |
| Topology | `oserv_Gen*`, `oserv_Line*` × 1072 | See below |

## Design Decisions

### LF files (`clean_files/LF/LF_main_N.csv`)

- **Separator / decimal**: `;` separator, `,` decimal (European locale) - same as `eles/2026-01`.
- **State ID**: `"{batch_index}_{row_index}"`, same convention as `eles/2026-01`.
- **`oserv_*` semantics** (confirmed directly from the bundled data dictionary): `0` means the element is active (in service), `1` means it is not active (out of service). Cast the same way as `eles/2026-01`: string-encoded floats -> bool, NaN filled as `"1.0"` (treated as out of service).
- **Topology column selection**: unlike `eles/2026-01`, this drop needs no external PowerFactory dictionary - `oserv_*` columns are usable directly as topology features (same simple `oserv_` prefix filter as `bus39`), since every `oserv_*` column here is already a real in/out-of-service flag rather than a mix that needs cross-referencing against a `Lines`/`Loads` sheet.
- **Bus names contain literal backslashes and parentheses** (e.g. `U_BusAJDOVSCINA110\T.L1.1_[pu]`, `U_BusBERICEVO110\T3.2(1)_[pu]`) - harmless for CSV/pandas parsing, just visually unusual.

### TSA files (`clean_files/TSA/TSA_main_N.csv`)

- **Wide-to-long**: identical structure to `eles/2026-01`/`bus39` - `CCT_N / Terminal_N / Crit_gen_N / Type_N / Location_N` groups reshaped with `pd.wide_to_long`.
- **No CSV quirks this time**: unlike `eles/2026-01`, the raw text has no missing-decimal-separator issue - `clean_files/` really is pre-cleaned, so no regex pre-processing is needed before `pd.read_csv`.
- **`Type`**: always `1` (line contingency), same as `eles/2026-01`.
- **`Crit_gen`**: 73 unique generators. **`Location`**: 120 unique lines. **`CCT`**: seconds, range 0.11-1.95s. All match `eles/2026-01` closely.

### No ignorelist this time

`eles/2026-01` needed `ignorelist.txt` to skip specific corrupted rows per batch. This drop has no equivalent file, and no rows were dropped for that reason during parsing - the `clean_files/` naming suggests ELES/SensorLab already filtered those out upstream before packaging.

### FSA files (`clean_files/FSA/FSA_main_N.csv`)

One row per state, then repeating 3-column groups per **contingency**, identified by a `<failed_gen>_<measured_gen>` pair (per the data dictionary: the first generator is the one disconnected for the fault, the second is where the frequency was measured) - e.g. `minF_28W-G-0000000017_28W-G-000000121Y`. Columns per contingency: `minF`, `maxF`, `maxRoCoF` (frequency in pu, RoCoF in pu/s). **No `M1/M2/M3` margin columns or `MThreshold` columns in this drop** (present in `eles/2026-01`, absent here - confirmed by direct header inspection, 7261 columns = 1 index + 2420 contingencies × 3 metrics exactly).

- **No CSV quirks**: parses fine as-is with plain `pd.read_csv`, same as this drop's LF/TSA.
- **Wide-to-long**: the compound generator-pair suffix isn't `pd.wide_to_long`-compatible (no numeric suffix) - reshaped manually (stack per metric, concat), then split into `failed_gen`/`measured_gen` columns on the single `_` separator (every id has exactly one, confirmed empirically). See `melt_fsa()` in `transform.py`.
- **`failed_gen`/`measured_gen`, not `Crit_gen`/`obs_gen`**: intentionally not named after TSA's `Crit_gen` - that column is a simulation *outcome* there (the generator whose rotor angle diverges), whereas `failed_gen` is the *initiating* contingency, a different concept. `obs_gen` was avoided too since `Obs*` already means something else in this dataset family (SSSA's observability factor, see below).
- **Missing results**: about half of all `(state, failed_gen, measured_gen)` triples have no result at all - `measured_gen` was itself out of service in that state. All 3 metrics are null together for those triples (confirmed empirically); such rows are dropped rather than treated as invalid data.
- **Row count**: 5,969,844 valid rows across all batches; 4,664 unique `(failed_gen, measured_gen)` pairs (not all present in every batch).

## SSSA - structure confirmed, parsing not yet implemented

Real structure, confirmed by inspecting the raw CSVs directly and the bundled data dictionaries (differs from `eles/2026-01` - do not assume `2026-01`'s eventual SSSA parser will apply unmodified):

One file per state (`<YYYYMMDD>_<HHMM>_SSSA.csv`, filename matches the `DateTime` column in the corresponding `Dates/Date_main_N.csv` row), not one wide file per batch like `eles/2026-01`. Each file has exactly one data row. Columns: `RealPart_ModeN`/`ImagPart_ModeN` (eigenvalue: damping in 1/s, frequency in rad/s) plus, per mode, optional `ConMag/ConAng/ObsMag/ObsAng/ParMag/ParAng_ModeN_<state_variable>_<generator>` columns (controllability/observability/participation factors) - **the data dictionary explicitly states these per-generator columns are optional depending on simulation settings**, so column sets differ file to file. Mode numbering does not reliably start at `Mode0` (observed starting at `Mode3` in one sample) and, per the data dictionary: **"Oscillatory modes of different operating points with the same names, aren't necessarily the same oscillatory modes"** - modes cannot be joined/grouped across states, only per-state aggregate statistics are safe to derive.
