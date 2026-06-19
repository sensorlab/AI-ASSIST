# ELES Dataset (2026-01)

Real-world power system data from ELES (Slovenian transmission system operator). Contains 4,402 hourly operating states sampled June 2024 – January 2025, with transient stability results for up to 172 line contingency scenarios per state (~324K total TSA records).

## Source Files

| File | Role | Notes |
|---|---|---|
| `raw/SLOJun2024_Jan2025_only_lne_1h.zip` | Archive of all raw CSVs | 89 batches × 6 file types = 534 CSVs |
| `raw/ignorelist.txt` | Rows to skip during loading | Corrupted or invalid measurements |
| `raw/powerfactory_dictionary.xlsx` | PowerFactory element dictionary | Used to identify topology columns via `Lines` and `Loads` sheets |

### Raw archive structure

Each batch `N` (0–88) contains:

| File | Role |
|---|---|
| `LF_main_N.csv` | Load-flow features for one time batch |
| `TSA_main_N.csv` | Transient stability results (wide format) |
| `FSA_main_N.csv` | Frequency stability indices — not used |
| `SSSA_main_N.csv` | Small-signal stability indices — not used |
| `PQdiff_main_N.csv` | P/Q differences — not used |
| `Date_main_N.csv` | Timestamps for each row — not used |

## Transform Script

`scripts/eles/transform.py` — processes all 89 batches in parallel, produces ML-ready pickles in `data/eles/2026-01/interim/`.

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

### Ignorelist (`ignorelist.txt`)

Lists specific rows to skip per batch file, format: `{date}_x_main_{N}.csv, ID = {row};`. Identified as corrupted or physically implausible operating states. Applied via `skiprows` when reading both LF and TSA CSVs.
