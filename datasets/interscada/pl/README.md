# Interscada PL Dataset

Power system simulation data for a 41-bus grid (based on the New England 39-bus system extended to 41 buses). Contains 761 operating states and transient stability results for 63 fault scenarios per state.

## Source Files

| File | Role | Rows |
|---|---|---|
| `raw/_power_flow_info_updt.csv` | Load-flow features (LF) | 761 |
| `raw/_CCT_results_names_ok.csv` | Critical Clearing Time results (TSA) | 761 × 63 experiments |

## Transform Script

`transform.py` produces ML-ready pickles in `interim/`.

## Design Decisions

### Power flow file (`_power_flow_info_updt.csv`)

- **Encoding**: UTF-8 with BOM (`utf-8-sig`) — the file starts with a BOM marker.
- **Decimal separator**: `.` (unlike the bus39 dataset which uses `,`).
- **Duplicate column headers**: Generator, load, and line groups repeat the same column names (`Gen_bus_no, status; Gen_P_MW; Gen_Q_mvar` × 10, load × 19, line × 34). Pandas auto-suffixes duplicates as `.1`, `.2`, etc., which become `.01`, `.02`, ... after `standardize_col_name`.
- **Compound `Gen_P_MW` field**: Each generator's P_MW column contains `" STATUS, P_MW"` (e.g. `" 1, 69.45"`), where status (0/1) and active power are packed together separated by a comma. These are split into separate `_gen_status_*` and `_Gen_P_MW_*` columns during transformation.
- **`Gen_bus_no, status` columns**: Despite the name, these only contain the bus number — the status is embedded in the `Gen_P_MW` column (see above).
- **Line columns**: Each line is represented as a `(line_bus_no_From, line_bus_no_To, line_status)` triplet. Since bus numbers are constant across all states (fixed grid topology), they are used to rename `line_status` columns to `line_status_{from}_{to}` (e.g. `line_status_01_02`), then dropped. Parallel lines between the same bus pair receive a `_p1` / `_p2` suffix (e.g. `line_status_01_40_p1`, `line_status_01_40_p2`).

### CCT file (`_CCT_results_names_ok.csv`)

- **Wide-to-long reshape**: Each row is one operating state; columns `3PF_location_001 / CCT_001 / critical_generator_001` ... `_063` encode 63 fault experiments. After `standardize_col_name` these become `03PF_location_01` / `CCT_01` / `critical_generator_01`, and `pd.wide_to_long` reshapes to one row per (state, experiment). The `03PF_location` column is then renamed to `Location` to drop the PowerFactory-specific "3PF" prefix that `standardize_col_name` mangles to `03PF`.
- **Stable faults**: When a fault is stable (CCT = 500 ms, the simulation ceiling), `critical_generator` is the string `"None"`. This is normalised to `NaN` — so `critical_generator` is intentionally nullable and excluded from the null-completeness assertion.
- **CCT units**: Milliseconds (integer), range 50–500 ms.
- **`critical_generator` format**: PowerFactory exports the critical generator as `"{bus_no} [{bus_name}   {voltage_kV}]"` (e.g. `"37 [BUS37       13.8]"`). The bus number is extracted into a separate column. The 13.8 kV is the generator terminal bus voltage level — all generators in this grid connect at 13.8 kV and step up to 345 kV via transformers.
- **`Crit_gen` column**: Named `Crit_gen` (not `Crit_gen_bus`) for compatibility with other datasets in this project, even though the value is a bus number rather than a generator identifier. The project partners did not provide generator names — only the terminal bus number is available.
- **`voltage_level_kV` column**: Generator terminal bus voltage level extracted from the same PowerFactory reference string. Constant at 13.8 kV for all generators in this grid.

### Topology columns

Columns treated as topology (discrete switching state) rather than continuous features:

- `line_status_*` — 37 columns, one per transmission line (1 = in service, 0 = tripped)
- `_gen_status`, `_gen_status.01` … `_gen_status.09` — 10 columns, one per generator (1 = online, 0 = offline)
