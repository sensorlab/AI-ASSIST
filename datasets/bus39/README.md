# Bus39 Dataset

Power system simulation data for the New England 39-bus test system. Contains 21,783 operating states and transient stability results for up to 59 fault scenarios per state (~1.04M total records after wide-to-long reshape).

Prepared by: Matjaž Škrlec, matjaz.skrlec@fe.uni-lj.si

## Source Files

All raw data is packed in `raw/data.zip`:

| File | Role | Notes |
|---|---|---|
| `LF_main.csv` | Load-flow features (LF) | 21,783 states × 221 columns |
| `TSA_main.csv` | Transient stability targets | Wide format, 5 fields × N experiments per row |
| `FSA_main.csv` | Frequency stability indices | Not used (parser stubbed as `NotImplemented`) |
| `SSSA_main.csv` | Small-signal stability indices | Not used |
| `Bus39.jpg` | Grid diagram | |
| `Razlaga_zapisa_vzorcev.docx` | Slovenian data dictionary (source of truth for column semantics) | |

## Transform Script

`transform.py` reads CSVs directly from the unpacked ZIP, produces ML-ready pickles in `interim/`.

## LF Feature Groups

| Group | Columns | Range | Notes |
|---|---|---|---|
| Bus voltage | `U_Bus_NN_[pu]` × 39 | 0.98–1.11 pu | 0 if bus not in service |
| Phase angle | `phi_Bus_NN_[deg]` × 39 | −2–21° | Relative to reference bus |
| Generator P | `P_G_NN_[MW]` × 10 | −177–1712 MW | 0 if generator offline |
| Generator Q | `Q_G_NN_[MVAr]` × 10 | −406–259 MVAr | 0 if generator offline |
| Load P | `P_Load_NN_[MW]` × 19 | −124–1237 MW | Buses with loads only |
| Load Q | `Q_Load_NN_[MVAr]` × 19 | — | |
| New load | `P_New_Load_[MW]`, `Q_New_Load_[MVAr]` | 0 | Always zero in this dataset |
| Short-circuit power | `Sk_Bus_NN[MVA]` × 39 | 3226–57782 MVA | Per-bus short-circuit capacity |
| Topology | `oserv_G_NN`, `oserv_Line_*` × 44 | 0 or 1 | See below |

## Design Decisions

### Load-flow file (`LF_main.csv`)

- **Separator / decimal**: `;` separator, `,` decimal (European locale).
- **`oserv_*` columns**: Arrive as numeric 0/1 and must be explicitly cast to `bool` before `convert_dtypes` — otherwise pyarrow infers them as integer. The source data dictionary states `0` = in service, `1` = not in service, **but that is contradicted by the archive**: all 44 flags are `1` in every one of the 21,783 states, while those same states carry non-zero generator dispatch and solved bus voltages. So `1` = in service here. There is no topology variation at all in this dataset, which also holds system inertia exactly constant. Grouping is polarity-invariant (it compares bitstrings for equality), so this affects documentation and any per-flag interpretation, not the retrieval results.

### TSA file (`TSA_main.csv`)

- **Wide-to-long**: Each row is one operating state; columns repeat as `CCT_N / Terminal_N / Crit_gen_N / Type_N / Location_N` for N = 0…58. `pd.wide_to_long` with these 5 stubnames reshapes to one row per (state, experiment), yielding ~1.04M rows and 7–59 experiments per state.
- **`Type` column**: `0` = bus short-circuit (self-clearing), `1` = line short-circuit near the terminal bus, cleared by tripping the line. Values arrive as a mix of `float` and string-of-float (e.g. `"1.0"`) — coerced with `apply(float).apply(int)`.
- **CCT units**: Already in seconds, range 0.06 to 1.97 s. Value `0` in the raw data means CCT < 0.06 s (system unconditionally unstable), i.e. those are **left-censored** observations rather than exact values; they land at 0.06 s after parsing (239 of 1,044,797 records, 0.02%). Anything trained or scored on this column treats them as exact 0.06 s, which understates error on them, and a distance-weighted average over records above the floor cannot reproduce the floor. Measured distribution of the parsed column: median 0.43 s, p90 0.69 s, p99 1.15 s, max 1.97 s, with 18.8% of records above 0.6 s and 1.8% above 1.0 s. (An earlier version of this note claimed "very high values (>> 0.6 s) are not recorded"; that is contradicted by the figures above and has been removed. Whether the source workflow caps CCT somewhere beyond 1.97 s is unknown - the observed maximum may be a real cap or simply the largest value encountered.)
- **`Crit_gen`**: Clean generator name string, e.g. `G 03`–`G 10` (first generator to lose synchronism).
- **`Location`**: Element where the short circuit occurred (bus name or line name).
- **`Terminal`**: Bus at or near the fault location.

### Topology columns

`oserv_G_NN` and `oserv_Line_*` — 44 columns identified by regex `oserv?_.*`. The data dictionary claims `0` = in service, `1` = not in service, but the archive is uniformly `1` with the grid demonstrably energised, so `1` = in service. See the Design Decisions note above.

### Unused datasets

- **FSA** (`FSA_main.csv`): Per-generator frequency stability indices — minimum frequency, maximum frequency, max RoCoF, and margins M1/M2/M3 at 95%/97%/99% of nominal frequency. Parser exists in `transform.py` but raises `NotImplementedError`.
- **SSSA** (`SSSA_main.csv`): Small-signal stability modes — real part (damping, 1/s) and imaginary part (frequency, rad/s). Not parsed. Note: mode indices across different operating states do not correspond to the same physical mode.
