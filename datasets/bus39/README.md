# Bus39 Dataset

Power system simulation data for the New England 39-bus test system. Contains 21,783 operating states and transient stability results for up to 59 fault scenarios per state (~1.04M total records after wide-to-long reshape).

Prepared by: Matjaž Škrlec, matjaz.skrlec@fe.uni-lj.si

## Source Files

All raw data is packed in `raw/data.zip`:

| File | Role | Notes |
|---|---|---|
| `LF_main.csv` | Load-flow features (LF) | 21,783 states × 221 columns |
| `TSA_main.csv` | Transient stability targets | Wide format, 5 fields × N experiments per row |
| `FSA_main.csv` | Frequency stability indices | Parsed - see "FSA file" below |
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

### FSA file (`FSA_main.csv`)

Columns repeat as `minF_N / maxF_N / maxRoCoF_N / M1_N / M2_N / M3_N` for N = 0…9 (10 generators), one row per operating state - `pd.wide_to_long` reshapes this to one row per (state, failed generator), 217,822 rows after dropping 8 corrupt rows (see below).

- **No generator names, no measured-generator dimension**: the data dictionary is explicit on both points - "Imena generatorjev niso podane" (generator names are not given) and "Vse veličine so gledane globalno" (all quantities are viewed globally, i.e. the worst case across the whole system is reported, not a per-generator measurement). So this dataset has neither of ELES's two FSA dimensions in their named form: `failed_gen` is stored as an anonymized `fsa_gen_N` label (the dictionary confirms N maps to the *same* physical generator in every state, just not which one), and is **not comparable to TSA's `Crit_gen` names** (`G 03`–`G 10`) even though both range over the same 10 generators. `measured_gen` is a constant `"system"` placeholder, since there's nothing to group by - purely there so this fits `EstimationService`'s shared `(failed_gen, measured_gen)` FSA report shape.
- **Corrupt rows dropped**: 8 of 217,830 rows (0.004%, 7 of them `failed_gen=fsa_gen_0`) carry a simulation-divergence artifact from the source tool rather than a real result - e.g. one row has `minF_0 = -3.7722735745865803e+149` verbatim in the raw CSV, another has `maxRoCoF_0` in the billions. Dropped by a generous physical-plausibility filter (`|minF|`/`|maxF|`/`|maxRoCoF| <= 5`, `M1`/`M2`/`M3` within `[0, 100]`) against a legitimate observed range of `|minF|`/`|maxF| <= 1.16`, `maxRoCoF <= 0.047` - wide enough margin that this should only catch genuine blow-ups, not real severe events. Logged as `fsa_long dropped N rows with implausible (simulation-divergence) values`.
- **`M1`/`M2`/`M3` stub-name collision**: `standardize_col_name()` pads every digit run to 2 digits, including the one inside `M1`/`M2`/`M3` themselves (`M1_0` → `M01_00`) - `transform.py` melts on the padded `M01`/`M02`/`M03` stub names, then renames back to `M1`/`M2`/`M3` for the returned columns. (This same collision looks unhandled in `datasets/eles/2026-01/transform.py`'s `FSA_METRICS`, which still lists `"M1"` as a `melt_fsa()` prefix after the same renaming step - not fixed here, out of scope for this dataset.)

### SSSA (`SSSA_main.csv`)

Not parsed. Small-signal stability modes — real part (damping, 1/s) and imaginary part (frequency, rad/s), 9 modes per state. Unlike FSA, there's no way to fit this into `EstimationService`'s SSSA shape at all: ELES's SSSA is `(state, mode_id, generator)` with a participation-magnitude vector per generator (used for `estimate_sssa_by_generator()`'s grouping and `matched_mode`'s cross-state cosine matching), but BUS39's `SSSA_main.csv` has no generator dimension whatsoever - just `(state, mode_id) -> (real_part, imag_part)`. Forcing it through the existing generator-grouping API would require either a meaningless placeholder generator (breaking `matched_mode`'s cosine-similarity math, which degenerates on empty/all-zero participation vectors) or new generator-less SSSA support in the shared domain model - deliberately not attempted, pending a real decision on which approach is worth the cost.
