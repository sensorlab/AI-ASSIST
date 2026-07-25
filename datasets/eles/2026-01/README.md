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
| `SSSA_main_N.csv` | Small-signal stability indices |
| `PQdiff_main_N.csv` | P/Q differences — not used |
| `Date_main_N.csv` | Timestamps for each row — not used |

## Transform Script

`transform.py` processes all 89 batches in parallel, produces ML-ready pickles in `interim/` (`lf.pkl`/`tsa.pkl`/`fsa.pkl`/`sssa.pkl`/`topology_cols.joblib.z`) - SSSA's per-mode and per-generator-participation tables are melted separately then joined into that one `sssa.pkl` (see SSSA section below). SSSA is parsed and wired into `EstimationService.estimate_sssa_by_generator()` (deliberately raw/unweighted - see `TODO.md`), but not yet exposed via the API.

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
- **`oserv_*` columns**: Arrive as string-encoded floats (`"1.0"`, `"0.0"`) with NaN for missing entries. Cast chain: `astype("string") → fillna("1.0") → astype(float) → astype(int) → astype(bool)`. **Documented-vs-actual encoding discrepancy**: the `make_scaler_eles` comment in `service.py` (and this file, previously) claimed `True` (~= out of service), `False` (~= in service). Empirical validation on 2026-07-25 against power output/flow columns contradicts this for LINES: for the 254 dictionary-matched "Slovenian" lines, `True` (their constant value across every record) correlates with nonzero power flow (mean 36.7 MW) - **the correct reading for lines/buses is `True` ~= in service/active**, the opposite of what was documented, and this is the one to trust. For GENERATORS the same clean correlation was found (power nonzero in 100% of `True` records, exactly 0 in 100% of `False` records, 165/165 checked) but is flagged as **unresolved, not confirmed** - it may reflect that `oserv_Gen*` is derived from the power value by the partner's export pipeline (a possible programmatic artifact) rather than an independently-recorded status, so don't treat the generator case as validated the way the line case is. Neither reading changes any topology-grouping results already computed (bitstring-equality grouping is invariant to which symbol means what), only the narrative of what a constant value means. NaN fills as `"1.0"`. See `datasets/eles/2026-06/README.md`'s "Topology Variants" section for the full investigation.
- **Topology column selection**: Not all `oserv_*` columns are used as topology features. The `powerfactory_dictionary.xlsx` (sheets: `Lines`, `Loads`) provides a list of elements to include. The `Generators` sheet is excluded — too many topology changes across operating states make generator status unreliable as a feature. **Observed 2026-07-25: this dictionary-matched subset (254 columns) is constant across every single record in the dataset** - it collapses `topology_id` to one value for all 4,402 states, meaning exact-topology retrieval is currently a no-op for this dataset. This reflects the current data batch, not a flaw in the dictionary-matching logic (re-derived independently and confirmed correct) or in the "Slovenian topology" concept itself: ELES/the partners are still producing more simulation batches (slow to generate), and this window simply contains no recorded Slovenian topology change yet - it may become a genuinely useful, arguably the most physically correct, topology definition once more diverse batches arrive. Real topology variation in the *current* data lives entirely outside this subset (in generator columns and in ~512 non-dictionary-matched line columns, mostly named after Croatian grid elements - i.e. the wider multi-country model). **This dataset (`2026-01`) is superseded by `eles/2026-06` for topology work going forward** - see `datasets/eles/2026-06/README.md`'s "Topology Variants" section for the full investigation, the corrected fix, and which dataset/variant the paper actually uses.

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

### SSSA files (`SSSA_main_N.csv`)

Batched the same way as LF/TSA/FSA (one row per state within each batch), unlike `eles/2026-06` where SSSA is one file per state instead. Reshaped separately (`melt_sssa_modes()`/`melt_sssa_participation()`, mirroring `eles/2026-06`'s split but with a genuinely different participation-table shape, see below) then joined into one `sssa.pkl`/`sssa.db`:

**Units** (per the data dictionary):

| Column(s) | Unit | Notes |
|---|---|---|
| `real_part` | 1/s | Damping |
| `imag_part` | rad/s | Frequency |
| `ConMag`/`ObsMag`/`ParMag` | dimensionless | "Relative" amplitude per the dictionary |
| `ConAng`/`ObsAng`/`ParAng` | degrees | |

- `RealPart_ModeN`/`ImagPart_ModeN` (eigenvalue: damping in 1/s, frequency in rad/s) -> one row per `(state, mode_id)` with `real_part`/`imag_part` columns. Plain numeric mode suffix, reshaped via `pd.wide_to_long` - same as `eles/2026-06`.
- Per mode, `{ConMag|ConAng|ObsMag|ObsAng|ParMag|ParAng}_ModeN_<generator>` -> one row per `(state, mode_id, generator)` with exactly 6 metric columns. **No per-generator-state-variable breakdown here** (confirmed empirically across several batches: every metric column is directly `{metric}_ModeN_{generator}`, never `{metric}_ModeN_{state_variable}_{generator}`) - unlike `eles/2026-06`, which does break these down per state variable (`speed`/`phi`/`Psi1d`/`Psifd`/`Psi1q`/`Psi2q`). `ConMag`/`ConAng` **are** present in this drop (confirmed absent in `eles/2026-06`) - PowerFactory's SSSA export evidently ran with different reporting settings for the two data drops (the dictionary itself flags the state-variable breakdown as "optional depending on the settings used when executing simulations").
- The two are inner-joined on `(state, mode_id)` into one `sssa.pkl`/`sssa.db` - `real_part`/`imag_part` repeat across every generator row sharing a mode, since they're mode-level values, not generator-specific ones. Done once at parse time, not per query. Some rows legitimately end up with all 6 metric columns `NaN` (a generator reported for a mode with no computed participation) - kept as-is, not dropped, per the "raw, unfiltered retrieval" design of `estimate_sssa_by_generator()` (see below).
- **Generators**: 51 unique named-plant generators observed in one batch, no `sym_<N>_<k>` numeric-ID machines seen (unlike `eles/2026-06`, which references both named plants and numeric-ID equivalents). `normalize_sssa_generator()` only fixes a stray underscore before some dashes (`NEK_-G1` -> `NEK-G1`) - it does **not** map to TSA/FSA's EIC codes. That mapping exists in `powerfactory_dictionary.xlsx`'s `Generators` sheet (`dummy_name` -> `for_name`) but is deliberately deferred until SSSA has an actual consumer.
- **Row counts (~35x `eles/2026-06`'s per-file scale)**: `sssa.pkl` is large (tens of millions of rows, multiple GB) - a direct consequence of the much larger mode count per batch here, not a parsing bug.
- **`mode_id` is a per-state local identifier only - never compare, join, or aggregate it across different `state` values.** Mode range observed 0-181 in a single batch (much larger than `eles/2026-06`'s per-file mode counts). Per the data dictionary: *"Nihajna načina v dveh različnih obratovalnih stanjih z istim imenom (se pravi Mode_0 v dveh različnih obratovalnih stanjih) nista nujno povezana (ni nujno da gre za isti nihajni način)"* - two oscillatory modes with the same name in two different operating states are not necessarily related; it's not necessarily the same oscillatory mode. Column named `mode_id` rather than `mode` specifically to make cross-state comparison harder to reach for by accident. Parsed and wired into `EstimationService.estimate_sssa_by_generator()`, which groups by `generator` only (never `mode_id`) and returns raw, unweighted retrieval - no aggregation, pending domain input on what (if any) is wanted (tracked in `TODO.md`). Not yet exposed via the API.

### Ignorelist (`ignorelist.txt`)

Lists specific rows to skip per batch file, format: `{date}_x_main_{N}.csv, ID = {row};`. Identified as corrupted or physically implausible operating states. Applied via `skiprows` when reading LF, TSA, and FSA CSVs.
