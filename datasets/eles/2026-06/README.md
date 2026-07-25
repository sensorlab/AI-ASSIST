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
| `clean_files/FSA/FSA_main_N.csv` | Frequency stability indices for batch `N` |
| `clean_files/SSSA/<YYYYMMDD>_<HHMM>_SSSA.csv` | Small-signal stability indices, **one file per state** (not per batch) |
| `clean_files/Dates/Date_main_N.csv` | Maps each row of batch `N` to its `<YYYYMMDD>_<HHMM>` timestamp, which is exactly the SSSA filename key for that state. Contains a stray extra `LF_main_9.csv` file that doesn't belong here (harmless, unused). |
| `Explenation_of_samples.docx` / `Razlaga_zapisa_vzorcev.docx` | English/Slovenian data dictionaries (same author/format as `datasets/bus39/README.md`'s dictionary) |

No `PQdiff_main_N.csv` in this drop (present in `2026-01`, was always unused). No `ignorelist.txt` and no `powerfactory_dictionary.xlsx` this time - see Design Decisions.

## Transform Script

`prepare.py` extracts the ZIP preserving its `clean_files/<Type>/` structure (not flattened - a flattened extract would let the stray `Dates/LF_main_9.csv` silently overwrite the real `LF/LF_main_9.csv`). `transform.py` processes all 89 batches in parallel for LF + TSA + FSA, producing `lf.pkl`/`tsa.pkl`/`fsa.pkl`/`topology_cols.joblib.z` in `interim/`. SSSA is discovered and processed separately (one file per state, not batched - see SSSA section below): its per-mode and per-generator-participation tables are melted separately, then joined into one `sssa.pkl` (`interim/`) - `real_part`/`imag_part` are mode-level values that legitimately repeat across every generator row sharing a mode, so joining them once at parse time avoids redoing that join on every query. `sssa.db` (`processed/`) is the `SqliteRecordStore`-backed form `EstimationService` actually queries, same convention as `tsa.db`/`fsa.db`. SSSA is parsed and wired into `EstimationService.estimate_sssa_by_generator()` (deliberately raw/unweighted - see `TODO.md`), but not yet exposed via the API.

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
- **`oserv_*` semantics**: previously documented here as `0`=active/in-service, `1`=not-active/out-of-service per the bundled data dictionary. **Correction (2026-07-25) for LINES: this is backwards.** After the string->bool cast (below), `True` correlates with nonzero power flow (in service/active) for all 254/254 dictionary-matched line columns checked - this reversed-polarity reading is the one to trust for lines/buses. **For GENERATORS, the same 100%-clean correlation was also observed (165/165 checked) but is flagged as unresolved, not confirmed** - it may instead reflect that `oserv_Gen*` is derived FROM the power value by the partner's export pipeline (a possible programmatic artifact) rather than an independent in-service flag, so don't cite the generator case as validation the way the line case can be. Either way, cast the same way as `eles/2026-01`: string-encoded floats -> bool, NaN filled as `"1.0"` -> `True`.
- **Topology column selection**: unlike `eles/2026-01`, this drop needs no external PowerFactory dictionary - `oserv_*` columns are usable directly as topology features (same simple `oserv_` prefix filter as `bus39`), since every `oserv_*` column here is already a real in/out-of-service flag rather than a mix that needs cross-referencing against a `Lines`/`Loads` sheet.
- **Bus names contain literal backslashes and parentheses** (e.g. `U_BusAJDOVSCINA110\T.L1.1_[pu]`, `U_BusBERICEVO110\T3.2(1)_[pu]`) - harmless for CSV/pandas parsing, just visually unusual.

## Topology Variants (Investigation and Plan, 2026-07-25)

**Status: investigated and scoped, not yet implemented.** This section exists so the investigation
isn't lost if picked up in a different session. `paper/TODO.md` and this repo's root `TODO.md` both
point here.

### What's wrong with the current single `topology_cols.json`

`EstimationService`'s exact-topology-match filter (`_get_topology_id()` in
`src/services/qdrant/repository.py`) concatenates a dataset's `topology_cols.json` columns into a
per-record bitstring; retrieval is restricted to records with an identical bitstring. Checking this
empirically against `eles/2026-01` (used in the paper) and `eles/2026-06` (identical underlying
1072-column `oserv_*` schema, verified column-name-for-column-name identical between the two drops):

| Topology definition | # cols | Unique groups (of 4,402/4,393 records) | % records with >=1 same-group neighbor | Max group size |
|---|---|---|---|---|
| **"full"** - every `oserv_*` column (current default for `eles/2026-06`) | 1072 | ~4,290 | ~4.8% | 5 |
| **"slovenia-only"** - `eles/2026-01`'s dictionary-matched subset (`powerfactory_dictionary.xlsx`, `Lines`+`Loads` sheets, generators excluded) | 254 | **1** | 100% (vacuously - it's one giant group) | all records |
| **"lines-only"** - all `oserv_Lne*`/`oserv_Line*`, generators excluded, no dictionary needed | 907 | 1,790 | **76.1%** | 38 |

So neither existing option is usable as a real topology filter today: "full" fragments almost every
record into its own singleton (near-zero chance of retrieving anything), and "slovenia-only" collapses
everything into one group (retrieval succeeds 100% of the time only because nothing is actually being
filtered - it's a no-op, not a working filter). **Important: "slovenia-only" collapsing to one group is
not a flaw in that definition or the dictionary-matching logic** (both were independently re-derived and
verified correct, see below) - **it reflects the current data snapshot, not a bad choice of columns.**
ELES/the partners are still producing more simulation batches (each one is slow to generate), and the
data available so far simply doesn't contain any recorded Slovenian topology change during this window.
"Slovenia-only" may become a perfectly good, arguably the most correct, topology definition once more
diverse batches (including actual domestic switching events) are delivered - it should be re-evaluated
against future data drops, not written off. **"lines-only" is the one candidate usable as a genuine
topology filter with the data available right now**: most records land in a group of size >=2, sizes are
usefully distributed (273 groups of 2, down to a handful of larger groups up to 38), and it fully agrees
with "slovenia-only" wherever the dictionary-matched columns aren't constant (they always are, in both
datasets checked so far, so in practice "lines-only" strictly generalizes "slovenia-only" on today's
data, at no cost - that changes if/when domestic topology variation shows up in a future batch).

### Why "lines-only" (excluding generators) is the physically correct choice, not just the one that scored well

Topology should mean network connectivity (which lines/buses are in service), not generator dispatch
(which unit happens to be committed this hour) - these are different concepts in power-system
operation, and conflating them is why "full" fragments so badly (165 independently-switching
generators multiply combinatorially). Empirically checked on 2026-07-25: for all 165 generators
checked, output power is nonzero in 100% of "committed" records and exactly 0 in 100% of "off" records -
a perfect, exceptionless correlation in the raw `P_Gen*`/`Q_Gen*` columns. So excluding generator
`oserv_*` columns from the topology bitstring loses no *retrieval-relevant* information either way:
"generator is off" is already fully recoverable from the continuous power features already in the
embedding (`embed_cols`).

**Caution on interpreting that correlation (flagged 2026-07-25, unresolved)**: a 100%-exceptionless
correlation between a status flag and a continuous value is also exactly what you'd see if
`oserv_Gen*` isn't an independently-recorded PowerFactory in-service flag at all, but is instead
*derived* from the power value by the partner's export/parsing pipeline (e.g. something equivalent to
`oserv = (P == 0)`) - a possible programmatic artifact on the partner's side, specific to generators.
If that's what's happening, the "generator off" flag and "zero power" aren't two independent
confirmations of the same physical fact, they're the same computed fact twice - which doesn't change
the practical recommendation (still exclude generators from topology; the information is redundant
either way) but does mean this shouldn't be cited as strong independent validation of anything beyond
"these two columns agree," and it should **not** be assumed to generalize to how `oserv_Lne*`/other
element types are recorded. **The reversed encoding polarity (`True` ~= in service / active) is
believed to hold reliably for lines/buses** (checked via power-flow correlation on the 254 dictionary-
matched line columns, mean 36.7 MW flow when `True`) **and is the one to trust; the generator case above
is flagged as unresolved/possibly-an-artifact, not confirmed.** See the correction inline above and in
`datasets/eles/2026-01/README.md`. None of this affects any of the grouping counts in the table above
(bitstring-equality grouping is invariant to which symbol means what), only the narrative of what a
constant value means physically, and how much weight to put on the generator-specific validation.

### The plan

1. **Scope**: this paper uses only `bus39` and `eles` data (not `interscada/*`) - the topology-variant
   work below is scoped to `eles` only. Between the two `eles` drops, **use `eles/2026-06` going
   forward** (not `2026-01`, which the exploratory script/results below were actually run against
   before this decision was made - treat those as superseded/exploratory only, not final).
2. **Three named, selectable topology variants**, generated by `transform.py` as separate
   `topology_cols_<variant>.json` files instead of one `topology_cols.json`:
   - `full` - every `oserv_*` column (today's default; kept for continuity/comparison, not recommended)
   - `lines-only` - every `oserv_Lne*`/`oserv_Line*` column, generators excluded - **the sane default**
     going forward, and the one used for every ELES number reported in the paper
   - `slovenia-only` - the dictionary-matched subset; degenerate (collapses to 1 group) on the data
     available as of 2026-07-25, but that's a reflection of the current, still-growing data batch (no
     recorded Slovenian topology change yet), not a flaw in the definition itself - keep it available
     and re-evaluate as ELES delivers more simulation batches
3. **Production config**: add a variant-selector (e.g. a new field on `QdrantConfig`, or an env var
   like `TOPOLOGY_VARIANT`, defaulting to `lines-only`) so `build_estimation_service()` loads
   `topology_cols_<variant>.json` instead of a single hardcoded filename. `_dataset_paths()` in
   `src/domain/estimation/service.py` resolves artifacts by basename glob today (see root
   `ai-assist-v2/CLAUDE.md`) - extend that resolution to take the variant into account.
4. **Implementation touches**: `datasets/eles/2026-06/transform.py` (emit three files instead of one),
   `src/domain/estimation/service.py` (`_dataset_paths()`/`build_estimation_service()`),
   `src/services/qdrant/config.py` (new config field), tests for variant selection, then rerun the
   `eles/2026-06` benchmark under `lines-only` (production path, not the exploratory script below) and
   rewrite `paper.tex` section 6.2's ELES numbers/narrative to match.
5. **Not yet done**: none of the above production/pipeline code exists yet as of 2026-07-25. What
   exists so far is exploratory only: `scripts/service/eles_topology_candidate_eval.py` (in-process,
   embedded-Qdrant evaluation script; does NOT touch any production `topology_cols.json`) was run
   against **`eles/2026-01`** (superseded choice, see point 1) with the `lines-only` column set, and
   produced `report-eles2026-01-candidate-topology.joblib` +
   `risk_coverage_eles2026-01-candidate-topology.csv` at the `ai-assist-v2` repo root (gitignored,
   confidential, not committed). Results (full-coverage MAE 0.0488 -> 0.0437, coverage 99.4% -> 67.4%,
   mixed nAURC changes - distance-based diagnostics got more informative, count/mass-based ones got
   less so) are informative for the design decision but **need to be regenerated against
   `eles/2026-06`** once the production variant-selection code above exists, before anything goes in
   the paper.

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

## SSSA files (`clean_files/SSSA/<YYYYMMDD>_<HHMM>_SSSA.csv`)

One file per state (filename matches the `DateTime` column in the corresponding `Dates/Date_main_N.csv` row), not one wide file per batch like `eles/2026-01` - `transform.py` discovers and processes these with their own `joblib.Parallel` loop (`sssa_processor()`), independent of the batch-`index`-based loop used for LF/TSA/FSA. Each file has exactly one data row. 4,282 files present; both output tables cover all 4,282 states.

**Units** (per the data dictionary):

| Column(s) | Unit | Notes |
|---|---|---|
| `real_part` | 1/s | Damping |
| `imag_part` | rad/s | Frequency |
| `ConMag`/`ObsMag`/`ParMag` (and their `_<state_variable>` variants) | dimensionless | "Relative" amplitude per the dictionary - `ConMag`/`ConAng` absent in this drop, see below |
| `ConAng`/`ObsAng`/`ParAng` (and their `_<state_variable>` variants) | degrees | |

Two column families, reshaped separately then joined into one output table:

- `RealPart_ModeN`/`ImagPart_ModeN` (eigenvalue: damping in 1/s, frequency in rad/s) -> `melt_sssa_modes()`, one row per `(state, mode_id)` with `real_part`/`imag_part` columns. Plain numeric mode suffix, reshaped via `pd.wide_to_long`.
- Per mode, `{ConMag|ConAng|ObsMag|ObsAng|ParMag|ParAng}_ModeN_<state_variable>_<generator>` (controllability/observability/participation factors, broken down per generator dynamic-model state variable: `speed`, `phi`, `Psi1d`, `Psifd`, `Psi1q`, `Psi2q`) -> `melt_sssa_participation()`, one row per `(state, mode_id, generator)`, with the metric+state-variable combination flattened into columns (`ObsMag_speed`, `ParAng_Psi2q`, ...; up to 24 columns - `ConMag`/`ConAng` confirmed absent from every file checked in this drop, unlike `eles/2026-01`). Missingness across state variables tracks generator model fidelity, not noise (`phi` populated in ~92% of `(mode_id, generator)` rows in a sampled file, down to `Psi2q` at ~3%) - deliberately not dropped.
- The two are inner-joined on `(state, mode_id)` into one `sssa.pkl`/`sssa.db` - `real_part`/`imag_part` are mode-level values that lived in the same raw CSV row as every per-generator participation column before melting split that row into one line per generator, so they legitimately repeat across every generator row sharing a mode. Done once at parse time, not per query.
- **`eles/2026-01`'s SSSA columns are a different shape** - no state-variable breakdown at all (`{metric}_ModeN_<generator>` directly), but it does carry `ConMag`/`ConAng` which this drop lacks. The two datasets' `melt_sssa_participation()` are genuinely different functions, not a shared one with different loading code.
- **Generator names**: either abbreviated named-plant codes (`TSOS-G6`, `HDRA-G3`) or bare PowerFactory internal numeric IDs (`sym_8769_1`) for machines with no plant label. `normalize_sssa_generator()` only fixes a stray underscore before some dashes (`NEK_-G1` -> `NEK-G1`) - it does **not** map to TSA/FSA's EIC codes. That mapping exists (`datasets/eles/2026-01/raw/powerfactory_dictionary.xlsx`'s `Generators` sheet, `dummy_name` -> `for_name`) and has been verified to still apply to this dataset's generators, but only covers ~72 named plants, not the numeric-ID ones, and is deliberately deferred until SSSA has an actual consumer.
- **State id mismatch fixed at parse time**: raw SSSA files are keyed by `<YYYYMMDD>_<HHMM>` timestamp (their filename), not the `{batch}_{row}` state id LF/TSA/FSA use. `_load_sssa_state_mapping()` translates via `clean_files/Dates/Date_main_N.csv` (its row index matches `LF_main_N.csv`'s row index for the same batch - verified across multiple batches, not just one). ~35 of 4,282 SSSA timestamps have no corresponding LF row at all (a genuine data gap, not a mapping bug - confirmed by checking every `Date_main_N.csv`) and are dropped (logged as `SSSA modes/participation drop N/M rows with no matching LF state`), not treated as an error.
- **`mode_id` is a per-state local identifier only - never compare, join, or aggregate it across different `state` values.** Per the data dictionary: *"Nihajna načina v dveh različnih obratovalnih stanjih z istim imenom (se pravi Mode_0 v dveh različnih obratovalnih stanjih) nista nujno povezana (ni nujno da gre za isti nihajni način)"* - two oscillatory modes with the same name in two different operating states are not necessarily related; it's not necessarily the same oscillatory mode. `mode_id` numbering doesn't even reliably start at `0` (observed starting at `Mode3` in one sample). Column named `mode_id` rather than `mode` specifically to make cross-state comparison harder to reach for by accident. Parsed and wired into `EstimationService.estimate_sssa_by_generator()`, which groups by `generator` only (never `mode_id`) and returns raw, unweighted retrieval - no aggregation, pending domain input on what (if any) is wanted (tracked in `TODO.md`). Not yet exposed via the API.
