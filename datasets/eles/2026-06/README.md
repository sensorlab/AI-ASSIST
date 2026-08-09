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

## Topology Variants (Investigation, Implementation, and Accuracy Ablation, 2026-07-25)

**Status: implemented and benchmarked.** `QdrantConfig.topology_variant` (env `TOPOLOGY_VARIANT`,
default `lines_only`) and variant-aware `_dataset_paths()`/`build_estimation_service()` resolution are in
production (see this repo's root `TODO.md`); the three `topology_cols_<variant>.json` files below are
generated by `transform.py` and already exist under `interim/`/`processed/`. This section is kept as the
investigation record - the reasoning behind the choice of `lines-only`, not a still-open plan. `paper/TODO.md`
and this repo's root `TODO.md` both point here.

**Important follow-up finding (2026-07-25, see "Topology Matching Accuracy Ablation" below): enabling
the `lines-only` filter does not improve retrieval accuracy relative to no filter at all - its only
demonstrated effect is a large coverage cost.** Read "the sane default" language throughout this section
as a physical-validity/interpretability justification, not a claim that it was shown to improve `CCT`
estimates - it wasn't, on the data checked so far.

### What's wrong with the current single `topology_cols.json`

`EstimationService`'s exact-topology-match filter (`_get_topology_id()` in
`src/services/qdrant/repository.py`) concatenates a dataset's `topology_cols.json` columns into a
per-record bitstring; retrieval is restricted to records with an identical bitstring. Checking this
empirically against `eles/2026-01` (used in the paper) and `eles/2026-06` (identical underlying
1072-column `oserv_*` schema, verified column-name-for-column-name identical between the two drops):

| Topology definition | # cols | Unique groups (of 4,402/4,393 records) | % records with >=1 same-group neighbor | Max group size |
|---|---|---|---|---|
| **"full"** - every `oserv_*` column (current default for `eles/2026-06`) | 1072 | 4,281 | 4.80% | 5 |
| **"slovenia-only"** - `eles/2026-01`'s dictionary-matched subset (`powerfactory_dictionary.xlsx`, `Lines`+`Loads` sheets, generators excluded) | 254 | **1** | 100% (vacuously - it's one giant group) | all records |
| **"lines-only"** - all `oserv_Lne*`/`oserv_Line*`, generators excluded, no dictionary needed | 907 | 1,785 | **76.12%** | 38 |

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
usefully distributed (273 groups of 2, down to a handful of larger groups up to 38), and it agrees
with "slovenia-only" in every case checked so far, since the dictionary-matched columns have been
constant in both datasets checked (so "lines-only" strictly generalizes "slovenia-only" on today's
data - that changes if/when domestic topology variation shows up in a future batch; this is a fact
about today's data, not a guarantee that holds independently of it).

### Rationale for excluding generators from the topology key

Topology should mean network connectivity (which lines/buses are in service), not generator dispatch
(which unit happens to be committed this hour) - these are different concepts in power-system
operation, and conflating them is why "full" fragments so badly (165 independently-switching
generators multiply combinatorially). Empirically checked on 2026-07-25: for all 165 generators
checked, output power is nonzero in 100% of "committed" records and exactly 0 in 100% of "off" records -
a perfect, exceptionless correlation in the raw `P_Gen*`/`Q_Gen*` columns. This shows only that the
on/off portion of generator status is recoverable from the continuous power features already in the
embedding (`embed_cols`) - it does not establish that every retrieval-relevant aspect of generator
commitment (availability, dispatch/controller semantics, or dynamic-model identity) survives dropping
`oserv_Gen*` from the hard equality key; two states matched on lines-only can still differ in generator
commitment in ways the continuous features don't fully capture. Excluding generators is a fragmentation
tradeoff accepted for coverage, not a proven equivalence (Codex review, `paper-sr/ai2ai.md`, 2026-08-10).

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
5. **Done 2026-07-25**: all of the above (points 2-4) is implemented and in production
   (`scripts/service/eles_benchmark.py` reruns the confidential `eles/2026-06` leave-one-group-out
   benchmark under `lines-only` through the real `build_estimation_service()` path, not a hand-rolled
   substitute; `scripts/service/eles_topology_candidate_eval.py`, referenced below, is the superseded
   exploratory precursor kept only for history). Full-population `lines-only` result: MAE 0.0488 ->
   0.0437, RMSE 0.1119 -> 0.1046, coverage 99.45% -> 63.6%. These are the numbers reported in
   `paper.tex` section 6.2. The earlier exploratory run against `eles/2026-01` (`report-eles2026-01-
   candidate-topology.joblib` / `risk_coverage_eles2026-01-candidate-topology.csv`, gitignored,
   confidential) is superseded, kept only as a sanity check that the production rerun reproduced the
   same pattern.

### Topology Matching Accuracy Ablation (2026-07-25)

The coverage cost above (63.6% of records get a location-specific estimate under `lines-only`) only
establishes that the filter is a real, non-degenerate restriction - it says nothing about whether
*enforcing* the match actually improves accuracy relative to ignoring topology altogether. This was
checked directly, using existing production code with no new development: `slovenia-only` (see the table
above) still collapses to a **single** topology value across every one of the 4,393 `eles/2026-06` states
(re-verified 2026-07-25), so on this data it is functionally a no-op filter - a genuine "topology matching
disabled" baseline, obtained without building a dedicated on/off switch.

A full leave-one-group-out pass with the filter disabled turned out to be computationally intractable
(500 of 4,393 states took 3.1h before being killed - unrestricted retrieval aggregates over the full
~4,400-state pool and far more distinct `Crit_gen` groups per query, not just a bigger loop). The ablation
instead uses a fixed random subsample of 300 of the 4,393 states as queries (seed 42, same convention as
the paper's K/alpha sensitivity sweep), identical across both arms; retrieval itself still searches the
full reference population - `scripts/service/eles_benchmark.py`'s `ELES_BENCHMARK_SAMPLE_STATES` env var.

**Result**: enabling `lines-only` reduces coverage from 99.6% to 63.0% on this subsample (consistent with
the full-population numbers above). Restricted to the 13,856 records both arms actually answer (matched by
state/generator/location/true CCT), the filter-enabled arm gets MAE 0.0426s/RMSE 0.0992s, and the
filter-disabled arm gets MAE 0.0413s/RMSE 0.0957s on the **identical** records - marginally *lower*, not
higher. The difference is small (~3% relative) but statistically resolvable (paired Wilcoxon signed-rank,
p=7.3e-6, n=13,856); 45.3% of paired predictions are byte-identical (the filter was already non-binding -
the true nearest neighbors already shared topology), and the median paired difference is exactly zero, so
the effect is concentrated in a minority of queries rather than a uniform shift.

**Conclusion: topology matching is not shown to improve scalar CCT accuracy on this data/K.** Its
justification is physical validity/interpretability (a query shouldn't be answered using neighbors from an
incompatible network configuration, whether or not doing so happens to help the error metric), and its
one measured cost is the coverage drop, not an accuracy trade-off in its favor. `paper.tex` sections 4
(Introduction contributions), 6.2 (Results), 7.2 (Discussion), and 7.3 (Limitations) were rewritten around
this framing - see `paper/TODO.md`'s ablation entry for the full write-up and exact wording.

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

## SSSA Generator-Set Matching (Investigation, 2026-07-28)

**Status: investigation only - no production code changed.** We're finalizing a beta version of the SSSA API endpoint, and `mode_id`'s non-comparability across states (see above) means the endpoint needs some other notion of what a live query actually matches on. `EstimationService.estimate_sssa_by_generator()` already groups by `generator` since generator identity, unlike `mode_id`, is at least nominally comparable across states - but that only works well if states broadly agree on which generators they report SSSA data for. This investigation checks whether they do, at two granularities: across different states, and across a single state's own modes. Both checks group by exact set equality only - `{"G1","G2","G3"}` is a different group from `{"G1","G2","G3","G4"}`, no partial-overlap credit given, mirroring the topology bitstring's exact-match semantics above.

### State-vs-state: does every state cover the same generators?

Produced by `scripts/service/eles_sssa_generator_set_eval.py` (pure pandas over `interim/sssa.pkl`, no Qdrant/`EstimationService` involved): per state, the set of generators with at least one SSSA participation row, grouped by exact-match signature.

| Dataset | States total | States w/ SSSA | Distinct generator-sets | Max group size | % states with an exact-match twin | Mean set size | Median set size |
|---|---|---|---|---|---|---|---|
| `eles/2026-01` | 4,402 | 4,402 (100%) | 86 | 149 | **100.0%** | 52.5 | 53.0 |
| `eles/2026-06` | 4,393 | 4,247 (96.7%) | **4,212** | 3 | **1.6%** | 70.2 | 70.0 |

Opposite outcomes: `eles/2026-01` collapses into only 86 sets total, every state has an exact-match twin (in fact most have many - max group size 149). `eles/2026-06` fragments almost entirely - 4,212 distinct sets across 4,247 covered states, so ~98.4% of states have no exact-match twin at all, and the largest group anywhere is 3 states. This is the same shape of finding as the topology-column investigation above (an overly strict exact-match filter either fragments into singletons or is a healthy filter, depending on the dataset).

### Within a state: do all its modes agree on which generators contribute?

A different question from the above (which unions generators across all of a state's modes before comparing states) - this checks consistency *within* one state, across its own modes. Produced interactively in `reports/xx_sssa_analysis.ipynb`, reading `interim/sssa.pkl` for each dataset, grouping by `(state, mode_id)` to get each mode's generator set, then per state: how many distinct sets appear across its modes, and the largest number of modes sharing one exact set.

`eles/2026-06` - distinct generator sets per state (`n_distinct_sets_per_state.value_counts()`), and the within-state max group size (`max_group_size_per_state.describe()`):

| Distinct sets in a state | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| # states | 13 | 55 | 77 | 136 | 247 | 371 | 408 | 493 | 508 | 479 | 425 | 342 | 263 | 180 | 125 | 61 | 43 | 12 | 6 | 3 |

Max group size per state: mean 1.32, median 1, 75th percentile 2, max 3 (n=4,247 states).

`eles/2026-01` - same two stats:

| Distinct sets in a state | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| # states | 348 | 495 | 971 | 1,148 | 740 | 400 | 151 | 149 |

Max group size per state: mean 155.0, median 155, min 101, max 203 (n=4,402 states; this dataset's mode range goes up to ~181 in a single batch, so a max group size of ~155 means the large majority of a state's modes share one identical generator set).

### Conclusion

**Exact generator-set matching is not usable for `eles/2026-06` at either granularity checked.** State-vs-state, it's nearly as fragmented as the topology "full" definition above; within a single state across its own modes, the median state has no two modes agreeing on the exact same generator set at all (max group size 1). Whatever the eventual beta SSSA endpoint contract looks like, it cannot gate retrieval on exact generator-set identity for this dataset - that's a materially different, more permissive matching problem than what's measured here (see the caution against conflating "relaxed" with "this, but looser" in `TODO.md`), and remains an open design question pending domain input.

`eles/2026-01` shows the opposite pattern at both granularities - coarse across states (86 sets total) and highly consistent within a state (most modes agree). Read this as a likely artifact of that drop's lower export fidelity (per-batch, no state-variable breakdown, and generally reflecting which generators are simply in service rather than genuine per-mode dynamics) rather than evidence that generator-set matching is a solved problem there - it just hasn't been checked against as fine-grained an export.
