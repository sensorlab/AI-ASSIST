# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General guidelines

- NEVER add AI attribution: no agent co-author trailers and no "Generated with ..." lines, in commit messages or PR descriptions.
- When making technical decisions, do not give much weight to development cost. Prefer quality, simplicity, robustness, and long-term maintainability.
- Never use em dashes "—" or en dashes "–". Use the plain dash "-" only where a hyphen belongs, and use even that sparingly.
- Don't hard-wrap Markdown: write each paragraph/list item as one line with no manual line breaks, and don't rewrap existing text to a fixed width. The user's IDE handles soft line wrapping for display.

## Project overview

AI-ASSIST is a research project for real-time power-grid security assessment. Given a live grid operating state, it retrieves the most similar historical/simulated states (by scaled feature distance) from a vector database and uses their known transient-stability outcomes to estimate the Critical Clearing Time (CCT). It is essentially a k-nearest-neighbor risk estimator over power-system simulation records, served via a FastAPI service. It's a joint SensorLab / University of Ljubljana / ELES initiative funded by ARIS (Grant L2-50053).

## Commands

Environment is managed with `uv`. `make` targets wrap the common workflows:

```bash
make install       # uv sync + pre-commit install
make update        # refresh deps, uv lock --upgrade
make pre-commit    # run all pre-commit hooks (ruff check --fix, ruff format, nb-clean, ...)
make clean         # remove __pycache__, *.pyc, and stray *.pkl/*.joblib under datasets/
```

Run the API locally:

```bash
uv run fastapi dev src/api/main.py --port 8000     # dev server with reload
# or use the "Run API (uvicorn)" launch config in .vscode/launch.json
```

Docker (API + Qdrant):

```bash
docker compose up --build
```

Tests (`unittest`, not pytest; pytest is not a project dependency):

```bash
python -m unittest discover -s tests -v                 # full suite
python -m unittest tests.test_estimation_endpoints -v    # single file
python -m unittest tests.test_estimation_endpoints.EstimationServiceEndpointTests.test_estimate_by_generator_matches_existing_generator_first_behavior  # single test
```

Lint/format (ruff, config in `ruff.toml`, line length 120):

```bash
ruff check --fix .
ruff format .
```

Data pipeline, per dataset:

```bash
uv run ai-assist-prepare <dataset>   # bus39 | eles/2026-01 | eles/2026-06 | interscada/pl | interscada/fr
# or: make prepare DATASET=<dataset>
```

Each dataset is a self-contained directory under `datasets/<dataset>/` (or `datasets/<dataset>/<version>/` for versioned datasets like `eles/2026-01`) holding its `raw/`, `interim/`, `processed/`, `prepare.py`, `transform.py` and `README.md`. `prepare.py` unpacks `raw/` (if it's a ZIP archive) into `interim/` and then runs `transform.py`, which writes ML-ready pickles (`lf.pkl`, `topology_cols.joblib`, and analyst-facing `tsa.pkl`/`fsa.pkl`) into `interim/`, plus the same `tsa`/`fsa` tables as indexed SQLite databases (`tsa.db`/`fsa.db`, via `scripts/_common.py::write_sqlite_table()`) and `topology_cols` as a plain JSON list (via `write_json_list()`) into `processed/` - `interim/` is for bulk/notebook consumption, `processed/` is what the live service actually queries (see Architecture below). `scripts/prepare.py` discovers datasets by recursively finding every `prepare.py` under `datasets/` (its path relative to `datasets/`, minus the filename, is the dataset name) and dispatches to the right one - adding a dataset means dropping in a new `datasets/<name>/prepare.py`, no changes to the dispatcher. See `datasets/<dataset>/README.md` for the exact source-file layout and non-obvious parsing decisions (unit quirks, encoding of `oserv_*` topology flags, CSV separator/decimal conventions, wide-to-long reshape rules). Read the relevant one before touching a `transform.py` script.

Standalone evaluation entry points in `scripts/evaluation/` (not part of the data-prep pipeline; some run against a running API, most build an `EstimationService` in-process):

```bash
uv run python scripts/evaluation/benchmark.py       # end-to-end estimation benchmark against a running API
uv run python scripts/evaluation/ml_benchmark.py    # ML-only benchmark
uv run python scripts/evaluation/latency.py         # latency measurement
```

## Architecture

**Request flow**: `src/api/main.py` builds the FastAPI app via `create_app()`; its `lifespan` calls `build_estimation_service()` once at startup and stores the resulting `EstimationService` on `app.state.estimation_service`. `src/api/estimate.py` defines five routes: `/api/v1/estimate/tsa/by-generator`, `/api/v1/estimate/tsa/by-location`, `/api/v1/estimate/fsa/by-observed-generator`, `/api/v1/estimate/fsa/by-failed-generator`, plus `/api/v1/columns`. There is deliberately no bare `/estimate` endpoint and no un-namespaced `/estimate/by-generator`/`by-location` (see `tests/test_estimation_endpoints.py::test_bare_estimate_endpoint_is_removed`/`test_pre_tsa_namespace_paths_are_removed`). The two `fsa/*` routes return HTTP 501 for datasets with no FSA data (`EstimationService.fsa is None`).

**`EstimationService` (`src/domain/estimation/service.py`)** is the core domain logic:
1. `build_estimation_service()` picks a dataset via `QdrantConfig.dataset_name` (env `DATASET_NAME`), loads its `lf.pkl` (load-flow features, full read - needed in bulk to fit the scaler and populate Qdrant) from `{DATA_DIR}/{dataset_name}/interim/`, and reads `topology_cols.json` plus opens `tsa.db`/optionally `fsa.db` (`None` if the dataset has none) from `{DATA_DIR}/{dataset_name}/processed/` - `tsa`/`fsa` as `SqliteRecordStore`s (`src/services/sqlite_store.py`) rather than loading them into memory, so `EstimationService` only ever fetches the handful of rows matching a retrieved neighbor set, not the whole table, and the target datasets aren't duplicated in full across every worker process. It also fits a dataset-specific scaler (`make_scaler_bus39`, `make_scaler_eles`, `make_scaler_interscada_pl`, `make_scaler_interscada_fr`, each a `sklearn.compose.ColumnTransformer` keyed on column-name regexes, encoding domain unit ranges: voltages, phase angles via `AngleSinCos` in `src/preprocessing.py`, active/reactive power, short-circuit power), and populates a `DatabaseQdrant` (`src/services/qdrant/repository.py`) with the scaled feature vectors.
2. On a query, the incoming state is scaled with the same fitted scaler, then `DatabaseQdrant.query()` does a Euclidean nearest-neighbor search in Qdrant **filtered by exact topology match** (`topology_id`, a bitstring of the dataset's significant `oserv_*`/topology columns): neighbors must share the same grid topology as the query, not just be numerically close. `EstimationService._query_neighbors()` (retrieval) and `_weight_group()` (per-group distance/weight/compactness) are shared by every `estimate_*` method below.
3. **TSA**: retrieved neighbors are left-merged with the `tsa.db`-backed `SqliteRecordStore.fetch()` result (every retrieved state must have a TSA record - missing is an error), then grouped by `Crit_gen` (critical generator, an outcome of the simulation) for `estimate_by_generator`, and further by `Location` (the fault location) for `estimate_by_location`, with weights from an exponential kernel `K(distance)` in `src/domain/estimation/weights.py`. Per-group summaries in `src/domain/estimation/models.py` (`Report`/`LocationReport`) report a weighted CCT plus quality indicators: `n_eff` (effective sample size from the normalized weights, not the raw neighbor count) and `neighborhood_compactness` (mean pairwise kernel similarity within the group, not a calibrated density).
4. **FSA**: retrieved neighbors are *inner*-merged with the `fsa.db`-backed `SqliteRecordStore.fetch()` result (a retrieved state legitimately may have zero FSA coverage - silently excluded, not an error, since a large fraction of `(failed_gen, measured_gen)` pairs have no result at all). `_fsa_reports_by_pair()` computes one `FsaReport` (generic `metrics_weighted: dict[str, float]`, since the metric set differs per dataset - `minF`/`maxF`/`maxRoCoF` plus `M1`/`M2`/`M3` for `eles/2026-01` and `bus39`, just the first three for `eles/2026-06`) per `(failed_gen, measured_gen)` pair; `estimate_by_observed_generator()` and `estimate_by_failed_generator()` just re-nest that same flat result in the two useful orders, mirroring `estimate_by_generator`/`estimate_by_location`'s primary/secondary split. `bus39`'s FSA has neither generator dimension in named form (see `datasets/bus39/README.md`'s "FSA file" section): its source data dictionary withholds real generator identity for `failed_gen`, and its metrics are global worst-case system values rather than per-generator measurements - so `failed_gen` there is an anonymized `fsa_gen_N` label (not comparable to TSA's `Crit_gen` names) and `measured_gen` is always the constant `"system"`.
5. **SSSA** (`eles/2026-01`/`eles/2026-06` only): retrieved neighbors are *inner*-merged with the `sssa.db`-backed `SqliteRecordStore.fetch()` result. Deliberately **unweighted**, unlike TSA/FSA above - `estimate_sssa_by_generator()` groups by `generator` only, never by `mode_id` (per the data dictionary, oscillatory mode indices aren't comparable across operating states, so `mode_id` is treated like an opaque per-state run id, not a stable identifier), and returns every retrieved `(state, mode_id, generator)` row as-is (`SssaNeighbor`: `real_part`/`imag_part` plus a generic `metrics: dict[str, float]`), sorted by distance - no `metrics_weighted`, no `stats`. This is intentional: shown to domain partners in raw form pending their input on what aggregation (if any) is wanted. Each `SssaNeighbor` also carries `matched_mode` (`SssaModeMatch | None`): since `mode_id` isn't comparable across states, `EstimationService._match_sssa_modes()` finds every retrieved mode's single best cross-state counterpart by participation-vector cosine similarity with an eigenvalue-proximity tiebreak, computed fresh per query over just the retrieved neighbor set (not a stable cross-corpus identity) - no confidence threshold is applied, so callers judge match quality themselves from the returned `cosine_distance`/`eigenvalue_distance`. `sssa.db`/`sssa.pkl` are themselves the result of joining that dataset's separately-melted per-mode (`real_part`/`imag_part`) and per-generator-participation tables once at parse time (`datasets/eles/*/transform.py`), not per query. Exposed via `POST /api/v1/estimate/sssa/by-generator` (HTTP 501 for datasets without SSSA data, same pattern as the FSA routes); `StateRequest.max_states` is capped at 500 across all estimate endpoints since the mode-matching cost is roughly quadratic in retrieved rows.

**Multi-dataset design**: the same service code supports five datasets (`bus39`, `eles/2026-01`, `eles/2026-06`, `interscada/pl`, `interscada/fr`), switched purely via `DATASET_NAME`/`QdrantConfig.dataset_name`. Adding a dataset means: a `datasets/<name>/prepare.py` + `datasets/<name>/transform.py` producing `lf.pkl` / `topology_cols.joblib` (plus analyst-facing `tsa.pkl`, and `fsa.pkl`/`sssa.pkl` if applicable) under `datasets/<name>/interim/`, and the same `tsa`/`fsa`/`sssa` tables as `tsa.db`/`fsa.db`/`sssa.db` (via `scripts/_common.py::write_sqlite_table()`) plus `topology_cols.json` (via `write_json_list()`) under `datasets/<name>/processed/`, a registry entry in `_make_scaler_for_dataset()` in `src/domain/estimation/service.py` (pointing at an existing `make_scaler_*()` if the LF schema matches closely enough, e.g. `eles/2026-06` reuses `make_scaler_eles()` unchanged since its column regex prefixes are unchanged despite a different archive layout), and a `datasets/<name>/README.md` documenting the source format. `_dataset_paths()`/`_fsa_dataset_path()`/`_sssa_dataset_path()` resolve each artifact by basename (`tsa`, `fsa`, `sssa`, `topology_cols`) via a glob within `processed/`, so the on-disk extension (`.db`/`.json`) is an implementation detail, not part of the discovery contract. Qdrant collections are namespaced per dataset (`{QDRANT_COLLECTION_PREFIX}_{dataset_name}`, slashes replaced with `-`), so multiple datasets can share one Qdrant instance.

**Config**: all runtime config is env-driven `pydantic-settings` (`src/config/settings.py` for `DATA_DIR`/`LOG_LEVEL`, `src/services/qdrant/config.py` for `QDRANT_*`/`DATASET_NAME`). See `.env.example` for the full variable list including a population lock (`QDRANT_POPULATE_LOCK_*`) that serializes concurrent Qdrant collection population across API workers (the Docker image runs 16 uvicorn workers, each rebuilding `EstimationService` at startup).

**Logging**: `src/config/logging.py::configure_logging()` sets up timestamped console logging (`force=True`, since uvicorn/joblib workers may already have attached handlers) and is called at every real entrypoint - FastAPI startup, `scripts/prepare.py`, every `datasets/<name>/prepare.py`/`transform.py`, and the `scripts/evaluation/*.py` benchmarking tools - since each runs as its own OS process/subprocess and doesn't inherit another process's logging config. Verbosity defaults to `LOG_LEVEL` (`INFO` if unset); set `LOG_LEVEL=DEBUG` to see per-file ZIP extraction and subprocess command lines from `scripts/_common.py`.

**`src/benchmarking.py`** holds shared metric/fold helpers (`regression_metrics`, `group_k_fold_indices`, `group_k_fold_test_groups`) used by both the notebooks in `reports/` and the standalone scripts in `scripts/evaluation/`; grouping is always by the pre-fault **state** id to prevent leakage between folds, since many state-contingency records share one operating state and splitting them at random would let a model be scored on a state it had effectively already seen. Every caller passes the state column: `ml_benchmark.py:104` (`groups = records["state"]`, the supervised baselines), `benchmark.py:210`, `alpha_k_sweep.py:130` and `false_confidence_check.py:116` (all `tsa["state"]`). Note that `Crit_gen` must **not** be used for grouping - it is a simulation *outcome*, so grouping on it would neither prevent state-level leakage nor be available at query time.

**`reports/`** contains numbered Jupyter notebooks (`NN_description.ipynb`) that consume the `report-*.joblib` benchmark artifacts and the `reports/*_report.csv` / `risk_coverage_*.csv` outputs. These are analysis/writeup, not part of the service runtime. As of 2026-08-05, benchmark artifacts no longer live at the repo root: `.joblib`/`.parquet` files (raw/intermediate, not directly consumed by any paper) go to `tmp/` (gitignored), and CSV summaries backing the manuscript's tables/figures go to `results/data/` (tracked, via a `!results/data/*.csv` exception to the blanket `*.csv` ignore) - every `scripts/evaluation/*.py` path constant was updated accordingly, and the `PROJECT_ROOT / "report-*.joblib"` references in `reports/*.ipynb` were fixed to point at `tmp/` (files generated for a different paper's scope, e.g. SSSA or the eles/2026-01 candidate-topology investigation, land in `tmp/` regardless of extension since they're not manuscript evidence). If a notebook or script still resolves a bare filename at the repo root, that's now a bug, not the convention.

**`scripts/evaluation/`** holds the 25 scripts that measure the estimator: the benchmarks (`benchmark.py`, `eles_benchmark.py`, `ml_benchmark.py`, `fsa_benchmark.py`, `sssa_benchmark.py`), the ablations and significance tests (`eles_topology_ablation_significance.py`, `eles_topology_candidate_eval.py`), the diagnostics (`retrieval_quality.py`, `false_confidence_check.py`, the `*_diagnostics_selected.py` pair), the de-oracled bounds (`*_deoracled_bound.py`, `deployment_style_bound.py`) and `latency.py`. They write `.joblib`/`.parquet` artifacts to `tmp/`, which `scripts/paper/` and the `reports/` notebooks then consume - so the usual chain is `datasets/` to `scripts/evaluation/` to `tmp/` to `scripts/paper/` to `results/`. **This directory was named `scripts/service/` until 2026-08-24**; it was renamed because the name suggested it contained the service, which lives in `src/`, when in fact these scripts only evaluate it. Older log entries and notebooks naming `scripts/service/*.py` mean these same files.

**`scripts/paper/`** holds the 17 standalone analysis scripts that produce every table and figure in the Scientific Reports manuscript (`matched_extratrees_comparison.py`, `eles_causal_evaluation.py`, `eles_chronological_topology_support.py`, `screening_matched_workload.py`, the three `make_*_figure.py`, and so on). They read `datasets/` and `tmp/` and write `results/data/*.csv` plus `results/figures/*.tex`; the figure `.tex` files are then inlined into the manuscript. They lived in the manuscript's own private repository until 2026-08-24 and were moved here so that the Code Availability statement is accurate: the public repository now holds the full chain from raw dataset to reported number. Each resolves `PROJECT_DIR` as `Path(__file__).resolve().parents[2]`, so they must stay exactly two levels below the repository root.
