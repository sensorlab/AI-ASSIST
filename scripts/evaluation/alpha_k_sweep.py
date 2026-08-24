"""Coarse sensitivity sweep over K (n_neighbors) and alpha (kernel decay) for AI-ASSIST.

Run in-process against an embedded (`:memory:`) Qdrant instance - no live API server or
external Qdrant needed. Restricted to a subset of GroupKFold folds to keep runtime bounded;
this is a sensitivity check, not a full benchmark replacement.

Run from repository root:
    DATASET_NAME=bus39 QDRANT_URL=":memory:" uv run python scripts/evaluation/alpha_k_sweep.py
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("DATASET_NAME", "bus39")
os.environ.setdefault("QDRANT_URL", ":memory:")

import joblib
import numpy as np
import pandas as pd

from scripts.evaluation.benchmark import normalize_label
from src.benchmarking import group_k_fold_test_groups, regression_metrics
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import EstimationService, _dataset_paths, build_estimation_service
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES: Final[tuple[int, ...]] = (25, 50, 100, 200)
ALPHA_VALUES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
N_SPLITS: Final[int] = 5
FOLDS_TO_EVALUATE: Final[tuple[int, ...]] = (0,)
# Embedded (:memory:) Qdrant uses brute-force local-mode search (~1.4s/query on 21,783
# points), so a full fold (~4,357 states) per (K, alpha) point is not tractable in-session.
# Evaluate a fixed random subsample of held-out states per fold instead - a coarse sweep,
# not a full-fold benchmark; this is disclosed explicitly wherever the results are reported.
STATES_PER_FOLD: Final[int] = 300
# Overridable via ALPHA_K_SWEEP_SAMPLE_SEED for multi-seed robustness checks (repeating the
# sweep across several draws to attach an uncertainty band to the reported MAE/RMSE spread);
# default 42 matches the seed used for the paper's originally reported sweep.
SAMPLE_SEED: Final[int] = int(os.environ.get("ALPHA_K_SWEEP_SAMPLE_SEED", "42"))
_SEED_SUFFIX = "" if SAMPLE_SEED == 42 else f"-seed{SAMPLE_SEED}"
REPORT_PATH: Final[Path] = TMP_DIR / f"report-alpha-k-sweep{_SEED_SUFFIX}.joblib"
CSV_PATH: Final[Path] = PAPER_DATA_DIR / f"alpha_k_sweep{_SEED_SUFFIX}.csv"


def _configured_dataset_paths() -> tuple[Path, Path]:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    lf_path, tsa_path, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path, tsa_path


def _evaluate_point(
    service: EstimationService,
    lf: pd.DataFrame,
    tsa_by_state: dict[str, pd.DataFrame],
    query_state_ids: frozenset[str],
    *,
    fold_exclusion: frozenset[str],
    n_neighbors: int,
    alpha: float,
) -> pd.DataFrame:
    """Run one (K, alpha) point over every state in `query_state_ids` (the sampled subset
    actually queried), matching scripts/evaluation/benchmark.py::_process_state's
    crit_gen/location matching logic, but in-process (no HTTP) and with configurable K/alpha.
    `fold_exclusion` is the FULL held-out fold (not just the sample) and is what's excluded
    from retrieval for every query, to preserve correct GroupKFold leakage semantics
    regardless of which states within the fold happen to be sampled as queries."""
    rows: list[dict[str, Any]] = []

    for state_id, state in lf.iterrows():
        state_uid = str(state_id)
        if state_uid not in query_state_ids:
            continue

        tsa_subset = tsa_by_state.get(state_uid)
        if tsa_subset is None or tsa_subset.empty:
            continue

        state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
        reports = service.estimate_by_generator(
            state=state_dict,
            exclude_uids=fold_exclusion,
            n_neighbors=n_neighbors,
            alpha=alpha,
        )
        outputs_by_crit_gen = {normalize_label(key): value for key, value in reports.items()}

        for _, row in tsa_subset.iterrows():
            location_true = normalize_label(row["Location"])
            crit_gen_true = normalize_label(row["Crit_gen"])

            pred = outputs_by_crit_gen.get(crit_gen_true)
            # Report is flat now (no group-level summary/stats blending every location
            # together - see Report's docstring in src/domain/estimation/models.py); fall
            # back to the single most-likely location's CCT (first entry of
            # location_likelihood, sorted descending) when the true location isn't covered,
            # instead of a blended-across-locations aggregate.
            per_location = {normalize_label(k): v for k, v in pred.per_location.items()} if pred is not None else {}
            cct_pred = per_location[location_true].summary.cct_weighted if location_true in per_location else None
            if cct_pred is None and pred is not None and pred.location_likelihood:
                top_location = next(iter(pred.location_likelihood))
                cct_pred = pred.per_location[top_location].summary.cct_weighted

            rows.append({"cct_true": row["CCT"], "cct_pred": cct_pred})

    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    lf_path, tsa_path = _configured_dataset_paths()
    logger.info(f"Sweep dataset: lf={lf_path}, tsa={tsa_path}")

    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}

    test_group_folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)

    logger.info("Building in-process EstimationService with embedded (:memory:) Qdrant...")
    t0 = time.monotonic()
    service = build_estimation_service()
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    rng = np.random.default_rng(SAMPLE_SEED)
    sweep_rows: list[dict[str, Any]] = []
    for fold in FOLDS_TO_EVALUATE:
        fold_state_ids = test_group_folds[fold]
        sample_size = min(STATES_PER_FOLD, len(fold_state_ids))
        query_state_ids = frozenset(rng.choice(sorted(fold_state_ids), size=sample_size, replace=False))
        logger.info(
            f"Fold {fold}: {len(fold_state_ids):,} held-out states, "
            f"sampling {sample_size} as queries (seed={SAMPLE_SEED})"
        )

        for n_neighbors in K_VALUES:
            for alpha in ALPHA_VALUES:
                t0 = time.monotonic()
                frame = _evaluate_point(
                    service,
                    lf,
                    tsa_by_state,
                    query_state_ids,
                    fold_exclusion=fold_state_ids,
                    n_neighbors=n_neighbors,
                    alpha=alpha,
                )
                elapsed = time.monotonic() - t0

                valid = frame["cct_pred"].notna()
                metrics = regression_metrics(
                    frame.loc[valid, "cct_true"].to_numpy(dtype=np.float64),
                    frame.loc[valid, "cct_pred"].to_numpy(dtype=np.float64),
                    coverage=float(valid.mean()),
                )
                sweep_rows.append({"fold": fold, "K": n_neighbors, "alpha": alpha, **metrics})
                logger.info(
                    f"fold={fold} K={n_neighbors} alpha={alpha}: "
                    f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
                    f"coverage={metrics['coverage']:.4f} ({elapsed:.1f}s, n={len(frame)})"
                )

    results = pd.DataFrame(sweep_rows)
    print("\nSweep results:")
    print(results[["fold", "K", "alpha", "coverage", "mae", "rmse"]].to_string(index=False))

    joblib.dump({"results": results, "k_values": K_VALUES, "alpha_values": ALPHA_VALUES}, REPORT_PATH)
    results.to_csv(CSV_PATH, index=False)
    logger.info(f"Saved report to {REPORT_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
