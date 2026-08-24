"""Checks whether a retrieved group can look well-supported by record-level diagnostics
(effective sample size, neighborhood compactness) while resting on very few distinct
pre-fault states - the false-confidence failure mode raised in paper.tex's Limitations
(record-level aggregation, Section 5.3, interacting with record-level diagnostics computed
in Section 4.5).

Uses the new n_unique_states diagnostic (src/domain/estimation/models.py::Stats,
src/domain/estimation/service.py) alongside the existing n_eff/neighborhood_compactness on
a coarse BUS39 subsample, following the same subsampling convention as
scripts/evaluation/alpha_k_sweep.py (300 of one GroupKFold fold's held-out states, seed 42) -
a full leave-one-group-out pass is not needed to characterize this relationship.

Run from repository root:
    DATASET_NAME=bus39 QDRANT_URL=":memory:" uv run python scripts/evaluation/false_confidence_check.py
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

import numpy as np
import pandas as pd

from scripts.evaluation.benchmark import normalize_label
from src.benchmarking import group_k_fold_test_groups
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import EstimationService, _dataset_paths, build_estimation_service
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# CSV summaries the paper actually consumes go to paper-sr/data/, not the repo root
# (2026-08-05 cleanup).
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "false_confidence_check.csv"

N_SPLITS: Final[int] = 5
FOLD_TO_EVALUATE: Final[int] = 0
STATES_PER_FOLD: Final[int] = 300
SAMPLE_SEED: Final[int] = 42

# A group is flagged as a false-confidence candidate if it looks well-supported by a
# record-level diagnostic (top quartile n_eff or compactness) while resting on very few
# distinct pre-fault states.
LOW_DIVERSITY_THRESHOLD: Final[int] = 3


def _configured_dataset_paths() -> tuple[Path, Path]:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    lf_path, tsa_path, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path, tsa_path


def _evaluate(
    service: EstimationService,
    lf: pd.DataFrame,
    tsa_by_state: dict[str, pd.DataFrame],
    query_state_ids: frozenset[str],
    *,
    fold_exclusion: frozenset[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for state_id, state in lf.iterrows():
        state_uid = str(state_id)
        if state_uid not in query_state_ids:
            continue

        tsa_subset = tsa_by_state.get(state_uid)
        if tsa_subset is None or tsa_subset.empty:
            continue

        state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
        reports = service.estimate_by_generator(state=state_dict, exclude_uids=fold_exclusion)
        outputs_by_crit_gen = {normalize_label(key): value for key, value in reports.items()}

        for _, row in tsa_subset.iterrows():
            crit_gen_true = normalize_label(row["Crit_gen"])
            pred = outputs_by_crit_gen.get(crit_gen_true)
            if pred is None:
                continue
            # Report (by-generator) deliberately has no group-level stats since the
            # 2026-07-30 rework - a blended stat across locations was misleading. Use the
            # top-likelihood location's own stats instead, the same "pick one concrete
            # answer" pattern already used in benchmark.py/generator_deoracled_bound.py.
            if not pred.location_likelihood:
                continue
            top_location = next(iter(pred.location_likelihood))
            location_report = pred.per_location.get(top_location)
            if location_report is None:
                continue
            stats = location_report.summary.stats
            rows.append(
                {
                    "state": state_uid,
                    "crit_gen": crit_gen_true,
                    "n": stats.n,
                    "n_eff": stats.n_eff,
                    "n_unique_states": stats.n_unique_states,
                    "neighborhood_compactness": stats.neighborhood_compactness,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    lf_path, tsa_path = _configured_dataset_paths()
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}")

    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}

    test_group_folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)
    fold_state_ids = test_group_folds[FOLD_TO_EVALUATE]

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_size = min(STATES_PER_FOLD, len(fold_state_ids))
    query_state_ids = frozenset(rng.choice(sorted(fold_state_ids), size=sample_size, replace=False))
    logger.info(f"Fold {FOLD_TO_EVALUATE}: sampling {sample_size} of {len(fold_state_ids):,} held-out states")

    logger.info("Building in-process EstimationService with embedded (:memory:) Qdrant...")
    t0 = time.monotonic()
    service = build_estimation_service()
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    df = _evaluate(service, lf, tsa_by_state, query_state_ids, fold_exclusion=fold_state_ids)
    logger.info(f"Evaluated {len(df):,} (state, crit_gen) group reports in {time.monotonic() - t0:.1f}s")

    df = df.dropna(subset=["neighborhood_compactness"]).copy()

    neff_q75 = df["n_eff"].quantile(0.75)
    compact_q75 = df["neighborhood_compactness"].quantile(0.75)
    low_diversity = df["n_unique_states"] <= LOW_DIVERSITY_THRESHOLD

    high_neff_low_diversity = ((df["n_eff"] >= neff_q75) & low_diversity).mean()
    high_compact_low_diversity = ((df["neighborhood_compactness"] >= compact_q75) & low_diversity).mean()

    corr_neff = df["n_eff"].corr(df["n_unique_states"])
    corr_compact = df["neighborhood_compactness"].corr(df["n_unique_states"])

    logger.info(f"n_eff vs n_unique_states correlation: {corr_neff:.3f}")
    logger.info(f"neighborhood_compactness vs n_unique_states correlation: {corr_compact:.3f}")
    logger.info(
        f"Fraction of groups with top-quartile n_eff (>= {neff_q75:.1f}) AND "
        f"n_unique_states <= {LOW_DIVERSITY_THRESHOLD}: {high_neff_low_diversity:.4%}"
    )
    logger.info(
        f"Fraction of groups with top-quartile compactness (>= {compact_q75:.3f}) AND "
        f"n_unique_states <= {LOW_DIVERSITY_THRESHOLD}: {high_compact_low_diversity:.4%}"
    )
    logger.info(f"Median n_unique_states: {df['n_unique_states'].median():.1f} (n={len(df):,} groups)")

    out = pd.DataFrame(
        [
            {"quantity": "n_groups", "value": len(df)},
            {"quantity": "corr_n_eff_vs_n_unique_states", "value": corr_neff},
            {"quantity": "corr_compactness_vs_n_unique_states", "value": corr_compact},
            {"quantity": "neff_q75", "value": neff_q75},
            {"quantity": "compactness_q75", "value": compact_q75},
            {"quantity": "frac_top_quartile_neff_and_low_diversity", "value": high_neff_low_diversity},
            {"quantity": "frac_top_quartile_compactness_and_low_diversity", "value": high_compact_low_diversity},
            {"quantity": "median_n_unique_states", "value": df["n_unique_states"].median()},
        ]
    )
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
