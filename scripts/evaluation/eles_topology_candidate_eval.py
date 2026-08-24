"""Evaluate a candidate topology definition for eles/2026-01 against the one currently in
production (datasets/eles/2026-01/processed/topology_cols.json).

Investigation trigger: the production topology_cols.json (254 columns, ELES's own
PowerFactory "Lines"+"Loads" dictionary match) turns out to be constant across every
record in this dataset - it collapses the entire dataset into a single topology_id, which
means exact-topology retrieval is currently a no-op for eles/2026-01, not a genuine filter.
Real topology variation lives in the ~512 line columns the dictionary excludes (mostly
foreign/neighboring-grid elements by naming) plus the ~123 generator columns (already
known to be too noisy per the dataset README).

Candidate definition tested here: ALL oserv_Lne* columns (Slovenian + foreign, i.e. do NOT
restrict to the dictionary-matched subset), EXCLUDING oserv_Gen* (generators) - this gave a
much healthier group-size distribution in exploratory analysis (1,790 groups, 76% of
records have >=1 same-topology neighbor, max group 38) versus either extreme (254-col
dictionary subset: 1 group; full 1072-col set: ~95% singleton).

This script does NOT modify datasets/eles/2026-01/processed/topology_cols.json - it builds
its own EstimationService variant in-process with the candidate columns, evaluates
leave-one-group-out (mirroring scripts/evaluation/benchmark.py::analyzer/_process_state, but
via direct Python calls against an embedded (:memory:) Qdrant instead of HTTP), and writes
a separate report + risk-coverage CSV for side-by-side comparison against the existing
production-topology report (report-2026-06-10-eles.joblib / data/risk_coverage_eles2026-01.csv).

Run from repository root:
    uv run python scripts/evaluation/eles_topology_candidate_eval.py
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Final, cast

os.environ.setdefault("DATASET_NAME", "eles/2026-01")
os.environ.setdefault("QDRANT_URL", ":memory:")

import joblib
import numpy as np
import pandas as pd

from scripts.evaluation.benchmark import normalize_label
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import (
    EstimationService,
    _dataset_paths,
    _make_scaler_for_dataset,
)
from src.services.qdrant.client import create_qdrant_client
from src.services.qdrant.config import get_qdrant_config
from src.services.qdrant.repository import DatabaseQdrant
from src.services.sqlite_store import SqliteRecordStore

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root; this script is superseded by
# eles_benchmark.py for eles/2026-06 (see its own docstring) and eles/2026-01 is out of
# paper-sr's scope, so both outputs go to tmp/ rather than paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH: Final[Path] = TMP_DIR / "report-eles2026-01-candidate-topology.joblib"
RISK_COVERAGE_PATH: Final[Path] = TMP_DIR / "risk_coverage_eles2026-01-candidate-topology.csv"

COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)


def _build_candidate_service() -> EstimationService:
    """Mirrors build_estimation_service() (src/domain/estimation/service.py) exactly,
    except the topology column set is the candidate (all Lne, no Gen) rather than
    whatever's in topology_cols.json - no production file is read or written."""
    config = get_qdrant_config()
    app_settings = get_app_settings()
    path_lf_dataset, path_tsa_dataset, _ = _dataset_paths(app_settings.data_dir, config.dataset_name)

    lf = pd.read_pickle(path_lf_dataset)
    tsa = SqliteRecordStore(path_tsa_dataset, table="tsa")

    scaler = _make_scaler_for_dataset(config.dataset_name)
    lf_scaled = cast(pd.DataFrame, scaler.fit_transform(lf))

    candidate_raw_cols = [c for c in lf.columns if c.lower().startswith("oserv_lne")]
    logger.info(f"Candidate topology definition: {len(candidate_raw_cols)} oserv_Lne* columns (no generators)")

    feature_map: dict[str, str] = {}
    for col in candidate_raw_cols:
        candidates = [c for c in lf_scaled.columns if c == col or c.startswith(col + "_")]
        if len(candidates) != 1:
            raise ValueError(f"cannot map topology col {col!r}: found {len(candidates)} candidates {candidates[:5]}")
        feature_map[col] = candidates[0]

    client = create_qdrant_client(config)
    db = DatabaseQdrant(
        client=client,
        collection_name=f"{config.collection_name}-candidate-topology",
        subset_topology_cols=feature_map.values(),
        populate_lock_path=config.populate_lock_path,
        populate_lock_timeout_seconds=config.populate_lock_timeout_seconds,
        use_population_lock=False,
    )
    db.fit(lf_scaled, force=False)

    return EstimationService(columns=list(lf.columns), scaler=scaler, tsa=tsa, db=db, fsa=None, sssa=None)


def _process_state(
    service: EstimationService,
    state_id: Any,
    state: pd.Series,
    tsa_subset: pd.DataFrame,
) -> list[dict[str, Any]]:
    """In-process equivalent of scripts/evaluation/benchmark.py::_process_state (leave-one-
    group-out variant: exclude only the query state itself, matching analyzer())."""
    state_id_norm = normalize_label(state_id)
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}

    reports = service.estimate_by_generator(state=state_dict, exclude_uids=[str(state_id)])
    outputs_by_crit_gen = {normalize_label(key): value for key, value in reports.items()}

    rows: list[dict[str, Any]] = []
    for _, row in tsa_subset.iterrows():
        location_true = normalize_label(row["Location"])
        crit_gen_true = normalize_label(row["Crit_gen"])

        pred = outputs_by_crit_gen.get(crit_gen_true)
        # Report is flat now (no group-level summary/stats blending every location together -
        # see Report's docstring in src/domain/estimation/models.py), so every diagnostic
        # below is sourced from the true location's own LocationReport within per_location.
        per_location = {normalize_label(k): v for k, v in pred.per_location.items()} if pred is not None else {}
        location_true_report = per_location.get(location_true)
        location_stats = location_true_report.summary.stats if location_true_report is not None else None
        distances = location_stats.distances if location_stats is not None else {}

        # No more group-level aggregate CCT to fall back on when the true location isn't
        # covered - fall back to the single most-likely location's CCT instead (the first
        # entry of location_likelihood, which is sorted descending by score).
        top_location_report = None
        if pred is not None and pred.location_likelihood:
            top_location = next(iter(pred.location_likelihood))
            top_location_report = pred.per_location.get(top_location)

        rows.append(
            {
                "state": state_id,
                "state_norm": state_id_norm,
                "cct_true": row["CCT"],
                "crit_gen_true": crit_gen_true,
                "location_true": location_true,
                "cct_weighted_per_location": (
                    location_true_report.summary.cct_weighted if location_true_report is not None else None
                ),
                "cct_weighted_global": (
                    top_location_report.summary.cct_weighted if top_location_report is not None else None
                ),
                "location_weight_mass": (location_stats.weight_mass if location_stats is not None else None),
                "n_neighbors": location_stats.n if location_stats is not None else 0,
                "n_eff": location_stats.n_eff if location_stats is not None else None,
                "neighborhood_compactness": (
                    location_stats.neighborhood_compactness if location_stats is not None else None
                ),
                "cct_weighted_std": (location_stats.cct_weighted_std if location_stats is not None else None),
                "distance_min": distances.get("min"),
                "distance_mean": distances.get("mean"),
                "distance_median": distances.get("median"),
                "distance_spread": distances.get("spread"),
                "distance_norm": distances.get("norm"),
            }
        )
    return rows


def _risk_coverage(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Reproduces reports/30_benchmark_results_analysis.ipynb::risk_coverage() exactly."""
    x = df.dropna(subset=[metric, "err"]).copy()
    x = x.sort_values(metric, ascending=not higher_is_better)

    rows = []
    n = len(x)
    for cov in COVERAGES:
        k = math.ceil(cov * n)
        kept = x.iloc[:k]
        rows.append(
            {
                "metric": metric,
                "coverage": cov,
                "n": k,
                "mae": kept["err"].mean(),
                "rmse": np.sqrt((kept["err"] ** 2).mean()),
                "q90": kept["err"].quantile(0.90),
                "q95": kept["err"].quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    app_settings = get_app_settings()
    lf_path, tsa_path, _ = _dataset_paths(app_settings.data_dir, config.dataset_name)
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}")

    lf = pd.read_pickle(lf_path)
    tsa_store = SqliteRecordStore(tsa_path, table="tsa")
    tsa = tsa_store.fetch(list(lf.index.astype(str)))
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}

    logger.info("Building candidate-topology EstimationService (embedded Qdrant)...")
    t0 = time.monotonic()
    service = _build_candidate_service()
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    all_rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        tsa_subset = tsa_by_state.get(str(state_id))
        if tsa_subset is None or tsa_subset.empty:
            continue
        all_rows.extend(_process_state(service, state_id, state, tsa_subset))
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(lf)} states ({time.monotonic() - t0:.1f}s elapsed)")

    logger.info(f"Done: {len(all_rows)} rows from {len(lf)} states in {time.monotonic() - t0:.1f}s")

    df = pd.DataFrame(all_rows)
    coverage_has_location = df["cct_weighted_per_location"].notna().mean()
    coverage_has_global = df["cct_weighted_global"].notna().mean()
    logger.info(
        f"Coverage: has_location_prediction={coverage_has_location:.4f}, has_any_prediction={coverage_has_global:.4f}"
    )

    joblib.dump(all_rows, REPORT_PATH)
    logger.info(f"Saved raw per-record report to {REPORT_PATH}")

    _df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    _df["err"] = (_df["cct_true"] - _df["cct_weighted_per_location"]).abs()

    metrics = {
        "location_weight_mass": True,
        "n_eff": True,
        "n_neighbors": True,
        "neighborhood_compactness": True,
        # Lower cct_weighted_std means the neighbors agree more on the outcome - i.e. more
        # confidence, same direction as the distance metrics below, not the weight/n_eff
        # ones above.
        "cct_weighted_std": False,
        "distance_min": False,
        "distance_mean": False,
        "distance_median": False,
        "distance_spread": False,
        "distance_norm": False,
    }
    out = [_risk_coverage(_df, metric, higher) for metric, higher in metrics.items()]
    rc = pd.concat(out, ignore_index=True)
    rc.to_csv(RISK_COVERAGE_PATH, index=False)
    logger.info(f"Saved risk-coverage CSV to {RISK_COVERAGE_PATH}")
    print(rc.to_string(index=False))


if __name__ == "__main__":
    main()
