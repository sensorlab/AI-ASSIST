"""Leave-one-state-out FSA benchmark, run in-process against the real service.

The FSA analog of eles_benchmark.py. For each query state, estimate_by_observed_generator()
is asked for the frequency-stability outcome of every (failed_gen, measured_gen) pair, and
each prediction is scored against the recorded value for that pair.

Coverage means the share of recorded pairs that receive an estimate. Unlike TSA, a retrieved
state legitimately may carry no FSA record for a given pair (see CLAUDE.md: FSA is inner-
merged for exactly this reason), so an uncovered pair is a normal outcome rather than an
error, and coverage is the first thing worth reporting.

Metric columns differ by dataset and are discovered from the FSA table rather than hardcoded:
eles/2026-06 carries minF/maxF/maxRoCoF, bus39 adds M1/M2/M3. Errors are reported per metric
because the three quantities are on different scales and a pooled error would be meaningless.

Run from the repository root, e.g.:

    DATASET_NAME=eles/2026-06 QDRANT_URL=":memory:" \\
        uv run python scripts/evaluation/fsa_benchmark.py [n_states] [n_jobs]

n_states subsamples the query side (default 200, 0 means every state) because ELES holds
5,969,844 FSA rows over 4,393 states, a median of 1,260 pairs per state, so a full sweep is
roughly eighteen times the work of the ELES TSA benchmark. The sample is drawn with
FSA_BENCHMARK_SAMPLE_SEED (default 42) and the seed is recorded in the output filename
whenever it is not the default, so two seeds cannot overwrite each other.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from src.config.logging import configure_logging
from src.domain.estimation.service import build_estimation_service
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# Deliberately not results/data. FSA is out of scope for the Scientific Reports manuscript
# (CLAUDE.md: "Analysis: TSA (CCT) only"), so these outputs are project-report material and
# belong in tmp/ with the other non-paper artifacts, not among the paper's evidence CSVs.

SAMPLE_SEED: Final[int] = int(os.environ.get("FSA_BENCHMARK_SAMPLE_SEED", "42"))
# Identity columns in the FSA table; everything else is a measured metric.
FSA_KEY_COLUMNS: Final[frozenset[str]] = frozenset({"state", "failed_gen", "measured_gen"})


def _metric_columns(fsa: pd.DataFrame) -> list[str]:
    """Metric columns, discovered rather than hardcoded, since the set is dataset-specific."""
    return [c for c in fsa.columns if c not in FSA_KEY_COLUMNS and pd.api.types.is_numeric_dtype(fsa[c])]


def _process_state(
    service: Any,
    state_id: Any,
    state: pd.Series,
    fsa_subset: pd.DataFrame,
    metric_cols: list[str],
) -> list[dict[str, Any]]:
    """Score one query state's recorded FSA pairs against the service's estimates."""
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
    reports = service.estimate_by_observed_generator(state=state_dict, exclude_uids=[str(state_id)])

    rows: list[dict[str, Any]] = []
    for _, record in fsa_subset.iterrows():
        failed_gen = str(record["failed_gen"])
        measured_gen = str(record["measured_gen"])
        report = reports.get(measured_gen, {}).get(failed_gen)
        summary = report.summary if report is not None else None
        stats = summary.stats if summary is not None else None
        predicted = summary.metrics_weighted if summary is not None else {}

        row: dict[str, Any] = {
            "state": state_id,
            "failed_gen": failed_gen,
            "measured_gen": measured_gen,
            "covered": report is not None,
            "n_neighbors": stats.n if stats is not None else 0,
            "n_eff": stats.n_eff if stats is not None else None,
            "neighborhood_compactness": (stats.neighborhood_compactness if stats is not None else None),
            "n_unique_states": stats.n_unique_states if stats is not None else None,
            "distance_min": (stats.distances or {}).get("min") if stats is not None else None,
            "distance_mean": (stats.distances or {}).get("mean") if stats is not None else None,
        }
        for metric in metric_cols:
            true_value = record[metric]
            pred_value = predicted.get(metric)
            row[f"{metric}_true"] = None if pd.isna(true_value) else float(true_value)
            row[f"{metric}_pred"] = pred_value
        rows.append(row)
    return rows


def _summarize(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Per-metric error on the covered pairs, with the population each figure rests on."""
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        true_col, pred_col = f"{metric}_true", f"{metric}_pred"
        scored = df[df["covered"] & df[true_col].notna() & df[pred_col].notna()]
        if scored.empty:
            summary_rows.append({"metric": metric, "n_scored": 0})
            continue
        error = (scored[true_col] - scored[pred_col]).abs()
        summary_rows.append(
            {
                "metric": metric,
                "n_scored": int(len(scored)),
                "coverage": (
                    float(len(scored) / n_recorded) if (n_recorded := int(df[true_col].notna().sum())) else 0.0
                ),
                "mae": float(error.mean()),
                "rmse": float(np.sqrt(((scored[true_col] - scored[pred_col]) ** 2).mean())),
                "median_ae": float(error.median()),
                "ae_q95": float(error.quantile(0.95)),
                "ae_q99": float(error.quantile(0.99)),
                "max_ae": float(error.max()),
                "true_mean": float(scored[true_col].mean()),
                "true_std": float(scored[true_col].std()),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    dataset = config.dataset_name
    dataset_slug = dataset.replace("/", "-")

    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    fsa_path = PROJECT_DIR / "datasets" / dataset / "interim" / "fsa.pkl"
    lf_path = PROJECT_DIR / "datasets" / dataset / "interim" / "lf.pkl"
    if not fsa_path.is_file():
        raise SystemExit(f"{dataset} has no FSA data at {fsa_path}; nothing to benchmark.")

    lf = pd.read_pickle(lf_path)
    fsa = pd.read_pickle(fsa_path)
    metric_cols = _metric_columns(fsa)
    logger.info(f"Dataset {dataset}: {len(fsa):,} FSA rows, metrics {metric_cols}")

    if n_states and n_states < len(lf):
        rng = np.random.default_rng(SAMPLE_SEED)
        sampled = set(rng.choice(sorted(lf.index.astype(str)), size=n_states, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(sampled)]
        logger.info(f"Query-side subsample: {len(lf)} states (seed={SAMPLE_SEED})")
    else:
        logger.info(f"Full query side: {len(lf)} states")

    fsa["state"] = fsa["state"].astype(str)
    query_ids = set(lf.index.astype(str))
    fsa_by_state = {str(sid): subset for sid, subset in fsa[fsa["state"].isin(query_ids)].groupby("state")}

    logger.info("Building EstimationService (embedded Qdrant)...")
    t0 = time.monotonic()
    service = build_estimation_service()
    if service.fsa is None:
        raise SystemExit(f"{dataset} exposes no FSA store; the service would return HTTP 501.")
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        subset = fsa_by_state.get(str(state_id))
        if subset is None or subset.empty:
            continue
        rows.extend(_process_state(service, state_id, state, subset, metric_cols))
        if (i + 1) % 25 == 0:
            logger.info(f"{i + 1}/{len(lf)} states, {len(rows):,} pairs ({time.monotonic() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    elapsed = time.monotonic() - t0
    logger.info(f"Done: {len(df):,} pairs from {len(lf)} states in {elapsed:.1f}s")
    logger.info(f"Pair coverage: {df['covered'].mean():.4f} ({int(df['covered'].sum()):,} of {len(df):,})")

    suffix = f"-sample{n_states}" if n_states else "-full"
    if SAMPLE_SEED != 42:
        suffix += f"-seed{SAMPLE_SEED}"
    joblib.dump(rows, TMP_DIR / f"report-fsa-{dataset_slug}{suffix}.joblib")
    logger.info(f"Saved per-pair report to {TMP_DIR / f'report-fsa-{dataset_slug}{suffix}.joblib'}")

    summary = _summarize(df, metric_cols)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "n_states", len(lf))
    summary.insert(2, "n_pairs", len(df))
    summary.insert(3, "pair_coverage", float(df["covered"].mean()))
    out_csv = TMP_DIR / f"fsa_benchmark_{dataset_slug}{suffix}.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Saved summary to {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
