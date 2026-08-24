import argparse
import logging
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

load_dotenv()

import httpx
import joblib
import numpy as np
import pandas as pd
from joblib import delayed
from tqdm.auto import tqdm

from src.api.estimate import StateResponse
from src.benchmarking import group_k_fold_test_groups, regression_metrics, summarize_results
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib, .parquet) go to
# tmp/, CSV summaries the manuscript reports go to results/data/ (tracked).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _configured_dataset_paths() -> tuple[Path, Path]:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir

    lf_path, tsa_path, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path, tsa_path


LF_PATH, TSA_PATH = _configured_dataset_paths()
# Both paths used to be hardcoded to an "-interscada-pl" suffix regardless of DATASET_NAME - a
# copy-paste leftover from whichever dataset this was last pointed at. That silently wrote every
# BUS39 group-k-fold run to report-service-group-kfold-interscada-pl.joblib instead of the
# report-service-group-kfold.joblib every downstream script (bootstrap_risk_coverage.py,
# clearing_time_threshold_crossing.py, deployment_style_bound.py, false_confidence_check.py)
# actually reads - a 6h45m BUS39 run went to the wrong file before this was caught (2026-08-05).
# Fixed to be dataset-aware, but bus39 keeps the historical bare filename those scripts expect.
_dataset_name_safe = get_qdrant_config().dataset_name.strip().lower().replace("/", "-")
_dataset_suffix = "" if _dataset_name_safe == "bus39" else f"-{_dataset_name_safe}"
REPORT_PATH: Final[Path] = TMP_DIR / f"report-service-leave-one-group-out{_dataset_suffix}.joblib"
GROUP_K_FOLD_REPORT_PATH: Final[Path] = TMP_DIR / f"report-service-group-kfold{_dataset_suffix}.joblib"


API_ENDPOINT: Final[str] = "http://localhost:8000/api/v1/estimate/tsa/by-generator"


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    return text

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def _process_state(
    state_id: Any,
    state: pd.Series,
    tsa_subset: pd.DataFrame,
    *,
    exclude_uids: Iterable[Any] | None = None,
    fold: int | None = None,
) -> list[dict[str, Any]]:
    if tsa_subset.empty:
        raise ValueError("Empty `tsa_subset` input")

    state_id_norm = normalize_label(state_id)
    excluded_uids = frozenset(str(uid) for uid in (exclude_uids or [state_id]))
    excluded_uids_norm = frozenset(normalize_label(uid) for uid in excluded_uids)

    with httpx.Client(timeout=None, http2=True) as client:
        state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
        res = client.post(
            API_ENDPOINT,
            json={
                "variant": "1.0.0",
                "state": state_dict,
                "exclude_uids": sorted(excluded_uids),
            },
        )
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Estimate request failed for state={state_id_norm}: {res.status_code} {res.text[:500]}"
            ) from exc

    out = StateResponse.model_validate_json(res.text)
    outputs_by_crit_gen = {normalize_label(key): value for key, value in out.outputs.items()}

    reports: list[dict[str, Any]] = []
    for _, row in tsa_subset.iterrows():
        location_true = normalize_label(row["Location"])
        crit_gen_true = normalize_label(row["Crit_gen"])

        pred = outputs_by_crit_gen.get(crit_gen_true)
        included_uids_norm = (
            frozenset(normalize_label(uid) for uid in pred.included_state_ids) if pred is not None else frozenset()
        )
        leaked_uids = sorted(excluded_uids_norm & included_uids_norm)
        if leaked_uids:
            raise RuntimeError(f"Data leakage: excluded states `{leaked_uids}` found in `{pred.included_state_ids=}`")

        # Report is flat now (no group-level summary/stats blending every location together
        # - see Report's docstring in src/domain/estimation/models.py), so every diagnostic
        # below is sourced from the true location's own LocationReport within per_location.
        per_location = {normalize_label(loc): lr for loc, lr in pred.per_location.items()} if pred is not None else {}
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

        cct_weighted_per_location = (
            location_true_report.summary.cct_weighted if location_true_report is not None else None
        )
        report = {
            "state": state_id,
            "state_norm": state_id_norm,
            "cct_true": row["CCT"],
            "crit_gen_true": crit_gen_true,
            "location_true": location_true,
            # prediction_summary (the full nested API response - every location x generator x
            # neighbor) used to be stored here for deployment_style_bound.py, but that script
            # reads flat attributes (ps.location_weight_mass, ps.cct_weighted_per_location) that
            # predate the 2026-07-30 TSA report-model rework and no longer exist on the current
            # response schema - it's silently broken (getattr(..., None) returns None for every
            # record) and superseded by full_deoracled_bound.py's results anyway.
            # bootstrap_risk_coverage.py, the only other reader, already drops this column before
            # use. Retaining ~1M full nested objects in memory across a multi-hour run is what
            # killed this script (OOM-style SIGKILL at 91% after 6h45m, 2026-08-05) - removed.
            "cct_weighted_per_location": cct_weighted_per_location,
            "cct_weighted_global": (
                top_location_report.summary.cct_weighted if top_location_report is not None else None
            ),
            "has_crit_gen_prediction": pred is not None,
            "has_location_prediction": cct_weighted_per_location is not None,
            "location_weight_mass": (location_stats.weight_mass if location_stats is not None else None),
            "location_neighbor_count": (location_stats.n if location_stats is not None else 0),
            "n_neighbors": location_stats.n if location_stats is not None else 0,
            "n_eff": location_stats.n_eff if location_stats is not None else None,
            # n_eff is mechanically bounded above by n_neighbors (uniform weights give
            # n_eff == n_neighbors exactly) - reporting n_eff alone across groups with different
            # retrieved-pool sizes confounds "evidence concentration" with "how many candidates
            # were even available." n_eff_fraction divides that out.
            "n_eff_fraction": (
                location_stats.n_eff / location_stats.n if location_stats is not None and location_stats.n > 0 else None
            ),
            "neighborhood_compactness": (
                location_stats.neighborhood_compactness if location_stats is not None else None
            ),
            "n_unique_states": (location_stats.n_unique_states if location_stats is not None else None),
            "cct_weighted_std": (location_stats.cct_weighted_std if location_stats is not None else None),
            "cct_distance_correlation": (
                location_stats.cct_distance_correlation if location_stats is not None else None
            ),
            "distance_min": distances.get("min"),
            "distance_mean": distances.get("mean"),
            "distance_median": distances.get("median"),
            "distance_spread": distances.get("spread"),
            "distance_norm": distances.get("norm"),
        }
        if fold is not None:
            report["fold"] = fold
        reports.append(report)

    return reports


def _run_tasks(tasks: list[Any]) -> list[dict[str, Any]]:
    # Matches the API's --workers count - with the route handler calling EstimationService
    # synchronously inside an async def, a single worker serializes concurrent requests behind
    # its one event loop regardless of client-side concurrency, so client n_jobs only pays off
    # once the server actually has that many workers to spread requests across. Dropped from 16
    # to 8 (2026-08-06) after a memory-pressure incident running the full multi-script BUS39
    # suite concurrently at 16 each - each server worker holds its own full EstimationService
    # (fitted scaler + scaled lf copy + Qdrant client), so worker count directly multiplies
    # resident memory, not just CPU.
    n_jobs = 8
    reports: list[dict[str, Any]] = []
    n_missing_crit_gen = 0
    n_missing_location = 0

    jobs = joblib.Parallel(n_jobs=n_jobs, backend="threading", return_as="generator_unordered")(tasks)
    for chunk in tqdm(jobs, total=len(tasks), desc="Processing states"):
        if chunk is None:
            continue

        if not isinstance(chunk, Iterable):
            raise TypeError(f"Expected iterable result chunk, got {type(chunk)!r}")

        reports.extend(chunk)
        n_missing_crit_gen += sum(0 if row.get("has_crit_gen_prediction") else 1 for row in chunk)
        n_missing_location += sum(0 if row.get("has_location_prediction") else 1 for row in chunk)

    logger.info(
        "Coverage diagnostics: "
        f"total_rows={len(reports)}, "
        f"missing_crit_gen={n_missing_crit_gen}, "
        f"missing_location={n_missing_location}"
    )

    return reports


def analyzer(lf: pd.DataFrame, tsa: pd.DataFrame) -> list[dict[str, Any]]:
    tasks = []

    # Send each unique state to the service and exclude that state from retrieval.
    for state_id, state in lf.iterrows():
        tsa_subset = tsa[tsa.state == state_id].copy()
        if tsa_subset.empty:
            logger.warning(f"Skipped: {state_id=} has no samples")
            continue

        tasks.append(delayed(_process_state)(state_id=state_id, state=state, tsa_subset=tsa_subset))

    return _run_tasks(tasks)


def analyze_group_k_fold(
    lf: pd.DataFrame,
    tsa: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}
    test_group_folds = group_k_fold_test_groups(tsa["state"], n_splits=n_splits)
    tasks = []

    for fold, excluded_uids in enumerate(test_group_folds):
        for state_id, state in lf.iterrows():
            state_uid = str(state_id)
            if state_uid not in excluded_uids:
                continue

            tsa_subset = tsa_by_state.get(state_uid)
            if tsa_subset is None or tsa_subset.empty:
                logger.warning(f"Skipped: {state_id=} has no samples")
                continue

            tasks.append(
                delayed(_process_state)(
                    state_id=state_id,
                    state=state,
                    tsa_subset=tsa_subset,
                    exclude_uids=excluded_uids,
                    fold=fold,
                )
            )

    return _run_tasks(tasks)


def build_group_k_fold_payload(
    predictions: list[dict[str, Any]],
    *,
    n_splits: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(predictions)
    if frame.empty:
        raise ValueError("Cannot summarize an empty service benchmark report")

    # Two variants, reported separately rather than blended: "strict" scores only records
    # where the true location itself was retrieved (the honest coverage/error numbers, and
    # the ones a "fails visibly rather than silently extrapolating" claim must be based on);
    # "then_global" additionally falls back to the single most-likely location's CCT when the
    # true location has no coverage, kept for comparison but never as the reported headline.
    frame["cct_pred"] = frame["cct_weighted_per_location"].fillna(frame["cct_weighted_global"])
    rows: list[dict[str, float | str | int]] = []
    for fold, subset in frame.groupby("fold", sort=True):
        strict_valid = subset["cct_weighted_per_location"].notna()
        rows.append(
            {
                "fold": int(fold),
                "model": "service_location_strict",
                **regression_metrics(
                    subset.loc[strict_valid, "cct_true"].to_numpy(dtype=np.float64),
                    subset.loc[strict_valid, "cct_weighted_per_location"].to_numpy(dtype=np.float64),
                    coverage=float(strict_valid.mean()),
                ),
            }
        )

        fallback_valid = subset["cct_pred"].notna()
        rows.append(
            {
                "fold": int(fold),
                "model": "service_location_then_global",
                **regression_metrics(
                    subset.loc[fallback_valid, "cct_true"].to_numpy(dtype=np.float64),
                    subset.loc[fallback_valid, "cct_pred"].to_numpy(dtype=np.float64),
                    coverage=float(fallback_valid.mean()),
                ),
            }
        )

    results = pd.DataFrame(rows)
    return {
        "predictions": predictions,
        "results": results,
        "summary": summarize_results(results),
        "n_records": len(frame),
        "n_groups": frame["state_norm"].nunique(),
        "target": "CCT",
        "split": "GroupKFold by pre-fault state via API fold exclusion",
        "n_splits": n_splits,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("leave-one-group-out", "group-k-fold"),
        default="leave-one-group-out",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    logger.info(f"Benchmark dataset: lf={LF_PATH}, tsa={TSA_PATH}")
    lf = pd.read_pickle(LF_PATH)
    with sqlite3.connect(TSA_PATH) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)

    if args.split == "leave-one-group-out":
        output = analyzer(lf=lf, tsa=tsa)
        report_path = REPORT_PATH
    else:
        predictions = analyze_group_k_fold(lf=lf, tsa=tsa, n_splits=args.n_splits)
        output = build_group_k_fold_payload(predictions, n_splits=args.n_splits)
        report_path = GROUP_K_FOLD_REPORT_PATH

    joblib.dump(output, report_path)
    logger.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
