import argparse
import re
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
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths
from src.services.qdrant.config import get_qdrant_config

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]


def _configured_dataset_paths() -> tuple[Path, Path]:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir

    lf_path, tsa_path, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path, tsa_path


LF_PATH, TSA_PATH = _configured_dataset_paths()
REPORT_PATH: Final[Path] = PROJECT_DIR / "report-2026-06-19-interscada-pl.joblib"
GROUP_K_FOLD_REPORT_PATH: Final[Path] = PROJECT_DIR / "report-service-group-kfold-interscada-pl.joblib"


API_ENDPOINT: Final[str] = "http://localhost:8000/api/v1/estimate/by-generator"


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

        summary = pred.summary if pred is not None else None
        distances = summary.distances if summary is not None else {}
        cct_by_location = (
            {normalize_label(key): value for key, value in summary.cct_weighted_per_location.items()}
            if summary is not None
            else {}
        )
        location_weight_mass = (
            {normalize_label(key): value for key, value in summary.location_weight_mass.items()}
            if summary is not None
            else {}
        )
        location_counts = (
            {normalize_label(key): value for key, value in summary.location_counts.items()}
            if summary is not None
            else {}
        )

        cct_weighted_per_location = cct_by_location.get(location_true)
        report = {
            "state": state_id,
            "state_norm": state_id_norm,
            "cct_true": row["CCT"],
            "crit_gen_true": crit_gen_true,
            "location_true": location_true,
            "prediction_summary": summary,
            "cct_weighted_per_location": cct_weighted_per_location,
            "cct_weighted_global": summary.cct_weighted if summary is not None else None,
            "has_crit_gen_prediction": pred is not None,
            "has_location_prediction": cct_weighted_per_location is not None,
            "location_weight_mass": location_weight_mass.get(location_true),
            "location_neighbor_count": location_counts.get(location_true, 0),
            "n_neighbors": summary.n if summary is not None else 0,
            "n_eff": summary.n_eff if summary is not None else None,
            "neighborhood_compactness": (summary.neighborhood_compactness if summary is not None else None),
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
    # n_jobs = min(32, max(1, joblib.cpu_count() * 4))
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

    print(
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
            print(f"Skipped: {state_id=} has no samples")
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
                print(f"Skipped: {state_id=} has no samples")
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

    frame["cct_pred"] = frame["cct_weighted_per_location"].fillna(frame["cct_weighted_global"])
    rows: list[dict[str, float | str | int]] = []
    for fold, subset in frame.groupby("fold", sort=True):
        valid = subset["cct_pred"].notna()
        rows.append(
            {
                "fold": int(fold),
                "model": "service_location_then_global",
                **regression_metrics(
                    subset.loc[valid, "cct_true"].to_numpy(dtype=np.float64),
                    subset.loc[valid, "cct_pred"].to_numpy(dtype=np.float64),
                    coverage=float(valid.mean()),
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
    args = _parse_args()
    print(f"Benchmark dataset: lf={LF_PATH}, tsa={TSA_PATH}")
    lf = pd.read_pickle(LF_PATH)
    tsa = pd.read_pickle(TSA_PATH)

    if args.split == "leave-one-group-out":
        output = analyzer(lf=lf, tsa=tsa)
        report_path = REPORT_PATH
    else:
        predictions = analyze_group_k_fold(lf=lf, tsa=tsa, n_splits=args.n_splits)
        output = build_group_k_fold_payload(predictions, n_splits=args.n_splits)
        report_path = GROUP_K_FOLD_REPORT_PATH

    joblib.dump(output, report_path)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
