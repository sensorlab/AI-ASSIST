import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import httpx
import joblib
import pandas as pd
from joblib import delayed
from tqdm.auto import tqdm

from src.api.estimate import StateResponse

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]


LF_PATH = Path(PROJECT_DIR / "data/eles/interim/lf.pkl")
TSA_PATH = Path(PROJECT_DIR / "data/bus39/interim/tsa.pkl")
REPORT_PATH: Final[Path] = PROJECT_DIR / "report-2026-05-14.joblib"


API_ENDPOINT: Final[str] = "http://localhost:8000/api/v1/estimate"


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def _process_state(state_id: Any, state: pd.Series, tsa_subset: pd.DataFrame) -> list[dict[str, Any]]:
    if tsa_subset.empty:
        raise ValueError("Empty `tsa_subset` input")

    state_id_norm = normalize_label(state_id)

    with httpx.Client(timeout=None, http2=True) as client:
        res = client.post(
            API_ENDPOINT,
            json={
                "variant": "1.0.0",
                "state": state.to_dict(),
                "exclude_uids": [state_id_norm],
            },
        )
        res.raise_for_status()

    out = StateResponse.model_validate_json(res.text)
    outputs_by_crit_gen = {normalize_label(key): value for key, value in out.outputs.items()}

    reports: list[dict[str, Any]] = []
    for _, row in tsa_subset.iterrows():
        location_true = normalize_label(row["Location"])
        crit_gen_true = normalize_label(row["Crit_gen"])

        pred = outputs_by_crit_gen.get(crit_gen_true)

        summary = pred.summary if pred is not None else None
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

        reports.append(
            {
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
            }
        )

    return reports


def analyzer(lf: pd.DataFrame, tsa: pd.DataFrame):
    n_jobs = min(32, max(1, joblib.cpu_count() * 4))

    tasks = []

    # send unique state to service to receive similar states
    for state_id, state in lf.iterrows():
        tsa_subset = tsa[tsa.state == state_id].copy()
        if tsa_subset.empty:
            print(f"Skipped: {state_id=} has no samples")
            continue

        task = delayed(_process_state)(state_id=state_id, state=state, tsa_subset=tsa_subset)
        tasks.append(task)

    reports = []
    n_missing_crit_gen = 0
    n_missing_location = 0

    results = joblib.Parallel(n_jobs=n_jobs, backend="threading", return_as="generator_unordered")(tasks)
    for chunk in tqdm(results, total=len(tasks), desc="Processing states"):
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


def main():
    lf = pd.read_pickle(LF_PATH)
    tsa = pd.read_pickle(TSA_PATH)

    output = analyzer(lf=lf, tsa=tsa)
    joblib.dump(output, REPORT_PATH)


if __name__ == "__main__":
    main()
