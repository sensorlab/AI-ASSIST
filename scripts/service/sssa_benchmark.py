"""SSSA service benchmark: retrieved eigenvalues against the query state's recorded ones.

estimate_sssa_by_generator() returns raw, unweighted neighbors - every retrieved
(state, mode_id, generator) row as-is, with no aggregated estimate (CLAUDE.md). So there is
no point prediction to score directly. What can be scored is the question a user actually
has: if this state had not been simulated, how close would retrieval have come to its modes?

The query state's own SSSA rows are the ground truth and are excluded from retrieval. Each
recorded mode is then paired with its best retrieved counterpart and the two eigenvalues are
compared: real_part is damping, imag_part is frequency in rad/s, and the derived damping
ratio and frequency in Hz are what an engineer reads.

Pairing cannot use mode_id, which is a state-local identifier explicitly not comparable
across states (datasets/eles/*/README.md, SSSA section). It uses the participation-vector
cosine similarity validated in scripts/service/eles_sssa_mode_similarity_eval.py, which found
a median eigenvalue distance of 0.064 for cosine-nearest cross-state pairs against a 2.22
random-pairing baseline. One vector per mode, one entry per generator, valued by that
generator's overall participation magnitude - the max across the dataset's ParMag_* columns -
and zero where the generator has no participation row for that mode, which the README records
as a real absence rather than missing data.

Every error is reported beside a random-pairing baseline computed on the same modes. Without
it a median damping error of some hundredths is uninterpretable: the baseline says what
picking an arbitrary mode would have cost, so the gap is the part retrieval earned.

Run from the repository root, e.g.:

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only QDRANT_URL=":memory:" \\
        uv run python scripts/service/sssa_benchmark.py [n_states]

bus39 and interscada/* carry no SSSA data; the service raises NotImplementedError and the API
returns HTTP 501, which this script reports rather than treating as a failure.
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
# Project-report material, not paper evidence: SSSA is out of scope for the manuscript
# (CLAUDE.md, "Analysis: TSA (CCT) only"), so outputs stay in tmp/.

SAMPLE_SEED: Final[int] = int(os.environ.get("SSSA_BENCHMARK_SAMPLE_SEED", "42"))
BASELINE_SEED: Final[int] = 0


def _parmag_columns(sssa: pd.DataFrame) -> list[str]:
    """Participation-magnitude columns, discovered per dataset: eles/2026-06 splits them per
    state variable (ParMag_speed, ParMag_phi, ...), eles/2026-01 has a single plain ParMag."""
    return [c for c in sssa.columns if c.startswith("ParMag")]


def _mode_vectors(rows: pd.DataFrame, generators: list[str], parmag_cols: list[str]) -> dict[Any, np.ndarray]:
    """One participation vector per mode_id, indexed by the dataset's full generator list."""
    index = {g: i for i, g in enumerate(generators)}
    vectors: dict[Any, np.ndarray] = {}
    magnitude = rows[parmag_cols].abs().max(axis=1)
    for (mode_id,), group in rows.assign(_mag=magnitude).groupby(["mode_id"]):
        vec = np.zeros(len(generators), dtype=float)
        for generator, mag in zip(group["generator"], group["_mag"], strict=False):
            position = index.get(str(generator))
            if position is not None:
                vec[position] = max(vec[position], float(mag))
        vectors[mode_id] = vec
    return vectors


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _damping_ratio(sigma: float, omega: float) -> float:
    magnitude = float(np.hypot(sigma, omega))
    return float(-sigma / magnitude) if magnitude else float("nan")


def _score_state(
    service: Any,
    state_id: Any,
    state: pd.Series,
    truth_rows: pd.DataFrame,
    generators: list[str],
    parmag_cols: list[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Pair each recorded mode of this state with its best retrieved counterpart and score."""
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
    t0 = time.perf_counter()
    reports = service.estimate_sssa_by_generator(state=state_dict, exclude_uids=[str(state_id)])
    query_ms = 1000 * (time.perf_counter() - t0)
    if not reports:
        return [{"state": state_id, "covered": False, "query_ms": query_ms}]

    # Rebuild retrieved modes from the flat neighbor rows. metrics carries the participation
    # columns, so the retrieved vectors are built exactly like the ground-truth ones.
    retrieved: dict[tuple[str, int], dict[str, Any]] = {}
    for generator, neighbors in reports.items():
        for neighbor in neighbors:
            key = (neighbor.state, neighbor.mode_id)
            entry = retrieved.setdefault(
                key,
                {"vec": np.zeros(len(generators)), "real": neighbor.real_part, "imag": neighbor.imag_part},
            )
            magnitudes = [abs(v) for k, v in neighbor.metrics.items() if k.startswith("ParMag")]
            position = {g: i for i, g in enumerate(generators)}.get(str(generator))
            if position is not None and magnitudes:
                entry["vec"][position] = max(entry["vec"][position], max(magnitudes))

    if not retrieved:
        return [{"state": state_id, "covered": False, "query_ms": query_ms}]

    keys = list(retrieved)
    matrix = np.vstack([retrieved[k]["vec"] for k in keys])
    truth_vectors = _mode_vectors(truth_rows, generators, parmag_cols)
    truth_eigen = truth_rows.groupby("mode_id")[["real_part", "imag_part"]].first()

    rows: list[dict[str, Any]] = []
    for mode_id, vector in truth_vectors.items():
        similarities = np.array([_cosine(vector, candidate) for candidate in matrix])
        best = int(np.argmax(similarities))
        chosen = retrieved[keys[best]]
        random_pick = retrieved[keys[int(rng.integers(len(keys)))]]

        true_sigma = float(truth_eigen.loc[mode_id, "real_part"])
        true_omega = float(truth_eigen.loc[mode_id, "imag_part"])
        rows.append(
            {
                "state": state_id,
                "covered": True,
                "mode_id": mode_id,
                "matched_state": keys[best][0],
                "cosine_similarity": float(similarities[best]),
                "real_true": true_sigma,
                "real_pred": float(chosen["real"]),
                "real_abs_err": abs(true_sigma - float(chosen["real"])),
                "imag_true": true_omega,
                "imag_pred": float(chosen["imag"]),
                "imag_abs_err": abs(true_omega - float(chosen["imag"])),
                "freq_hz_abs_err": abs(true_omega - float(chosen["imag"])) / (2 * np.pi),
                "damping_true": _damping_ratio(true_sigma, true_omega),
                "damping_pred": _damping_ratio(float(chosen["real"]), float(chosen["imag"])),
                "damping_abs_err": abs(
                    _damping_ratio(true_sigma, true_omega)
                    - _damping_ratio(float(chosen["real"]), float(chosen["imag"]))
                ),
                "eigen_distance": float(np.hypot(true_sigma - chosen["real"], true_omega - chosen["imag"])),
                # Same mode, an arbitrary retrieved counterpart: what matching bought.
                "baseline_eigen_distance": float(
                    np.hypot(true_sigma - random_pick["real"], true_omega - random_pick["imag"])
                ),
                "baseline_real_abs_err": abs(true_sigma - float(random_pick["real"])),
                "n_retrieved_modes": len(keys),
                "query_ms": query_ms,
            }
        )
    return rows


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    dataset = config.dataset_name
    dataset_slug = dataset.replace("/", "-")
    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    sssa_path = PROJECT_DIR / "datasets" / dataset / "interim" / "sssa.pkl"
    if not sssa_path.is_file():
        raise SystemExit(f"{dataset} has no SSSA data at {sssa_path}; the API returns HTTP 501 for it.")

    lf = pd.read_pickle(PROJECT_DIR / "datasets" / dataset / "interim" / "lf.pkl")
    sssa = pd.read_pickle(sssa_path)
    sssa["state"] = sssa["state"].astype(str)
    parmag_cols = _parmag_columns(sssa)
    generators = sorted(sssa["generator"].astype(str).unique())
    logger.info(f"{dataset}: {len(sssa):,} SSSA rows, {len(generators)} generators, ParMag cols {parmag_cols}")

    if n_states and n_states < len(lf):
        rng = np.random.default_rng(SAMPLE_SEED)
        sampled = set(rng.choice(sorted(lf.index.astype(str)), size=n_states, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(sampled)]
    logger.info(f"Query states: {len(lf)} (seed={SAMPLE_SEED})")

    truth_by_state = {str(sid): subset for sid, subset in sssa.groupby("state")}

    logger.info("Building EstimationService (embedded Qdrant)...")
    t0 = time.monotonic()
    service = build_estimation_service()
    if getattr(service, "sssa", None) is None:
        raise SystemExit(f"{dataset} exposes no SSSA store; the API returns HTTP 501.")
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    baseline_rng = np.random.default_rng(BASELINE_SEED)
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        truth = truth_by_state.get(str(state_id))
        if truth is None or truth.empty:
            continue
        rows.extend(_score_state(service, state_id, state, truth, generators, parmag_cols, baseline_rng))
        if (i + 1) % 10 == 0:
            logger.info(f"{i + 1}/{len(lf)} states ({time.monotonic() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    logger.info(f"Done: {len(df):,} rows from {len(lf)} states in {time.monotonic() - t0:.1f}s")

    suffix = f"-sample{n_states}" if n_states else "-full"
    if SAMPLE_SEED != 42:
        suffix += f"-seed{SAMPLE_SEED}"
    joblib.dump(rows, TMP_DIR / f"report-sssa-{dataset_slug}{suffix}.joblib")

    scored = df[df["covered"] & df.get("mode_id", pd.Series(dtype=object)).notna()]
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_states": int(df["state"].nunique()),
                "state_coverage": float(df.groupby("state")["covered"].max().mean()),
                "n_modes_scored": int(len(scored)),
                "cosine_similarity_median": float(scored["cosine_similarity"].median()),
                "damping_mae": float(scored["damping_abs_err"].mean()),
                "damping_median_ae": float(scored["damping_abs_err"].median()),
                "real_mae": float(scored["real_abs_err"].mean()),
                "real_median_ae": float(scored["real_abs_err"].median()),
                "imag_mae_rad_s": float(scored["imag_abs_err"].mean()),
                "freq_mae_hz": float(scored["freq_hz_abs_err"].mean()),
                "eigen_distance_median": float(scored["eigen_distance"].median()),
                "baseline_eigen_distance_median": float(scored["baseline_eigen_distance"].median()),
                "baseline_real_median_ae": float(scored["baseline_real_abs_err"].median()),
                "query_ms_median": float(df["query_ms"].median()),
            }
        ]
    )
    out_csv = TMP_DIR / f"sssa_benchmark_{dataset_slug}{suffix}.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Saved summary to {out_csv}")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
