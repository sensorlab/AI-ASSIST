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
cosine similarity validated in scripts/evaluation/eles_sssa_mode_similarity_eval.py, which found
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
        uv run python scripts/evaluation/sssa_benchmark.py [n_states]

bus39 and interscada/* carry no SSSA data; the service raises NotImplementedError and the API
returns HTTP 501, which this script reports rather than treating as a failure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_eles_state_timestamps  # noqa: E402

from src.config.logging import configure_logging
from src.domain.estimation.service import build_estimation_service
from src.domain.estimation.weights import K
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# Project-report material, not paper evidence: SSSA is out of scope for the manuscript
# (CLAUDE.md, "Analysis: TSA (CCT) only"), so outputs stay in tmp/.

SAMPLE_SEED: Final[int] = int(os.environ.get("SSSA_BENCHMARK_SAMPLE_SEED", "42"))
BASELINE_SEED: Final[int] = 0
BOOTSTRAP_RESAMPLES: Final[int] = int(os.environ.get("SSSA_BOOTSTRAP_RESAMPLES", "1000"))
BOOTSTRAP_SEED: Final[int] = 42
# Splits the two families the critical mode belongs to. The critical frequency is bimodal on
# eles/2026-06 (p25 0.78 Hz, median 1.75 Hz, full range 0.66 to 2.13 Hz), so this sits in the
# empty middle rather than on any standard: it separates the observed clusters, nothing more.
FAMILY_SPLIT_HZ: Final[float] = 1.2


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


def _critical_by_state(sssa: pd.DataFrame) -> pd.DataFrame:
    """Each state's least-damped mode: the field-standard scalar summary of small-signal risk.

    Minimum damping ratio over the state's modes, with that mode's frequency. This is a
    summary over modes within one state, never a comparison of mode_id across states, so it
    sidesteps the data dictionary's warning that mode indices are state-local.
    """
    modes = sssa.drop_duplicates(subset=["state", "mode_id"])[["state", "mode_id", "real_part", "imag_part"]]
    magnitude = np.hypot(modes["real_part"], modes["imag_part"])
    damping = np.where(magnitude > 0, -modes["real_part"] / magnitude, np.nan)
    frame = modes.assign(damping=damping, freq_hz=modes["imag_part"].abs() / (2 * np.pi))
    critical = frame.loc[frame.groupby("state")["damping"].idxmin()]
    return critical.set_index("state")[["damping", "freq_hz"]].assign(
        family=lambda f: np.where(f["freq_hz"] >= FAMILY_SPLIT_HZ, "local", "interarea")
    )


def _topology_group_by_state(lf: pd.DataFrame, dataset: str, topology_variant: str | None) -> pd.Series:
    """Group each state by the switching key the service actually filters on.

    Reads the same topology_cols file the service resolves for this variant, so the grouping a
    baseline sees is the grouping retrieval was restricted to. Grouping only needs the values
    to be distinguishable, so the raw byte pattern stands in for the service's topology_id.
    """
    processed = PROJECT_DIR / "datasets" / dataset / "processed"
    candidates = [processed / f"topology_cols_{topology_variant}.json"] if topology_variant else []
    candidates.append(processed / "topology_cols.json")
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise SystemExit(f"No topology_cols file for {dataset} under {processed}")
    columns = [c for c in json.loads(path.read_text()) if c in lf.columns]
    logger.info(f"Topology grouping on {len(columns)} columns from {path.name}")
    values = lf[columns].to_numpy(dtype=np.int8)
    return pd.Series([row.tobytes() for row in values], index=lf.index.astype(str))


def _score_state(
    service: Any,
    state_id: Any,
    state: pd.Series,
    truth_rows: pd.DataFrame,
    generators: list[str],
    parmag_cols: list[str],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Pair each recorded mode of this state with its best retrieved counterpart and score.

    Also returns the retrieved states with their query distance, so the critical-mode arm can
    reuse this one service call instead of paying for a second identical query.
    """
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
    t0 = time.perf_counter()
    reports = service.estimate_sssa_by_generator(state=state_dict, exclude_uids=[str(state_id)])
    query_ms = 1000 * (time.perf_counter() - t0)
    if not reports:
        return [{"state": state_id, "covered": False, "query_ms": query_ms}], {}

    # Nearest distance per retrieved state; a state appears once per generator and mode.
    state_distance: dict[str, float] = {}
    for neighbors in reports.values():
        for neighbor in neighbors:
            previous = state_distance.get(neighbor.state)
            if previous is None or neighbor.distance < previous:
                state_distance[neighbor.state] = float(neighbor.distance)

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
        return [{"state": state_id, "covered": False, "query_ms": query_ms}], state_distance

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
    return rows, state_distance


CRITICAL_ARMS: Final[tuple[str, ...]] = (
    "retrieval",
    "topo_median",
    "global_median",
    "random_topo",
    "persistence",
)


def _critical_row(
    state_id: str,
    state_distance: dict[str, float],
    critical: pd.DataFrame,
    groups: pd.Series,
    previous_state: dict[str, str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One state's critical damping ratio under retrieval and under each baseline.

    Every arm predicts the same quantity for the same state, and every one of them excludes
    that state from its own evidence, so the comparison isolates what the arm knows rather
    than how much data it saw.
    """
    truth = critical.loc[state_id]
    row: dict[str, Any] = {
        "state": state_id,
        "damping_true": float(truth["damping"]),
        "freq_hz_true": float(truth["freq_hz"]),
        "family_true": str(truth["family"]),
        "n_retrieved_states": len(state_distance),
    }

    # Retrieval: kernel-weighted over retrieved states, using each state's own critical mode
    # from the archive rather than whichever modes the response happened to carry.
    known = [(s, d) for s, d in state_distance.items() if s in critical.index]
    if known:
        distances = np.array([d for _, d in known], dtype=float)
        weights = K(distances)
        values = critical.loc[[s for s, _ in known], "damping"].to_numpy(dtype=float)
        total = float(weights.sum())
        row["pred_retrieval"] = float((weights * values).sum() / total) if total > 0 else None
        families = critical.loc[[s for s, _ in known], "family"].to_numpy()
        local_weight = float(weights[families == "local"].sum())
        row["family_pred_retrieval"] = "local" if local_weight * 2 > total else "interarea"

    others = critical.index[critical.index != state_id]
    row["pred_global_median"] = float(critical.loc[others, "damping"].median())

    group_members = groups.index[(groups == groups.get(state_id)) & (groups.index != state_id)]
    group_members = [m for m in group_members if m in critical.index]
    if group_members:
        row["pred_topo_median"] = float(critical.loc[group_members, "damping"].median())
        row["pred_random_topo"] = float(critical.loc[group_members[int(rng.integers(len(group_members)))], "damping"])
    row["n_topology_group"] = len(group_members)

    earlier = previous_state.get(state_id)
    if earlier is not None and earlier in critical.index:
        row["pred_persistence"] = float(critical.loc[earlier, "damping"])

    for arm in CRITICAL_ARMS:
        prediction = row.get(f"pred_{arm}")
        row[f"abs_err_{arm}"] = None if prediction is None else abs(row["damping_true"] - prediction)
    return row


def _previous_state_map(state_times: pd.Series | None, states: list[str]) -> dict[str, str]:
    """State -> the chronologically previous state, for the persistence baseline."""
    if state_times is None:
        return {}
    known = state_times[state_times.index.astype(str).isin(states)].sort_values()
    ordered = list(known.index.astype(str))
    return {state: ordered[i - 1] for i, state in enumerate(ordered) if i > 0}


def _bootstrap_mae_gap(
    errors_retrieval: np.ndarray, errors_baseline: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """Paired bootstrap over states on baseline MAE minus retrieval MAE; positive favors
    retrieval. Paired because both arms are resampled on the same states."""
    rng = np.random.default_rng(seed)
    n = len(errors_retrieval)
    if n == 0:
        return {}
    gaps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gaps[i] = errors_baseline[idx].mean() - errors_retrieval[idx].mean()
    low, high = float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))
    return {
        "mae_gap_vs_retrieval": float(gaps.mean()),
        "mae_gap_ci_low": low,
        "mae_gap_ci_high": high,
        "mae_gap_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _summarize_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm error on the critical damping ratio, each against retrieval."""
    summary_rows: list[dict[str, Any]] = []
    for arm in CRITICAL_ARMS:
        error_col, pred_col = f"abs_err_{arm}", f"pred_{arm}"
        if error_col not in df.columns:
            continue
        # Each arm is scored on the states where retrieval also produced a prediction, so the
        # arms are compared on one population rather than on their own most convenient one.
        scored = df[df[error_col].notna() & df["abs_err_retrieval"].notna()]
        if scored.empty:
            continue
        errors = scored[error_col].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "arm": arm,
            "n_states": int(len(scored)),
            "mae": float(errors.mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
            "median_ae": float(np.median(errors)),
            "spearman_rho": float(spearmanr(scored["damping_true"], scored[pred_col]).statistic),
        }
        if arm != "retrieval":
            row.update(
                _bootstrap_mae_gap(
                    scored["abs_err_retrieval"].to_numpy(dtype=float),
                    errors,
                    BOOTSTRAP_RESAMPLES,
                    BOOTSTRAP_SEED,
                )
            )
        summary_rows.append(row)

    family = df[df["family_pred_retrieval"].notna()] if "family_pred_retrieval" in df.columns else df.iloc[:0]
    if not family.empty:
        majority = family["family_true"].mode().iat[0]
        summary_rows.append(
            {
                "arm": "family_classification",
                "n_states": int(len(family)),
                "accuracy_retrieval": float((family["family_pred_retrieval"] == family["family_true"]).mean()),
                "accuracy_majority_class": float((family["family_true"] == majority).mean()),
            }
        )
    return pd.DataFrame(summary_rows)


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

    full_lf = lf
    if n_states and n_states < len(lf):
        rng = np.random.default_rng(SAMPLE_SEED)
        sampled = set(rng.choice(sorted(lf.index.astype(str)), size=n_states, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(sampled)]
    logger.info(f"Query states: {len(lf)} (seed={SAMPLE_SEED})")

    truth_by_state = {str(sid): subset for sid, subset in sssa.groupby("state")}

    # The critical-mode arm and its baselines are built from every state in the archive, not
    # from the query subsample, so a short run and a full run face the same baselines.
    critical = _critical_by_state(sssa)
    groups = _topology_group_by_state(full_lf, dataset, config.topology_variant)
    logger.info(
        f"Critical mode: {len(critical)} states, median damping {critical['damping'].median():.4f}, "
        f"{groups.nunique()} topology groups"
    )
    state_times: pd.Series | None = None
    raw_zip = PROJECT_DIR / "datasets" / dataset / "raw" / "Podatki_DSA.zip"
    if raw_zip.is_file():
        try:
            state_times = load_eles_state_timestamps(raw_zip)
        except Exception as exc:  # noqa: BLE001 - a missing persistence arm must not stop the run
            logger.warning(f"No persistence baseline: {exc}")
    else:
        logger.info(f"No raw archive at {raw_zip}; skipping the persistence baseline.")
    previous_state = _previous_state_map(state_times, list(critical.index))

    logger.info("Building EstimationService (embedded Qdrant)...")
    t0 = time.monotonic()
    service = build_estimation_service()
    if getattr(service, "sssa", None) is None:
        raise SystemExit(f"{dataset} exposes no SSSA store; the API returns HTTP 501.")
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    baseline_rng = np.random.default_rng(BASELINE_SEED)
    rows: list[dict[str, Any]] = []
    critical_rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        truth = truth_by_state.get(str(state_id))
        if truth is None or truth.empty:
            continue
        mode_rows, state_distance = _score_state(service, state_id, state, truth, generators, parmag_cols, baseline_rng)
        rows.extend(mode_rows)
        if str(state_id) in critical.index:
            critical_rows.append(
                _critical_row(str(state_id), state_distance, critical, groups, previous_state, baseline_rng)
            )
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

    if critical_rows:
        critical_df = pd.DataFrame(critical_rows)
        joblib.dump(critical_rows, TMP_DIR / f"report-sssa-critical-{dataset_slug}{suffix}.joblib")
        critical_summary = _summarize_critical(critical_df)
        critical_summary.insert(0, "dataset", dataset)
        critical_csv = TMP_DIR / f"sssa_critical_{dataset_slug}{suffix}.csv"
        critical_summary.to_csv(critical_csv, index=False)
        logger.info(f"Saved critical-mode summary to {critical_csv}")
        print(critical_summary.to_string(index=False))


if __name__ == "__main__":
    main()
