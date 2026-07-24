"""Bootstrap confidence intervals for the BUS39 risk-coverage/nAURC analysis.

Resamples pre-fault STATES (not records) with replacement, so the full block of
simulation records belonging to a resampled state moves together - the same grouping
unit used everywhere else in this paper (GroupKFold, LOGO). Reproduces exactly the
same risk_coverage() computation as reports/30_benchmark_results_analysis.ipynb, and
the same normalized-AURC formula as paper/scripts/compute_aurc.py, on each bootstrap
replicate, then reports percentile confidence intervals.

Only covers BUS39: the ELES risk-coverage numbers are computed from confidential
per-record data not present in this repository, so no equivalent bootstrap is possible
for them here.

Run from repository root:
    uv run python scripts/service/bootstrap_risk_coverage.py
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
REPORT_PATH: Final[Path] = PROJECT_DIR / "report-2026-05-29.joblib"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "bootstrap_ci_bus39.csv"

COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
N_BOOTSTRAP: Final[int] = 200
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5

# (metric, higher_is_better) - identical set and directions to
# reports/30_benchmark_results_analysis.ipynb / paper/scripts/compute_aurc.py.
METRICS: Final[tuple[tuple[str, bool], ...]] = (
    ("location_weight_mass", True),
    ("n_eff", True),
    ("n_neighbors", True),
    ("neighborhood_compactness", True),
    ("distance_min", False),
    ("distance_mean", False),
    ("distance_median", False),
    ("distance_spread", False),
    ("distance_norm", False),
)


def _load_frame() -> pd.DataFrame:
    payload = joblib.load(REPORT_PATH)
    df = pd.DataFrame(payload if isinstance(payload, list) else payload["data"])
    df = df.drop(columns=["prediction_summary"], errors="ignore")
    df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    return df


def _risk_coverage_point(
    metric_values: np.ndarray,
    err_values: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[float, tuple[float, float]]:
    """Return {coverage: (mae, rmse)} for one metric on one (bootstrap) sample,
    matching reports/30_benchmark_results_analysis.ipynb::risk_coverage() exactly:
    sort by metric, keep the top ceil(coverage * n), compute MAE/RMSE of err on that
    retained subset."""
    valid = np.isfinite(metric_values) & np.isfinite(err_values)
    metric_values = metric_values[valid]
    err_values = err_values[valid]

    order = np.argsort(metric_values)
    if higher_is_better:
        order = order[::-1]
    err_sorted = err_values[order]

    n = len(err_sorted)
    out: dict[float, tuple[float, float]] = {}
    for cov in COVERAGES:
        k = math.ceil(cov * n)
        kept = err_sorted[:k]
        out[cov] = (float(kept.mean()), float(np.sqrt((kept**2).mean())))
    return out


def _naurc(coverage_maes: dict[float, float]) -> float:
    """Normalized AURC via trapezoidal integration, matching
    paper/scripts/compute_aurc.py exactly: trapz(err(c), c) / (err_full * (c_max - c_min))."""
    covs = sorted(coverage_maes.keys())
    errs = [coverage_maes[c] for c in covs]
    area = float(np.trapezoid(errs, covs))
    err_full = coverage_maes[max(covs)]
    span = max(covs) - min(covs)
    return area / (err_full * span)


def main() -> None:
    configure_logging()
    logger.info(f"Loading {REPORT_PATH} ...")
    df = _load_frame()
    logger.info(f"Loaded {len(df):,} covered records across {df['state'].nunique():,} unique states")

    # DataFrameGroupBy.indices is an O(N) group-position lookup (dict: group key ->
    # ndarray of positional row indices) - avoid any O(states x records) approach here,
    # since a boolean-mask-per-state loop over ~1M rows x ~20k states is intractable.
    state_row_indices = df.groupby(df["state"].astype(str)).indices
    unique_states = np.array(list(state_row_indices.keys()))

    metric_arrays = {name: df[name].to_numpy(dtype=np.float64) for name, _ in METRICS}
    err_array = df["err"].to_numpy(dtype=np.float64)

    # Point estimate (full sample, no resampling) - sanity check against the existing
    # risk_coverage_bus39.csv this should reproduce exactly.
    point_naurc: dict[str, float] = {}
    for name, higher_is_better in METRICS:
        point_cov = _risk_coverage_point(metric_arrays[name], err_array, higher_is_better=higher_is_better)
        point_naurc[name] = _naurc({c: mae for c, (mae, _rmse) in point_cov.items()})
    logger.info(f"Point-estimate nAURC(MAE): {point_naurc}")

    rng = np.random.default_rng(SEED)
    n_states = len(unique_states)

    # bootstrap_mae[metric][coverage] -> list of length N_BOOTSTRAP
    bootstrap_mae: dict[str, dict[float, list[float]]] = {name: {c: [] for c in COVERAGES} for name, _ in METRICS}
    bootstrap_rmse: dict[str, dict[float, list[float]]] = {name: {c: [] for c in COVERAGES} for name, _ in METRICS}
    bootstrap_naurc: dict[str, list[float]] = {name: [] for name, _ in METRICS}

    t_start = time.monotonic()
    for b in range(N_BOOTSTRAP):
        sampled_states = rng.choice(unique_states, size=n_states, replace=True)
        idx = np.concatenate([state_row_indices[s] for s in sampled_states])

        boot_err = err_array[idx]
        for name, higher_is_better in METRICS:
            boot_metric = metric_arrays[name][idx]
            cov_result = _risk_coverage_point(boot_metric, boot_err, higher_is_better=higher_is_better)
            for cov, (mae, rmse) in cov_result.items():
                bootstrap_mae[name][cov].append(mae)
                bootstrap_rmse[name][cov].append(rmse)
            bootstrap_naurc[name].append(_naurc({c: mae for c, (mae, _rmse) in cov_result.items()}))

        if (b + 1) % 20 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(f"Bootstrap {b + 1}/{N_BOOTSTRAP} ({elapsed:.1f}s elapsed)")

    rows: list[dict[str, object]] = []
    for name, _ in METRICS:
        naurc_samples = np.array(bootstrap_naurc[name])
        rows.append(
            {
                "metric": name,
                "coverage": None,
                "quantity": "nAURC(MAE)",
                "point_estimate": point_naurc[name],
                "ci_low": float(np.percentile(naurc_samples, CI_LOW)),
                "ci_high": float(np.percentile(naurc_samples, CI_HIGH)),
                "n_bootstrap": N_BOOTSTRAP,
            }
        )
        for cov in COVERAGES:
            mae_samples = np.array(bootstrap_mae[name][cov])
            rmse_samples = np.array(bootstrap_rmse[name][cov])
            rows.append(
                {
                    "metric": name,
                    "coverage": cov,
                    "quantity": "mae",
                    "point_estimate": float(np.mean(mae_samples)),
                    "ci_low": float(np.percentile(mae_samples, CI_LOW)),
                    "ci_high": float(np.percentile(mae_samples, CI_HIGH)),
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
            rows.append(
                {
                    "metric": name,
                    "coverage": cov,
                    "quantity": "rmse",
                    "point_estimate": float(np.mean(rmse_samples)),
                    "ci_low": float(np.percentile(rmse_samples, CI_LOW)),
                    "ci_high": float(np.percentile(rmse_samples, CI_HIGH)),
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
