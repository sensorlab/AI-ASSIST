"""Bootstrap confidence intervals for the risk-coverage/nAURC analysis.

Resamples pre-fault STATES (not records) with replacement, so the full block of
simulation records belonging to a resampled state moves together - the same grouping
unit used everywhere else in this paper (GroupKFold, LOGO). Reproduces exactly the
same risk_coverage() computation as reports/30_benchmark_results_analysis.ipynb, and
the same normalized-AURC formula as paper/scripts/compute_aurc.py, on each bootstrap
replicate, then reports percentile confidence intervals.

Dataset-parameterized (2026-08-06): originally BUS39-only ("the ELES risk-coverage
numbers are computed from confidential per-record data not present in this repository"),
but that no longer holds - ELES's own report-service-group-kfold-*.joblib is generated
locally by benchmark.py same as BUS39's, so both are covered via --dataset.

Run from repository root:
    uv run python scripts/service/bootstrap_risk_coverage.py --dataset bus39
    uv run python scripts/service/bootstrap_risk_coverage.py --dataset eles/2026-06
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib, .parquet) go to
# tmp/, CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _dataset_safe_name(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("/", "-")


def _report_path(dataset_name: str) -> Path:
    # Mirrors benchmark.py's REPORT_PATH suffix convention exactly: bus39 keeps the bare
    # historical filename, every other dataset gets a "-{safe_name}" suffix.
    safe_name = _dataset_safe_name(dataset_name)
    suffix = "" if safe_name == "bus39" else f"-{safe_name}"
    return TMP_DIR / f"report-service-group-kfold{suffix}.joblib"


def _output_path(dataset_name: str) -> Path:
    return PAPER_DATA_DIR / f"bootstrap_ci_{_dataset_safe_name(dataset_name)}.csv"


COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
# Overridable via BOOTSTRAP_RISK_COVERAGE_N for a higher-resolution percentile-CI rerun
# (200 resamples is on the low side for stable percentile CIs, particularly at the 50%
# coverage tail where fewer states remain after selective filtering).
N_BOOTSTRAP: Final[int] = int(os.environ.get("BOOTSTRAP_RISK_COVERAGE_N", "200"))
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5

# (metric, higher_is_better) - the original set matches reports/30_benchmark_results_analysis.ipynb
# / paper/scripts/compute_aurc.py exactly. The four added 2026-08-06 (n_eff_fraction,
# n_unique_states, cct_weighted_std, cct_distance_correlation_abs) are new: n_eff_fraction fixes
# n_eff's mechanical confound with retrieved-pool size (uniform weights give n_eff == n_neighbors
# exactly, so pooling groups of different sizes conflates "concentrated evidence" with "small
# pool"); the other three were already computed by LocationReportStats but never extracted into
# the benchmark harness or tested against error before now.
METRICS: Final[tuple[tuple[str, bool], ...]] = (
    ("location_weight_mass", True),
    ("n_eff", True),
    ("n_eff_fraction", True),
    ("n_neighbors", True),
    ("n_unique_states", True),
    ("neighborhood_compactness", True),
    ("cct_weighted_std", False),
    ("cct_distance_correlation_abs", True),
    ("distance_min", False),
    ("distance_mean", False),
    ("distance_median", False),
    ("distance_spread", False),
    ("distance_norm", False),
)


def _load_frame(dataset_name: str = "bus39") -> pd.DataFrame:
    payload = joblib.load(_report_path(dataset_name))
    if isinstance(payload, list):
        records = payload
    elif "predictions" in payload:
        records = payload["predictions"]
    else:
        records = payload["data"]
    df = pd.DataFrame(records)
    df = df.drop(columns=["prediction_summary"], errors="ignore")
    df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    # cct_distance_correlation is signed and can be informative in either direction (a strong
    # negative correlation is just as meaningful as a strong positive one) - risk_coverage()
    # sorting needs a single monotonic "more informative" direction, so rank by magnitude.
    if "cct_distance_correlation" in df.columns:
        df["cct_distance_correlation_abs"] = df["cct_distance_correlation"].abs()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bus39")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    report_path = _report_path(args.dataset)
    output_path = _output_path(args.dataset)
    logger.info(f"Loading {report_path} ...")
    df = _load_frame(args.dataset)
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
    out.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
