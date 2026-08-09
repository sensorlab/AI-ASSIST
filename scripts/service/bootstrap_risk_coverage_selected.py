"""Selected-generator counterpart of bootstrap_risk_coverage.py - identical nAURC and
state-level bootstrap methodology (COVERAGES, METRICS, _risk_coverage_point, _naurc,
resampling loop all unchanged), applied to generator_diagnostics_selected.py's/
eles_generator_diagnostics_selected.py's per-record output instead of benchmark.py's
oracle-conditioned reports. See those scripts' docstrings and ai2ai.md (2026-08-09, Codex
review) for why: Table 2 as originally computed characterizes diagnostic informativeness
only under the true-generator-conditioned report, not the deployable one this promotes to
the main text.

Run from the repository root:
    uv run python scripts/service/bootstrap_risk_coverage_selected.py --dataset bus39
    uv run python scripts/service/bootstrap_risk_coverage_selected.py --dataset eles/2026-06
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
N_BOOTSTRAP: Final[int] = int(os.environ.get("BOOTSTRAP_RISK_COVERAGE_N", "200"))
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5

# Identical metric list to bootstrap_risk_coverage.py - location_weight_mass here is the
# selected generator's own weight_mass (comparable within that report the same way the
# oracle report's was), not the cross-group raw kernel mass used to pick the generator.
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


def _report_path(dataset_name: str) -> Path:
    if dataset_name == "bus39":
        return TMP_DIR / "generator_diagnostics_selected_bus39.parquet"
    safe = dataset_name.replace("/", "-")
    topology_variant = os.environ.get("TOPOLOGY_VARIANT", "lines_only")
    return TMP_DIR / f"eles_generator_diagnostics_selected_{safe}_{topology_variant}.parquet"


def _output_path(dataset_name: str) -> Path:
    safe = dataset_name.strip().lower().replace("/", "-")
    return PAPER_DATA_DIR / f"bootstrap_ci_selected_{safe}.csv"


def _load_frame(dataset_name: str) -> pd.DataFrame:
    df = pd.read_parquet(_report_path(dataset_name))
    df = df[df["covered"]].dropna(subset=["cct_weighted_per_location"]).copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    if "cct_distance_correlation" in df.columns:
        df["cct_distance_correlation_abs"] = df["cct_distance_correlation"].abs()
    return df


def _risk_coverage_point(
    metric_values: np.ndarray,
    err_values: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[float, tuple[float, float]]:
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

    state_row_indices = df.groupby(df["state"].astype(str)).indices
    unique_states = np.array(list(state_row_indices.keys()))

    metric_arrays = {name: df[name].to_numpy(dtype=np.float64) for name, _ in METRICS}
    err_array = df["err"].to_numpy(dtype=np.float64)

    point_naurc: dict[str, float] = {}
    for name, higher_is_better in METRICS:
        point_cov = _risk_coverage_point(metric_arrays[name], err_array, higher_is_better=higher_is_better)
        point_naurc[name] = _naurc({c: mae for c, (mae, _rmse) in point_cov.items()})
    logger.info(f"Point-estimate nAURC(MAE): {point_naurc}")

    rng = np.random.default_rng(SEED)
    n_states = len(unique_states)

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
