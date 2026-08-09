"""Selected-generator counterpart of bootstrap_risk_coverage.py - same nAURC and
state-level bootstrap methodology (COVERAGES, _risk_coverage_point, _naurc, resampling loop
all unchanged in spirit), applied to generator_diagnostics_selected.py's/
eles_generator_diagnostics_selected.py's per-record output instead of benchmark.py's
oracle-conditioned reports.

Fixed 2026-08-09 (Codex review, ai2ai.md): the original per-metric independent NaN-dropping
(inside _risk_coverage_point alone) let each metric rank over a different-sized population -
on ELES, distance metrics saw all 237,168 selected-covered records but cct_weighted_std/
neighborhood_compactness only ~183k and cct_distance_correlation only ~156k (records below
n=2 neighbors have no defined dispersion/compactness/correlation). Ranking metrics against
each other on different populations makes "which diagnostic is best" incomparable - the
apparent ELES reordering in an earlier pass of this analysis was exactly this artifact, not
a real result. Confirmed the same flaw was already present in bootstrap_risk_coverage.py
itself, i.e. in the oracle-conditioned table currently in the manuscript, not something this
script introduced.

Now computes two independent, internally-consistent metric sets, each masked to its OWN
common finite-value population before any point estimate or resampling (not per-metric
inside the loop - that guard stays only as a defensive assertion that should remove zero
rows once the common mask is applied upstream):

* "main" - the five-metric set proposed for the manuscript's main table
  (cct_weighted_std, distance_min, location_weight_mass, n_eff, neighborhood_compactness -
  spanning outcome agreement, query distance, weight concentration, and neighborhood
  geometry without paying cct_distance_correlation's much larger availability cost).
* "full" - all thirteen candidate diagnostics, for a supplementary table, masked to their
  own (smaller) common support - not comparable to "main"'s numbers since the populations
  differ; report each on its own record/state count, never blended with "main".

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

ALL_METRICS: Final[tuple[tuple[str, bool], ...]] = (
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

# Codex's proposed main-table set (ai2ai.md, 2026-08-09): one representative per category
# (outcome agreement, query distance, weight concentration/effective support, neighborhood
# geometry), excluding cct_distance_correlation_abs since it is not a distinct category and
# its availability cost (largest of all thirteen, especially on ELES) is not worth paying
# for the main table.
MAIN_METRIC_NAMES: Final[tuple[str, ...]] = (
    "cct_weighted_std",
    "distance_min",
    "location_weight_mass",
    "n_eff",
    "neighborhood_compactness",
)
METRIC_SETS: Final[dict[str, tuple[tuple[str, bool], ...]]] = {
    "main": tuple(m for m in ALL_METRICS if m[0] in MAIN_METRIC_NAMES),
    "full": ALL_METRICS,
}


def _report_path(dataset_name: str) -> Path:
    if dataset_name == "bus39":
        return TMP_DIR / "generator_diagnostics_selected_bus39.parquet"
    safe = dataset_name.replace("/", "-")
    topology_variant = os.environ.get("TOPOLOGY_VARIANT", "lines_only")
    return TMP_DIR / f"eles_generator_diagnostics_selected_{safe}_{topology_variant}.parquet"


def _output_path(dataset_name: str, metric_set: str) -> Path:
    safe = dataset_name.strip().lower().replace("/", "-")
    suffix = "" if metric_set == "main" else f"_{metric_set}"
    return PAPER_DATA_DIR / f"bootstrap_ci_selected_{safe}{suffix}.csv"


def _load_frame(dataset_name: str) -> pd.DataFrame:
    df = pd.read_parquet(_report_path(dataset_name))
    df = df[df["covered"]].dropna(subset=["cct_weighted_per_location"]).copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    if "cct_distance_correlation" in df.columns:
        df["cct_distance_correlation_abs"] = df["cct_distance_correlation"].abs()
    return df


def _common_support_mask(df: pd.DataFrame, metrics: tuple[tuple[str, bool], ...]) -> pd.Series:
    mask = pd.Series(np.isfinite(df["err"].to_numpy(dtype=np.float64)), index=df.index)
    for name, _ in metrics:
        mask &= np.isfinite(df[name].to_numpy(dtype=np.float64))
    return mask


def _risk_coverage_point(
    metric_values: np.ndarray,
    err_values: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[float, tuple[float, float]]:
    # Defensive guard, not the primary mechanism: with the common mask already applied
    # upstream, this must remove zero rows for every displayed metric and for err. Asserts
    # rather than silently filtering, so a future regression that desynchronizes the mask
    # fails loudly here too, not just in _run_metric_set's point-estimate pass.
    valid = np.isfinite(metric_values) & np.isfinite(err_values)
    assert valid.all(), (
        f"_risk_coverage_point received {int((~valid).sum())} non-finite value(s) - "
        f"the common-support mask upstream should have removed these already."
    )
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


def _run_metric_set(
    df_covered: pd.DataFrame,
    *,
    dataset_name: str,
    metric_set_name: str,
    metrics: tuple[tuple[str, bool], ...],
) -> None:
    mask = _common_support_mask(df_covered, metrics)
    df = df_covered[mask].copy()
    n_records = len(df)
    n_states = df["state"].nunique()
    # Exact integer counts, not round(rate * n) - same round-tripping concern as the
    # generator-identification denominator fix earlier in this thread (ai2ai.md).
    availability_counts = {
        name: int(np.isfinite(df_covered[name].to_numpy(dtype=np.float64)).sum()) for name, _ in metrics
    }
    availability = {name: count / len(df_covered) for name, count in availability_counts.items()}
    logger.info(
        f"[{dataset_name}/{metric_set_name}] common-support population: {n_records:,} records, "
        f"{n_states:,} states (of {len(df_covered):,} selected-covered records, "
        f"{df_covered['state'].nunique():,} states). Per-metric availability within selected-covered: "
        f"{availability}"
    )

    state_row_indices = df.groupby(df["state"].astype(str)).indices
    unique_states = np.array(list(state_row_indices.keys()))

    metric_arrays = {name: df[name].to_numpy(dtype=np.float64) for name, _ in metrics}
    err_array = df["err"].to_numpy(dtype=np.float64)

    n_err_finite = int(np.isfinite(err_array).sum())
    assert n_err_finite == len(err_array), (
        f"Common-support mask did not eliminate missingness in err "
        f"({len(err_array) - n_err_finite} of {len(err_array)} still non-finite) - "
        f"the upstream mask in _common_support_mask is out of sync."
    )
    point_naurc: dict[str, float] = {}
    for name, higher_is_better in metrics:
        n_finite = int(np.isfinite(metric_arrays[name]).sum())
        assert n_finite == len(metric_arrays[name]), (
            f"Common-support mask did not eliminate missingness for {name!r} "
            f"({len(metric_arrays[name]) - n_finite} of {len(metric_arrays[name])} still non-finite) - "
            f"the upstream mask in _common_support_mask is out of sync with this metric set."
        )
        point_cov = _risk_coverage_point(metric_arrays[name], err_array, higher_is_better=higher_is_better)
        point_naurc[name] = _naurc({c: mae for c, (mae, _rmse) in point_cov.items()})
    logger.info(f"[{dataset_name}/{metric_set_name}] Point-estimate nAURC(MAE): {point_naurc}")

    rng = np.random.default_rng(SEED)
    n_boot_states = len(unique_states)

    bootstrap_mae: dict[str, dict[float, list[float]]] = {name: {c: [] for c in COVERAGES} for name, _ in metrics}
    bootstrap_rmse: dict[str, dict[float, list[float]]] = {name: {c: [] for c in COVERAGES} for name, _ in metrics}
    bootstrap_naurc: dict[str, list[float]] = {name: [] for name, _ in metrics}

    t_start = time.monotonic()
    for b in range(N_BOOTSTRAP):
        sampled_states = rng.choice(unique_states, size=n_boot_states, replace=True)
        idx = np.concatenate([state_row_indices[s] for s in sampled_states])

        boot_err = err_array[idx]
        for name, higher_is_better in metrics:
            boot_metric = metric_arrays[name][idx]
            cov_result = _risk_coverage_point(boot_metric, boot_err, higher_is_better=higher_is_better)
            for cov, (mae, rmse) in cov_result.items():
                bootstrap_mae[name][cov].append(mae)
                bootstrap_rmse[name][cov].append(rmse)
            bootstrap_naurc[name].append(_naurc({c: mae for c, (mae, _rmse) in cov_result.items()}))

        if (b + 1) % 20 == 0:
            elapsed = time.monotonic() - t_start
            logger.info(f"[{dataset_name}/{metric_set_name}] Bootstrap {b + 1}/{N_BOOTSTRAP} ({elapsed:.1f}s elapsed)")

    n_selected_covered_records = len(df_covered)
    n_selected_covered_states = int(df_covered["state"].nunique())

    def _base_row(name: str) -> dict[str, object]:
        n_available = availability_counts[name]
        return {
            "metric_set": metric_set_name,
            "n_records": n_records,
            "n_states": n_states,
            "n_selected_covered_records": n_selected_covered_records,
            "n_selected_covered_states": n_selected_covered_states,
            "n_available": n_available,
            "availability_rate": availability[name],
        }

    rows: list[dict[str, object]] = []
    for name, _ in metrics:
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
                **_base_row(name),
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
                    **_base_row(name),
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
                    **_base_row(name),
                }
            )

    out = pd.DataFrame(rows)
    output_path = _output_path(dataset_name, metric_set_name)
    out.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")
    print(f"=== {dataset_name} / {metric_set_name} (n_records={n_records:,}, n_states={n_states:,}) ===")
    print(out[out["coverage"].isna()].to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bus39")
    parser.add_argument("--metric-set", choices=["main", "full", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    logger.info(f"Loading {_report_path(args.dataset)} ...")
    df_covered = _load_frame(args.dataset)
    logger.info(
        f"Selected-covered population: {len(df_covered):,} records across "
        f"{df_covered['state'].nunique():,} unique states"
    )

    metric_sets = ["main", "full"] if args.metric_set == "both" else [args.metric_set]
    for metric_set_name in metric_sets:
        _run_metric_set(
            df_covered,
            dataset_name=args.dataset,
            metric_set_name=metric_set_name,
            metrics=METRIC_SETS[metric_set_name],
        )


if __name__ == "__main__":
    main()
