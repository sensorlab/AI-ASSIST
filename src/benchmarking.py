import math
from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

CONTINGENCY_CATEGORICAL_COLUMNS = ("Location", "Terminal", "Type")
ABSOLUTE_ERROR_QUANTILES = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)
COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)


def group_k_fold_indices(
    groups: Iterable[Any],
    *,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    group_values = pd.Series(groups, copy=False).astype(str).reset_index(drop=True)
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(group_values, groups=group_values))


def group_k_fold_test_groups(
    groups: Iterable[Any],
    *,
    n_splits: int = 5,
) -> list[frozenset[str]]:
    group_values = pd.Series(groups, copy=False).astype(str).reset_index(drop=True)
    return [
        frozenset(group_values.iloc[test_idx]) for _, test_idx in group_k_fold_indices(group_values, n_splits=n_splits)
    ]


def absolute_error_quantiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    absolute_error = np.abs(y_true - y_pred)
    values = np.quantile(absolute_error, ABSOLUTE_ERROR_QUANTILES)
    return {
        f"ae_q{int(quantile * 100):02d}": float(value)
        for quantile, value in zip(ABSOLUTE_ERROR_QUANTILES, values, strict=True)
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    coverage: float = 1.0,
) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "coverage": float(coverage),
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "max_ae": float("nan"),
            **{f"ae_q{int(q * 100):02d}": float("nan") for q in ABSOLUTE_ERROR_QUANTILES},
        }

    absolute_error = np.abs(y_true - y_pred)
    return {
        "coverage": float(coverage),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "median_ae": float(np.median(absolute_error)),
        "max_ae": float(np.max(absolute_error)),
        **absolute_error_quantiles(y_true, y_pred),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [column for column in results.columns if column not in {"fold", "model"}]
    return results.groupby("model")[metric_cols].agg(["mean", "std"]).sort_values(("mae", "mean"))


def _risk_coverage_fractional(
    metric_values: np.ndarray,
    err_values: np.ndarray,
    *,
    higher_is_better: bool,
    coverages: tuple[float, ...],
) -> dict[float, tuple[float, float]]:
    # Groups of identical (direction-adjusted) metric values, in most-trusted-first order.
    # A coverage cutoff either falls between groups (every group fully in or out - identical
    # to "hard") or inside one group, which "hard" resolves by an arbitrary tie order. Here
    # that boundary group is weighted by the fraction of it needed to reach the target count,
    # equal to the expected MAE under uniform-random selection within the group - no specific
    # tied record is treated as more "in" than another.
    primary = -metric_values if higher_is_better else metric_values
    order = np.argsort(primary, kind="stable")
    primary_sorted = primary[order]
    err_sorted = err_values[order]
    n = len(err_sorted)

    change = np.nonzero(np.diff(primary_sorted))[0] + 1
    group_starts = np.concatenate(([0], change))
    group_sizes = np.diff(np.concatenate((group_starts, [n])))
    group_sums = np.add.reduceat(err_sorted, group_starts)
    cum_counts = np.cumsum(group_sizes)
    cum_sums = np.cumsum(group_sums)

    out: dict[float, tuple[float, float]] = {}
    for cov in coverages:
        k = math.ceil(cov * n)
        if k <= 0:
            out[cov] = (float("nan"), float("nan"))
            continue
        gi = int(np.searchsorted(cum_counts, k, side="left"))
        prev_count = cum_counts[gi - 1] if gi > 0 else 0
        prev_sum = cum_sums[gi - 1] if gi > 0 else 0.0
        remaining = k - prev_count
        frac_sum = (remaining / group_sizes[gi]) * group_sums[gi] if remaining > 0 else 0.0
        # RMSE has no analogous closed-form expectation under fractional weighting
        # (sqrt(E[MSE]) != E[RMSE] in general) - not defined here, see risk_coverage_point.
        out[cov] = (float((prev_sum + frac_sum) / k), float("nan"))
    return out


def risk_coverage_point(
    metric_values: np.ndarray,
    err_values: np.ndarray,
    *,
    higher_is_better: bool,
    coverages: Iterable[float] = COVERAGES,
    tie_policy: str = "hard",
    rng: np.random.Generator | None = None,
) -> dict[float, tuple[float, float]]:
    """Return {coverage: (mae, rmse)} for one diagnostic on one sample: sort records by
    the diagnostic (most-trusted first), keep the top ceil(coverage * n), score err on
    that retained subset.

    tie_policy governs the order of records sharing an identical diagnostic value, when a
    coverage cutoff falls inside such a group:
    - "hard" (default): argsort's implementation-default order. Byte-identical to every
      previously committed bootstrap_risk_coverage*.py result - this is what they used
      before tie handling was made explicit (2026-08-09).
    - "randomized": ties broken by an independent random permutation *within* each exact-
      value group (via `rng`, required). For auditing tie sensitivity only; not used for
      any committed result.
    - "fractional": no arbitrary tie-break - see _risk_coverage_fractional. RMSE is
      returned as NaN under this policy (see below), so use "hard" or "randomized" where
      RMSE is needed.
    """
    valid = np.isfinite(metric_values) & np.isfinite(err_values)
    metric_values = metric_values[valid]
    err_values = err_values[valid]
    coverages = tuple(coverages)

    if tie_policy == "fractional":
        return _risk_coverage_fractional(
            metric_values, err_values, higher_is_better=higher_is_better, coverages=coverages
        )

    if tie_policy == "hard":
        order = np.argsort(metric_values)
        if higher_is_better:
            order = order[::-1]
    elif tie_policy == "randomized":
        if rng is None:
            raise ValueError("tie_policy='randomized' requires an rng")
        primary = -metric_values if higher_is_better else metric_values
        order = np.lexsort((rng.random(len(primary)), primary))
    else:
        raise ValueError(f"unknown tie_policy: {tie_policy!r}")

    err_sorted = err_values[order]
    n = len(err_sorted)
    out: dict[float, tuple[float, float]] = {}
    for cov in coverages:
        k = math.ceil(cov * n)
        kept = err_sorted[:k]
        out[cov] = (float(kept.mean()), float(np.sqrt((kept**2).mean())))
    return out


def naurc(coverage_maes: dict[float, float]) -> float:
    covs = sorted(coverage_maes.keys())
    errs = [coverage_maes[c] for c in covs]
    area = float(np.trapezoid(errs, covs))
    err_full = coverage_maes[max(covs)]
    span = max(covs) - min(covs)
    return area / (err_full * span)
