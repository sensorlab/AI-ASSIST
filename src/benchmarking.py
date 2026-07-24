from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

CONTINGENCY_CATEGORICAL_COLUMNS = ("Location", "Terminal", "Type")
ABSOLUTE_ERROR_QUANTILES = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00)


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
