from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Final

from sklearnex import patch_sklearn

patch_sklearn()

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from src.benchmarking import (
    CONTINGENCY_CATEGORICAL_COLUMNS,
    group_k_fold_indices,
    regression_metrics,
    summarize_results,
)
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths, _make_scaler_for_dataset
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
REPORT_PATH: Final[Path] = PROJECT_DIR / "report-ml-regression-2026-05-29.joblib"


def _configured_dataset_paths() -> tuple[Path, Path]:
    data_dir = get_app_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir

    lf_path, tsa_path, _ = _dataset_paths(data_dir, get_qdrant_config().dataset_name)
    return lf_path, tsa_path


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_record_table(
    lf: pd.DataFrame,
    tsa: pd.DataFrame,
    *,
    scaler: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build a supervised regression table.

    Each row corresponds to one state-contingency simulation record:

        features = transformed pre-fault state features + contingency descriptors
        target   = CCT
        group    = pre-fault state ID

    The group column must be used for cross-validation to avoid leakage between
    simulation records derived from the same pre-fault state.
    """

    if "state" not in tsa.columns:
        raise ValueError("Expected `tsa` to contain a `state` column.")
    if "CCT" not in tsa.columns:
        raise ValueError("Expected `tsa` to contain a `CCT` column.")

    lf_scaled = scaler.fit_transform(lf)
    if not isinstance(lf_scaled, pd.DataFrame):
        lf_scaled = pd.DataFrame(lf_scaled, index=lf.index)

    lf_scaled = lf_scaled.copy()
    lf_scaled["state"] = lf_scaled.index.astype(str)

    tsa_records = tsa.copy()
    tsa_records["state"] = tsa_records["state"].astype(str)

    records = tsa_records.merge(
        lf_scaled,
        on="state",
        how="inner",
        validate="many_to_one",
    )

    if records.empty:
        raise ValueError("No records after merging LF and TSA tables. Check state IDs.")

    y = records["CCT"].astype(float)
    groups = records["state"]

    # Keep only contingency descriptors that are available in the TSA table.
    categorical_cols = [c for c in CONTINGENCY_CATEGORICAL_COLUMNS if c in records.columns]

    # Exclude outcomes and identifiers.
    excluded_cols = {
        "CCT",
        "experiment",  # wide-to-long ordinal, not an inference input
        "target",
        "state",
    }

    numeric_cols = [
        c
        for c in records.columns
        if c not in excluded_cols and c not in categorical_cols and pd.api.types.is_numeric_dtype(records[c])
    ]

    X = records[numeric_cols + categorical_cols].copy()

    return X, y, groups


def make_models(categorical_cols: list[str]) -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", _one_hot_encoder(), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return {
        "global_median": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        max_iter=300,
                        learning_rate=0.05,
                        l2_regularization=1e-3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def location_median_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    location_col: str = "Location",
) -> np.ndarray:
    global_median = float(y_train.median())

    if location_col not in X_train.columns or location_col not in X_test.columns:
        return np.full(len(X_test), global_median, dtype=float)

    medians = y_train.groupby(X_train[location_col].astype(str)).median()

    return X_test[location_col].astype(str).map(medians).fillna(global_median).to_numpy(dtype=float)


def run_group_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    categorical_cols = [c for c in CONTINGENCY_CATEGORICAL_COLUMNS if c in X.columns]
    models = make_models(categorical_cols)

    rows: list[dict[str, float | str | int]] = []
    iterator = group_k_fold_indices(groups, n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(tqdm(iterator, total=n_splits, desc="GroupKFold regression")):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        y_test_np = y_test.to_numpy(dtype=float)

        # Simple contingency-aware baseline.
        y_pred = location_median_baseline(X_train, y_train, X_test)
        rows.append(
            {
                "fold": fold,
                "model": "location_median",
                **regression_metrics(y_test_np, y_pred),
            }
        )

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            rows.append(
                {
                    "fold": fold,
                    "model": name,
                    **regression_metrics(y_test_np, pred),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    lf_path, tsa_path = _configured_dataset_paths()

    logger.info(f"Benchmark dataset: lf={lf_path}, tsa={tsa_path}")
    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)

    scaler = _make_scaler_for_dataset(get_qdrant_config().dataset_name)

    X, y, groups = build_record_table(lf, tsa, scaler=scaler)

    logger.info(
        f"Records: {len(X):,}; States/groups: {groups.nunique():,}; Features: {X.shape[1]:,}; "
        f"CCT mean: {y.mean():.3f}; CCT median: {y.median():.3f}; CCT min/max: {y.min():.3f} / {y.max():.3f}"
    )

    results = run_group_cv(X, y, groups, n_splits=5)
    summary = summarize_results(results)

    print("\nPer-fold results:")
    print(results)

    print("\nSummary:")
    print(summary)

    payload = {
        "results": results,
        "summary": summary,
        "n_records": len(X),
        "n_groups": groups.nunique(),
        "features": list(X.columns),
        "target": "CCT",
        "split": "GroupKFold by pre-fault state",
    }

    joblib.dump(payload, REPORT_PATH)
    logger.info(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
