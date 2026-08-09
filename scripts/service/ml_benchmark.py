from __future__ import annotations

import logging
import os
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
from sklearn.feature_selection import VarianceThreshold
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
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
# Report/CSV filenames are keyed by dataset name (2026-08-08 fix): the previous fixed
# "report-ml-regression-2026-05-29.joblib" name meant running this for a second dataset
# (e.g. eles/2026-06) would silently overwrite the first dataset's report.


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
    # Clear the index name before adding a same-named "state" column: on datasets whose lf.pkl
    # index is itself named "state" (e.g. ELES), leaving the name in place makes pandas' merge()
    # below raise "'state' is both an index level and a column label, which is ambiguous."
    # BUS39's lf.pkl index happens to be unnamed, which is why this went unnoticed until this
    # script was first run against ELES (2026-08-08).
    lf_scaled.index = lf_scaled.index.rename(None)
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


def make_models(categorical_cols: list[str], *, max_features: float | str = 1.0) -> dict[str, Pipeline]:
    # max_features defaults to 1.0 (every feature considered at every split), matching
    # RandomForestRegressor/ExtraTreesRegressor's own sklearn default -- unlike the classifier
    # variants, which default to "sqrt". That default is tractable on BUS39's 260 scaled
    # features but not on ELES's 12,524 (mostly one-hot categorical): evaluating splits over
    # every feature at every node, for 300 trees x 5 folds, made a single ELES run run for
    # 1.5+ hours and ~100 GB RSS without finishing (2026-08-08). Overridable via
    # ML_BENCHMARK_MAX_FEATURES so BUS39's already-reported number stays reproducible at its
    # original setting while a wide dataset can opt into "sqrt"/"log2"/a fraction.
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", _one_hot_encoder(), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # Drops exact-zero-variance columns (fit per training fold, so this cannot leak test
    # information) before the model sees them. Strictly a no-op on results: a column that
    # never varies within a fold carries no information and a tree can never split on it
    # anyway. On ELES's one-hot-heavy 12,524-column representation, many of those columns are
    # per-bus/per-branch indicators that are constant (usually all-zero) within most folds --
    # dropping them shrinks the matrix every tree-based model actually has to scan (2026-08-08,
    # prompted by the ELES extra_trees run above being far slower than max_features=sqrt alone
    # explained).
    drop_constant = VarianceThreshold(threshold=0.0)

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
                ("drop_constant", drop_constant),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features=max_features,
                        n_jobs=8,  # capped, not -1: avoids starving other concurrent benchmark jobs of CPU/memory
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("drop_constant", drop_constant),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features=max_features,
                        n_jobs=8,  # capped, not -1: avoids starving other concurrent benchmark jobs of CPU/memory
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("drop_constant", drop_constant),
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
    max_features: float | str = 1.0,
    model_names: set[str] | None = None,
) -> pd.DataFrame:
    categorical_cols = [c for c in CONTINGENCY_CATEGORICAL_COLUMNS if c in X.columns]
    models = make_models(categorical_cols, max_features=max_features)
    if model_names is not None:
        # Each model's Pipeline independently re-fits the (shared, but per-Pipeline-refitted)
        # ColumnTransformer preprocessing step, so every extra model in this dict multiplies
        # the cost of one-hot + passthrough transforming the full dense feature matrix, once
        # per fold. On a wide dataset (ELES: 12,524 columns) that redundant preprocessing, not
        # just model fitting, dominates runtime -- restricting to only the model(s) actually
        # needed (typically just "extra_trees", the one the paper cites) avoids paying for the
        # other three every time.
        models = {name: pipeline for name, pipeline in models.items() if name in model_names}

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
    dataset = get_qdrant_config().dataset_name
    dataset_slug = dataset.replace("/", "-")
    lf_path, tsa_path = _configured_dataset_paths()

    logger.info(f"Benchmark dataset: {dataset} (lf={lf_path}, tsa={tsa_path})")
    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)

    scaler = _make_scaler_for_dataset(dataset)

    X, y, groups = build_record_table(lf, tsa, scaler=scaler)

    logger.info(
        f"Records: {len(X):,}; States/groups: {groups.nunique():,}; Features: {X.shape[1]:,}; "
        f"CCT mean: {y.mean():.3f}; CCT median: {y.median():.3f}; CCT min/max: {y.min():.3f} / {y.max():.3f}"
    )

    max_features_env = os.environ.get("ML_BENCHMARK_MAX_FEATURES", "1.0")
    try:
        max_features: float | str = float(max_features_env)
    except ValueError:
        max_features = max_features_env
    logger.info(f"RandomForest/ExtraTrees max_features={max_features!r}")

    models_env = os.environ.get("ML_BENCHMARK_MODELS")
    model_names = {name.strip() for name in models_env.split(",")} if models_env else None
    if model_names is not None:
        logger.info(f"Restricting to models: {sorted(model_names)}")

    results = run_group_cv(X, y, groups, n_splits=5, max_features=max_features, model_names=model_names)
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
        "max_features": max_features,
    }

    report_path = TMP_DIR / f"report-ml-regression-{dataset_slug}.joblib"
    joblib.dump(payload, report_path)
    logger.info(f"Saved report to {report_path}")

    # Flat, paper-consumable CSV: one row per model, GroupKFold-by-state MAE/RMSE etc., with
    # pre-fault state features plus Location/Terminal/Type supplied as input columns -
    # Crit_gen is deliberately excluded above (it's a simulation outcome, not a pre-fault
    # input; see build_record_table and tests/test_service_benchmark.py's assertion of this).
    # This is the source for the paper's ExtraTrees-vs-retrieval comparison (Results,
    # De-oracling), compared there against retrieval's non-oracle, highest-support-generator
    # row - not its oracle row, since this model never receives the critical generator either
    # (corrected 2026-08-09, journal.tex:112 previously claimed otherwise).
    flat_summary = summary.copy()
    flat_summary.columns = ["_".join(str(part) for part in col if part) for col in flat_summary.columns]
    flat_summary = flat_summary.reset_index()
    flat_summary.insert(0, "dataset", dataset)
    flat_summary.insert(1, "n_records", len(X))
    flat_summary.insert(2, "n_groups", int(groups.nunique()))
    flat_summary.insert(3, "max_features", str(max_features))
    csv_path = PAPER_DATA_DIR / f"ml_benchmark_summary-{dataset_slug}.csv"
    flat_summary.to_csv(csv_path, index=False)
    logger.info(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
