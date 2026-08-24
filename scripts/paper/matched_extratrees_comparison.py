"""Build the matched ExtraTrees-versus-retrieval comparison used by the paper.

Both methods must use five GroupKFold splits by pre-fault state. ExtraTrees must use
transformed pre-fault state features plus Location only. Retrieval must use the
highest-support candidate generator at that same recorded location. Metrics on the
"common" population include only records for which retrieval returned an estimate.

Run from the parent repository root:
    uv run python scripts/paper/matched_extratrees_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from src.benchmarking import regression_metrics

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
OUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "matched_extratrees_comparison.csv"

# Table 2 cannot lose its two original comparators, so their absence is an error rather than a
# silently shorter table. Any further model classes present are reported alongside them.
REQUIRED_MODELS: Final[set[str]] = {"extra_trees", "location_median"}
MODEL_ORDER: Final[tuple[str, ...]] = (
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "location_median",
)
# max_features governs the per-split feature budget of the tree ensembles only. Reporting the
# run's value against a model that does not use it would misdescribe that model.
MAX_FEATURES_APPLIES: Final[frozenset[str]] = frozenset({"extra_trees", "random_forest"})

CONFIGS: Final[dict[str, dict[str, Any]]] = {
    "BUS39": {
        "slug": "bus39",
        "retrieval_path": TMP_DIR / "generator_deoracled_records.parquet",
        "retrieval_prediction": "pred_selection",
    },
    "ELES": {
        "slug": "eles-2026-06",
        "retrieval_path": TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_group_kfold.parquet",
        "retrieval_prediction": "cct_weighted_per_location",
    },
}


def _label(model: str, max_features: str) -> str:
    """Row identity for the comparison table.

    A tree ensemble trained on every feature at each split and one trained on sqrt(p) are
    different comparators, and the reviewers' objection to Table 2 was precisely that ELES and
    BUS39 were compared at different budgets. Keying those rows by model alone would collapse
    them onto each other, so the budget is part of the identity. Models that do not consult
    max_features keep a bare name: labelling them by a budget they ignore would invent a
    distinction, and would also split one computation into duplicate rows across runs.
    """
    return f"{model}@{max_features}" if model in MAX_FEATURES_APPLIES else model


def _load_ml(slug: str) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Load every supervised run for this dataset, keyed by model and feature budget.

    Each ml_benchmark run writes a report and a prediction artifact sharing a filename suffix,
    so the two are paired by that suffix rather than assumed. Runs may differ in max_features -
    that is the point of this comparison - but must agree on the split and the contingency
    inputs, since rows disagreeing on those describe different experiments.
    """
    prefix = f"report-ml-regression-{slug}"
    report_paths = sorted(TMP_DIR.glob(f"{prefix}*.joblib"))
    if not report_paths:
        raise FileNotFoundError(f"{slug}: no {prefix}*.joblib under {TMP_DIR}")

    frames: list[pd.DataFrame] = []
    budgets: dict[str, str] = {}
    for report_path in report_paths:
        suffix = report_path.stem[len(prefix) :]
        prediction_path = TMP_DIR / f"ml_benchmark_predictions-{slug}{suffix}.parquet"
        if not prediction_path.exists():
            raise FileNotFoundError(f"{report_path.name} has no matching {prediction_path.name}")
        report = joblib.load(report_path)
        # Temporally blocked runs are a different experiment: they withhold training states near
        # each test state, so their predictions are not comparable with the unblocked arm this
        # table reports. They share the dataset prefix and sort after the plain artifact, so
        # without this skip the concat's keep="last" silently replaces the unblocked predictions
        # with the blocked ones - which moved ELES gradient boosting from 0.0555 to 0.0595 before
        # it was caught. The blocked comparison belongs to the near-duplicate analysis instead.
        if report.get("temporal_exclusion_hours"):
            continue
        if report.get("split") != "GroupKFold by pre-fault state":
            raise ValueError(f"{report_path.name}: unexpected supervised split: {report.get('split')!r}")
        if report.get("contingency_columns") != ["Location"]:
            raise ValueError(
                f"{report_path.name}: expected Location-only inputs, got {report.get('contingency_columns')!r}"
            )
        max_features = str(report["max_features"])
        frame = pd.read_parquet(prediction_path)
        frame["model"] = [_label(model, max_features) for model in frame["model"]]
        for label in frame["model"].unique():
            budgets[label] = max_features if "@" in label else "not applicable"
        frames.append(frame)

    key = ["state", "record_ordinal"]
    predictions = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["model", *key], keep="last")

    labels = set(predictions["model"])
    missing = {base for base in REQUIRED_MODELS if not any(lbl.split("@")[0] == base for lbl in labels)}
    if missing:
        raise ValueError(f"{slug}: required supervised models absent: {sorted(missing)}")
    for label, subset in predictions.groupby("model", observed=True):
        if subset.duplicated(key).any():
            raise ValueError(f"{slug}: duplicate record key for {label}")

    wide = predictions.pivot(
        index=["state", "record_ordinal", "fold", "cct_true"],
        columns="model",
        values="cct_pred",
    ).reset_index()
    incomplete = [label for label in sorted(labels) if wide[label].isna().any()]
    if incomplete:
        raise ValueError(f"{slug}: {incomplete} do not predict every record; populations must match to share a table")

    ordered = sorted(labels, key=lambda lbl: (MODEL_ORDER.index(lbl.split("@")[0]), lbl))
    return wide, ordered, budgets


def _load_retrieval(path: Path, prediction_col: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "record_ordinal" not in frame:
        frame["record_ordinal"] = frame.groupby("state", sort=False).cumcount()

    required = {"state", "record_ordinal", "fold", "cct_true", "covered", prediction_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")
    if frame.duplicated(["state", "record_ordinal"]).any():
        raise ValueError(f"{path.name}: duplicate record keys")
    return frame[list(required)].rename(columns={prediction_col: "retrieval"})


def _summary_row(
    frame: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    population: str,
    prediction_col: str,
    coverage: float,
    max_features: str,
) -> dict[str, Any]:
    metrics = regression_metrics(
        frame["cct_true"].to_numpy(dtype=float),
        frame[prediction_col].to_numpy(dtype=float),
        coverage=coverage,
    )
    per_fold_mae = (
        frame.assign(absolute_error=(frame["cct_true"] - frame[prediction_col]).abs())
        .groupby("fold", observed=True)["absolute_error"]
        .mean()
    )
    return {
        "dataset": dataset,
        "model": model,
        "population": population,
        "n": len(frame),
        "coverage": coverage,
        "max_features": max_features,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "ae_q50": metrics["ae_q50"],
        "ae_q95": metrics["ae_q95"],
        "ae_q99": metrics["ae_q99"],
        "ae_q100": metrics["ae_q100"],
        "fold_mae_mean": float(per_fold_mae.mean()),
        "fold_mae_std": float(per_fold_mae.std()),
    }


def compare_dataset(dataset: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    ml, models, budgets = _load_ml(config["slug"])
    retrieval = _load_retrieval(config["retrieval_path"], config["retrieval_prediction"])
    merged = retrieval.merge(
        ml,
        on=["state", "record_ordinal"],
        how="inner",
        validate="one_to_one",
        suffixes=("_retrieval", "_ml"),
    )
    if len(merged) != len(retrieval) or len(merged) != len(ml):
        raise ValueError(f"{dataset}: record populations differ: retrieval={len(retrieval)}, ml={len(ml)}")
    if not np.allclose(merged["cct_true_retrieval"], merged["cct_true_ml"], rtol=0.0, atol=1e-12):
        raise ValueError(f"{dataset}: CCT targets differ after record-key join")
    if not (merged["fold_retrieval"] == merged["fold_ml"]).all():
        raise ValueError(f"{dataset}: fold assignments differ after record-key join")

    merged = merged.rename(columns={"cct_true_retrieval": "cct_true", "fold_retrieval": "fold"})
    covered = merged["covered"] & merged["retrieval"].notna()
    common = merged.loc[covered].copy()
    coverage = float(len(common) / len(merged))

    def _mf(model: str) -> str:
        return budgets[model]

    rows = [
        _summary_row(
            common,
            dataset=dataset,
            model="retrieval_highest_support",
            population="common_retrieval_covered",
            prediction_col="retrieval",
            coverage=coverage,
            max_features="not applicable",
        )
    ]
    rows += [
        _summary_row(
            common,
            dataset=dataset,
            model=model,
            population="common_retrieval_covered",
            prediction_col=model,
            coverage=coverage,
            max_features=_mf(model),
        )
        for model in models
    ]
    rows += [
        _summary_row(
            merged,
            dataset=dataset,
            model=model,
            population="all_records",
            prediction_col=model,
            coverage=1.0,
            max_features=_mf(model),
        )
        for model in models
    ]
    return rows


def main() -> None:
    rows = [row for dataset, config in CONFIGS.items() for row in compare_dataset(dataset, config)]
    output = pd.DataFrame(rows)
    output.to_csv(OUT_PATH, index=False)
    print(output.to_string(index=False))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
