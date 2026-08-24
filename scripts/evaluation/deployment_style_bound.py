"""Coarse, partially de-oracled accuracy bound for the BUS39 5-fold benchmark
(paper.tex Section 5.2/6.1), computed entirely from the already-collected
report-service-group-kfold.joblib artifact - no rerun needed.

The paper's headline scalar comparison (Table results_baseline) scores AI-ASSIST on the
oracle-selected true (critical generator, location) slice - an outcome the supervised
baselines never see. The paper states the deployment-style gap "can only be larger" but
never computes it. Full de-oracling (also predicting the critical generator) would need a
new benchmark run that additionally persists every retrieved generator group's report, not
only the one matching the true generator, and is out of scope here.

This script computes a coarser, still-honest step in that direction: LOCATION de-oracling
only. Each retained record's ReportSummary already carries `stats.location_weight_mass`,
the retrieval method's own weight mass across every location retrieved WITHIN the true
generator's group (i.e., the group is still oracle-selected, but the location is not).
Scoring the location with the highest weight mass instead of the true location gives an
accuracy figure that no longer uses the true location as an input, while still using the
true critical generator - a partial, not full, deployment-style bound.

Run from repository root:
    uv run python scripts/evaluation/deployment_style_bound.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
# NOTE: this script is stale and silently broken against the current schema - see
# paper-sr/EXPERIMENTS.md 2026-08-05 - it reads flat ps.location_weight_mass/
# ps.cct_weighted_per_location attributes that predate the 2026-07-30 TSA report-model rework
# and no longer exist, and it's superseded by full_deoracled_bound.py's results anyway.
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
BUS39_PATH: Final[Path] = TMP_DIR / "report-service-group-kfold.joblib"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "deployment_style_bound.csv"


def _argmax_location_prediction(record: dict[str, Any]) -> float | None:
    ps = record.get("prediction_summary")
    if ps is None:
        return None
    # report-service-group-kfold.joblib predates the ReportSummary.stats nested-model
    # refactor - this pickled artifact has location_weight_mass as a flat top-level
    # attribute (restored directly into __dict__ on unpickling), not ps.stats.*.
    weight_mass = getattr(ps, "location_weight_mass", None)
    if not weight_mass:
        return None
    best_location = max(weight_mass, key=weight_mass.get)
    return ps.cct_weighted_per_location.get(best_location)


def main() -> None:
    configure_logging()
    payload = joblib.load(BUS39_PATH)
    predictions: list[dict[str, Any]] = payload["predictions"]
    logger.info(f"Loaded {len(predictions):,} BUS39 5-fold records")

    rows: list[dict[str, Any]] = []
    for record in predictions:
        cct_true = record.get("cct_true")
        oracle_pred = record.get("cct_weighted_per_location")
        if cct_true is None or oracle_pred is None:
            continue
        deoracled_pred = _argmax_location_prediction(record)
        if deoracled_pred is None:
            continue
        rows.append(
            {
                "cct_true": cct_true,
                "oracle_pred": oracle_pred,
                "deoracled_pred": deoracled_pred,
                "location_true": record.get("location_true"),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"Records with both oracle and location-de-oracled predictions: {len(df):,}")

    err_oracle = (df["cct_true"] - df["oracle_pred"]).abs().to_numpy()
    err_deoracled = (df["cct_true"] - df["deoracled_pred"]).abs().to_numpy()

    mae_oracle = float(err_oracle.mean())
    rmse_oracle = float(np.sqrt((err_oracle**2).mean()))
    mae_deoracled = float(err_deoracled.mean())
    rmse_deoracled = float(np.sqrt((err_deoracled**2).mean()))

    logger.info(f"Oracle location:      MAE={mae_oracle:.4f}s RMSE={rmse_oracle:.4f}s")
    logger.info(f"De-oracled location:  MAE={mae_deoracled:.4f}s RMSE={rmse_deoracled:.4f}s")

    out = pd.DataFrame(
        [
            {"variant": "oracle_location_and_generator", "mae": mae_oracle, "rmse": rmse_oracle, "n": len(df)},
            {
                "variant": "deoracled_location_oracle_generator",
                "mae": mae_deoracled,
                "rmse": rmse_deoracled,
                "n": len(df),
            },
        ]
    )
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
