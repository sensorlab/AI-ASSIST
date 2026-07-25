"""Connects AI-ASSIST's regression error to the 100-200 ms protection-clearing-time band
used in paper.tex's Introduction to motivate CCT estimation, rather than leaving the
reported MAE/RMSE/percentile numbers disconnected from that operational threshold.

For each covered record, checks whether the true CCT and the predicted CCT fall on
opposite sides of a representative clearing-time boundary tau - i.e., whether the
estimation error alone would flip a criticality classification at that boundary,
regardless of the error's absolute magnitude elsewhere in the CCT range. Reports this for
tau in {0.10s, 0.15s} on both BUS39 (5-fold covered records) and the full-population ELES
lines_only benchmark - both already-collected artifacts, no new experiments.

Run from repository root:
    uv run python scripts/service/clearing_time_threshold_crossing.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
BUS39_PATH: Final[Path] = PROJECT_DIR / "report-service-group-kfold.joblib"
ELES_PATH: Final[Path] = PROJECT_DIR / "report-eles-2026-06-lines_only.joblib"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "clearing_time_threshold_crossing.csv"

THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15)


def _flip_rate(cct_true: np.ndarray, cct_pred: np.ndarray, tau: float) -> float:
    true_side = cct_true >= tau
    pred_side = cct_pred >= tau
    return float((true_side != pred_side).mean())


def _conditional_flip_rate(cct_true: np.ndarray, cct_pred: np.ndarray, tau: float, band: float) -> tuple[float, int]:
    """Flip rate restricted to records whose true CCT is within `band` of `tau` - the
    subset where classification error is actually a live possibility, as opposed to the
    unconditional rate which is dominated by records far from the boundary either way."""
    near = np.abs(cct_true - tau) <= band
    if near.sum() == 0:
        return float("nan"), 0
    true_side = cct_true[near] >= tau
    pred_side = cct_pred[near] >= tau
    return float((true_side != pred_side).mean()), int(near.sum())


def _bus39_frame() -> pd.DataFrame:
    payload = joblib.load(BUS39_PATH)
    df = pd.DataFrame(payload["predictions"])
    df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    return df


def _eles_frame() -> pd.DataFrame:
    rows = joblib.load(ELES_PATH)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    return df


def main() -> None:
    configure_logging()

    bus39 = _bus39_frame()
    eles = _eles_frame()
    logger.info(f"BUS39 covered records: {len(bus39):,}")
    logger.info(f"ELES covered records: {len(eles):,}")

    rows: list[dict[str, object]] = []
    for name, df in (("BUS39", bus39), ("ELES", eles)):
        cct_true = df["cct_true"].to_numpy(dtype=np.float64)
        cct_pred = df["cct_weighted_per_location"].to_numpy(dtype=np.float64)
        for tau in THRESHOLDS:
            rate = _flip_rate(cct_true, cct_pred, tau)
            near_frac = float((np.abs(cct_true - tau) <= 0.05).mean())
            cond_rate, n_near = _conditional_flip_rate(cct_true, cct_pred, tau, band=0.05)
            rows.append(
                {
                    "dataset": name,
                    "tau_s": tau,
                    "flip_rate_unconditional": rate,
                    "n": len(df),
                    "frac_true_within_50ms_of_tau": near_frac,
                    "n_near_tau": n_near,
                    "flip_rate_conditional_on_near_tau": cond_rate,
                }
            )
            logger.info(
                f"{name}, tau={tau:.2f}s: unconditional flip rate {rate:.4%}; "
                f"{near_frac:.4%} of records ({n_near:,}) have true CCT within 50ms of tau, "
                f"of which {cond_rate:.4%} flip classification"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
