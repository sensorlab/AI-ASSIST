"""Connects AI-ASSIST's regression error to the 100-200 ms protection-clearing-time band
used in paper.tex's Introduction to motivate CCT estimation, rather than leaving the
reported MAE/RMSE/percentile numbers disconnected from that operational threshold.

For each covered record, checks whether the true CCT and the predicted CCT fall on
opposite sides of a representative clearing-time boundary tau - i.e., whether the
estimation error alone would flip a criticality classification at that boundary,
regardless of the error's absolute magnitude elsewhere in the CCT range. Reports this for
tau in {0.10, 0.15, 0.20}s on both BUS39 (5-fold covered records) and the full-population
ELES lines_only benchmark - both already-collected artifacts, no new experiments.

Also computes the state-level screening rule (added 2026-08-05, was missing entirely): the
flip rate above is *not* the quantity a deployed screen is actually judged on - a screen
enumerates candidate locations for a state, takes the worst-case (minimum) predicted CCT
across them, and flags the state for detailed simulation if that minimum falls below tau.
Reports recall of truly-unsafe states, false-alarm rate, and referral load (fraction of all
states flagged) under that rule - this is Contribution 4 in paper-sr/DIRECTION.md.

Run from repository root:
    uv run python scripts/service/clearing_time_threshold_crossing.py
"""

from __future__ import annotations

import logging
import os
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
BUS39_PATH: Final[Path] = TMP_DIR / "report-service-group-kfold.joblib"
ELES_PATH: Final[Path] = TMP_DIR / "report-eles-2026-06-lines_only.joblib"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "clearing_time_threshold_crossing.csv"

# 0.20 added (2026-08-05) - the paper's primary operational threshold (DIRECTION.md Decisions:
# 200 ms, the worst-case bound of the 100-200 ms literature range, matching ELES partner practice).
# 0.10/0.15 kept as sensitivity bounds.
THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)

# Bootstrap 95% CIs for the screening metrics (added 2026-08-08, per peer review: Table 3
# reported point estimates only while Table 2's diagnostics already carry CIs). Same
# convention as bootstrap_risk_coverage.py/bootstrap_deoracling_ci.py: state-level resampling
# (each state's true/predicted-unsafe label already the resampling unit here, since
# _state_screening_metrics has already aggregated to one row per state), 200 resamples, seed
# 42, 2.5/97.5 percentiles.
N_BOOTSTRAP: Final[int] = int(os.environ.get("BOOTSTRAP_SCREENING_N", "200"))
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5


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


def _state_screening_metrics(df: pd.DataFrame, tau: float) -> dict[str, float]:
    """State-level screening rule: flag a state unsafe if the minimum predicted CCT among
    its covered records falls below tau (a screen enumerates candidate locations and takes
    the worst case); the true label is unsafe if any of its records' true CCT falls below
    tau. This is the operationally relevant quantity, unlike the record-level flip rate
    above - see the module docstring."""
    g = df.groupby("state")
    true_unsafe = g["cct_true"].min() < tau
    pred_unsafe = g["cct_weighted_per_location"].min() < tau

    tp = int((true_unsafe & pred_unsafe).sum())
    fp = int((~true_unsafe & pred_unsafe).sum())
    tn = int((~true_unsafe & ~pred_unsafe).sum())
    n_states = len(true_unsafe)
    n_unsafe = int(true_unsafe.sum())

    return {
        "n_states": n_states,
        "n_unsafe_states": n_unsafe,
        "screening_recall": tp / n_unsafe if n_unsafe > 0 else float("nan"),
        "screening_false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "screening_referral_load": (tp + fp) / n_states if n_states > 0 else float("nan"),
    }


def _bootstrap_screening_ci(
    true_unsafe: pd.Series, pred_unsafe: pd.Series, *, rng: np.random.Generator
) -> dict[str, float]:
    """Bootstrap 95% CIs for recall/false-alarm-rate/referral-load, resampling states (with
    replacement) from the already state-level true_unsafe/pred_unsafe boolean series - each
    already has exactly one row per state (from the groupby in _state_screening_metrics), so
    a plain positional resample is a state-level bootstrap, no state->indices map needed."""
    true_arr = true_unsafe.to_numpy()
    pred_arr = pred_unsafe.to_numpy()
    n = len(true_arr)

    recall_samples = np.full(N_BOOTSTRAP, np.nan)
    fa_samples = np.full(N_BOOTSTRAP, np.nan)
    referral_samples = np.full(N_BOOTSTRAP, np.nan)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        t = true_arr[idx]
        p = pred_arr[idx]
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        tn = int((~t & ~p).sum())
        n_unsafe = int(t.sum())
        if n_unsafe > 0:
            recall_samples[b] = tp / n_unsafe
        if (fp + tn) > 0:
            fa_samples[b] = fp / (fp + tn)
        referral_samples[b] = (tp + fp) / n

    def _ci(samples: np.ndarray) -> tuple[float, float]:
        valid = samples[~np.isnan(samples)]
        if len(valid) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(valid, CI_LOW)), float(np.percentile(valid, CI_HIGH))

    recall_lo, recall_hi = _ci(recall_samples)
    fa_lo, fa_hi = _ci(fa_samples)
    referral_lo, referral_hi = _ci(referral_samples)
    return {
        "screening_recall_ci_low": recall_lo,
        "screening_recall_ci_high": recall_hi,
        "screening_false_alarm_rate_ci_low": fa_lo,
        "screening_false_alarm_rate_ci_high": fa_hi,
        "screening_referral_load_ci_low": referral_lo,
        "screening_referral_load_ci_high": referral_hi,
    }


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

    rng = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    for name, df in (("BUS39", bus39), ("ELES", eles)):
        cct_true = df["cct_true"].to_numpy(dtype=np.float64)
        cct_pred = df["cct_weighted_per_location"].to_numpy(dtype=np.float64)
        for tau in THRESHOLDS:
            rate = _flip_rate(cct_true, cct_pred, tau)
            near_frac = float((np.abs(cct_true - tau) <= 0.05).mean())
            cond_rate, n_near = _conditional_flip_rate(cct_true, cct_pred, tau, band=0.05)
            screening = _state_screening_metrics(df, tau)

            g = df.groupby("state")
            true_unsafe = g["cct_true"].min() < tau
            pred_unsafe = g["cct_weighted_per_location"].min() < tau
            screening_ci = _bootstrap_screening_ci(true_unsafe, pred_unsafe, rng=rng)

            rows.append(
                {
                    "dataset": name,
                    "tau_s": tau,
                    "flip_rate_unconditional": rate,
                    "n": len(df),
                    "frac_true_within_50ms_of_tau": near_frac,
                    "n_near_tau": n_near,
                    "flip_rate_conditional_on_near_tau": cond_rate,
                    **screening,
                    **screening_ci,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
            logger.info(
                f"{name}, tau={tau:.2f}s: unconditional flip rate {rate:.4%}; "
                f"{near_frac:.4%} of records ({n_near:,}) have true CCT within 50ms of tau, "
                f"of which {cond_rate:.4%} flip classification"
            )
            logger.info(
                f"{name}, tau={tau:.2f}s screening: "
                f"recall={screening['screening_recall']:.4%} "
                f"(95% CI {screening_ci['screening_recall_ci_low']:.4%}-{screening_ci['screening_recall_ci_high']:.4%}), "
                f"false_alarm_rate={screening['screening_false_alarm_rate']:.4%} "
                f"(95% CI {screening_ci['screening_false_alarm_rate_ci_low']:.4%}-"
                f"{screening_ci['screening_false_alarm_rate_ci_high']:.4%}), "
                f"referral_load={screening['screening_referral_load']:.4%} "
                f"(95% CI {screening_ci['screening_referral_load_ci_low']:.4%}-"
                f"{screening_ci['screening_referral_load_ci_high']:.4%}) "
                f"({screening['n_unsafe_states']:,}/{screening['n_states']:,} states truly unsafe)"
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
