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
states referred) under a conservative deployment rule: a state is referred when its minimum
predicted CCT is below tau *or* any of its simulated contingency records lacks an estimate.
Keeping the missing records in the state aggregation is essential; dropping them would report
screening performance only on the subset for which retrieval already succeeded.

Fixed 2026-08-09 (peer review caught this): both flip-rate and screening previously read
`cct_weighted_per_location` from benchmark.py's per-record reports, which is the *oracle*
prediction - `outputs_by_crit_gen.get(crit_gen_true)`, i.e. it already assumes the true
critical generator is known before dispatching the query. That assumption doesn't hold for a
genuine pre-fault screen (which generator becomes critical is an outcome of the fault, not an
input to it), and it's exactly what generator_deoracled_bound.py/eles_deoracled_bound.py's
"screening_min" variant is built to avoid: the retrieval-visible candidate generators at that
location are enumerated without needing to know which one is correct, and the minimum of
their estimates is used as the conservative, generator-agnostic CCT bound. This script now
reads those per-record deoracled artifacts instead of benchmark.py's oracle-conditioned
report, so both coverage and screening reflect only information available before the fault.

Run from repository root:
    uv run python scripts/evaluation/clearing_time_threshold_crossing.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

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
# BUS39 leave-one-state-out, matching the ELES arm. The grouped-fold artifact
# (no _loso suffix) is retained for the matched supervised comparison in Table 2, which
# cannot use leave-one-state-out because a regressor would need one refit per state.
BUS39_PATH: Final[Path] = TMP_DIR / "generator_deoracled_records_loso.parquet"
ELES_PATH: Final[Path] = TMP_DIR / "eles_deoracled_records_eles-2026-06_lines_only.parquet"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "clearing_time_threshold_crossing.csv"

# 0.20 added (2026-08-05) - the paper's primary operational threshold (DIRECTION.md Decisions:
# 200 ms, the worst-case bound of the 100-200 ms literature range, matching ELES partner practice).
# 0.10/0.15 kept as sensitivity bounds.
THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)

# Bootstrap 95% CIs for the screening metrics (added 2026-08-08, per peer review: Table 3
# reported point estimates only while Table 2's diagnostics already carry CIs). Same
# convention as bootstrap_risk_coverage.py/bootstrap_deoracling_ci.py: state-level resampling
# (each state's true/predicted-unsafe label already the resampling unit here, since
# _state_screening_labels has already aggregated to one row per state), 2,000 resamples, seed
# 42, 2.5/97.5 percentiles.
N_BOOTSTRAP: Final[int] = int(os.environ.get("BOOTSTRAP_SCREENING_N", "2000"))
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


def _state_screening_labels(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Return one screening row per state without discarding missing estimates.

    A state is truly unsafe if any simulated contingency is below ``tau``. It is referred
    for simulation if either (a) the minimum available location-specific estimate is below
    ``tau`` or (b) at least one simulated contingency lacks an estimate. The latter is the
    fail-visible behavior claimed by the paper: an abstention must become operational work,
    not disappear from the evaluation denominator.
    """
    g = df.groupby("state")
    n_records = g.size()
    n_estimates = g["cct_weighted_per_location"].count()
    true_unsafe = g["cct_true"].min() < tau
    threshold_flag = (g["cct_weighted_per_location"].min() < tau).fillna(False)
    complete_coverage = n_estimates.eq(n_records)
    coverage_referral = ~complete_coverage
    referred = threshold_flag | coverage_referral

    return pd.DataFrame(
        {
            "true_unsafe": true_unsafe,
            "threshold_flag": threshold_flag,
            "coverage_referral": coverage_referral,
            "referred": referred,
            "any_estimate": n_estimates.gt(0),
            "complete_coverage": complete_coverage,
            "n_records": n_records,
            "n_estimates": n_estimates,
        }
    )


def _state_screening_metrics(labels: pd.DataFrame) -> dict[str, float]:
    true_unsafe = labels["true_unsafe"]
    referred = labels["referred"]

    tp = int((true_unsafe & referred).sum())
    fp = int((~true_unsafe & referred).sum())
    tn = int((~true_unsafe & ~referred).sum())
    n_states = len(true_unsafe)
    n_unsafe = int(true_unsafe.sum())

    return {
        "n_states": n_states,
        "n_unsafe_states": n_unsafe,
        "screening_recall": tp / n_unsafe if n_unsafe > 0 else float("nan"),
        "screening_false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "screening_referral_load": (tp + fp) / n_states if n_states > 0 else float("nan"),
        "threshold_flag_load": float(labels["threshold_flag"].mean()),
        "coverage_referral_load": float(labels["coverage_referral"].mean()),
        "state_any_estimate_rate": float(labels["any_estimate"].mean()),
        "state_complete_coverage_rate": float(labels["complete_coverage"].mean()),
        "n_states_any_estimate": int(labels["any_estimate"].sum()),
        "n_states_complete_coverage": int(labels["complete_coverage"].sum()),
        # Disjoint decomposition of the two referral causes (added 2026-08-09, peer review):
        # threshold_flag_load + coverage_referral_load double-counts states referred for both
        # reasons, so those two rates are not additive shares of screening_referral_load. These
        # four categories partition the state population exactly (sum to 1.0) and are the
        # correct basis for "which cause drives referral" claims.
        "referral_threshold_only_load": float((labels["threshold_flag"] & ~labels["coverage_referral"]).mean()),
        "referral_coverage_only_load": float((~labels["threshold_flag"] & labels["coverage_referral"]).mean()),
        "referral_both_load": float((labels["threshold_flag"] & labels["coverage_referral"]).mean()),
        "referral_neither_load": float((~labels["threshold_flag"] & ~labels["coverage_referral"]).mean()),
    }


def _bootstrap_screening_ci(labels: pd.DataFrame, *, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap 95% CIs for recall/false-alarm-rate/referral-load, resampling states (with
    replacement) from the already state-level labels."""
    true_arr = labels["true_unsafe"].to_numpy()
    pred_arr = labels["referred"].to_numpy()
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
    # generator_deoracled_bound.py's per-record output: "screening_min" is the minimum
    # estimate across retrieval-visible candidate generators at the true location, requiring
    # no knowledge of which one actually turns out to be critical (see module docstring).
    df = pd.read_parquet(BUS39_PATH)
    return df.rename(columns={"pred_screening_min": "cct_weighted_per_location"})


def _eles_frame() -> pd.DataFrame:
    # eles_deoracled_bound.py's equivalent, location-fixed/generator-deoracled variant.
    df = pd.read_parquet(ELES_PATH)
    return df.rename(columns={"pred_gen_deoracled_screening_min": "cct_weighted_per_location"})


def main() -> None:
    configure_logging()

    bus39 = _bus39_frame()
    eles = _eles_frame()
    logger.info(f"BUS39 records (including abstentions): {len(bus39):,}")
    logger.info(f"ELES records (including abstentions): {len(eles):,}")

    rng = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    for name, df in (("BUS39", bus39), ("ELES", eles)):
        covered = df.dropna(subset=["cct_weighted_per_location"])
        cct_true = covered["cct_true"].to_numpy(dtype=np.float64)
        cct_pred = covered["cct_weighted_per_location"].to_numpy(dtype=np.float64)
        for tau in THRESHOLDS:
            rate = _flip_rate(cct_true, cct_pred, tau)
            near_frac = float((np.abs(cct_true - tau) <= 0.05).mean())
            cond_rate, n_near = _conditional_flip_rate(cct_true, cct_pred, tau, band=0.05)
            labels = _state_screening_labels(df, tau)
            screening = _state_screening_metrics(labels)
            screening_ci = _bootstrap_screening_ci(labels, rng=rng)

            rows.append(
                {
                    "dataset": name,
                    "tau_s": tau,
                    "flip_rate_unconditional": rate,
                    "n_records_total": len(df),
                    "n_records_estimated": len(covered),
                    "record_coverage_rate": len(covered) / len(df),
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
