"""State-clustered significance test and tail-quantile check for the ELES topology
with/without-filter accuracy ablation (see scripts/service/eles_benchmark.py and
datasets/eles/2026-06/README.md's "Topology Matching Accuracy Ablation" section).

The original ablation (paper.tex Section 6.2) reported a paired Wilcoxon signed-rank test
over the 13,856 matched records treating them as independent observations, even though they
nest inside only 300 independently sampled query states (~46 records/state) - the same
clustering the paper's own state-resampling bootstrap (Section 6.3, scripts/service/
bootstrap_risk_coverage.py) exists specifically to respect elsewhere. This script:

1. Recomputes the significance test at the correct unit of analysis: a paired test on the
   300 per-state mean paired differences, plus a state-resampling bootstrap CI on the mean
   paired MAE/RMSE difference (same convention as bootstrap_risk_coverage.py).
2. Reports the Wilcoxon/Rosenthal standardized effect size r = z/sqrt(n) alongside the
   p-value (not the matched-pairs rank-biserial correlation - a different statistic).
3. Reports Q90/Q95/Q99/max absolute error for both arms on the same matched record set, to
   test the specific prediction implied by framing topology matching as a physical-validity
   safeguard: if it suppresses admission of physically incompatible neighbors, tail error
   (not necessarily central tendency) should differ between arms.

Run from repository root:
    uv run python scripts/service/eles_topology_ablation_significance.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.benchmarking import signed_wilcoxon_z
from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
# ELES_ABLATION_SAMPLE_SEED lets this recompute the same significance/tail-quantile check
# against a different ablation sample seed's joblib pair (see scripts/service/
# eles_benchmark.py's ELES_BENCHMARK_SAMPLE_SEED), for a multi-seed robustness check on the
# original (seed 42) finding - not just a single-draw result.
_ABLATION_SEED = os.environ.get("ELES_ABLATION_SAMPLE_SEED")
_SEED_SUFFIX = f"-seed{_ABLATION_SEED}" if _ABLATION_SEED else ""
LINES_ONLY_PATH: Final[Path] = TMP_DIR / f"report-eles-2026-06-lines_only-sample300{_SEED_SUFFIX}.joblib"
SLOVENIA_ONLY_PATH: Final[Path] = TMP_DIR / f"report-eles-2026-06-slovenia_only-sample300{_SEED_SUFFIX}.joblib"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / f"eles_topology_ablation_significance{_SEED_SUFFIX}.csv"

N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5


def _matched_frame() -> pd.DataFrame:
    wf = pd.DataFrame(joblib.load(LINES_ONLY_PATH))
    nf = pd.DataFrame(joblib.load(SLOVENIA_ONLY_PATH))

    key = ["state", "crit_gen_true", "location_true"]
    wf["_occ"] = wf.groupby(key).cumcount()
    nf["_occ"] = nf.groupby(key).cumcount()
    key2 = key + ["_occ"]
    wf = wf.set_index(key2)
    nf = nf.set_index(key2)
    assert wf.index.is_unique and nf.index.is_unique

    wf_cov = wf.dropna(subset=["cct_weighted_per_location"])
    nf_cov = nf.dropna(subset=["cct_weighted_per_location"])
    common = wf_cov.index.intersection(nf_cov.index)

    wf_c = wf_cov.loc[common].reset_index()
    nf_c = nf_cov.loc[common].reset_index()
    assert (wf_c["cct_true"].to_numpy() == nf_c["cct_true"].to_numpy()).all()

    out = wf_c[["state", "crit_gen_true", "location_true", "cct_true"]].copy()
    out["err_filter_on"] = (wf_c["cct_true"] - wf_c["cct_weighted_per_location"]).abs()
    out["err_filter_off"] = (nf_c["cct_true"] - nf_c["cct_weighted_per_location"]).abs()
    return out


def main() -> None:
    configure_logging()
    df = _matched_frame()
    logger.info(f"Matched records: {len(df):,} across {df['state'].nunique():,} query states")

    # --- Record-level test as originally reported (for comparison only) ---
    w_record, p_record = stats.wilcoxon(df["err_filter_on"], df["err_filter_off"])
    logger.info(f"Record-level (n={len(df):,}, NOT accounting for clustering): W={w_record:.1f}, p={p_record:.3e}")

    # --- State-level test: paired Wilcoxon on the 300 per-state mean differences ---
    per_state = df.groupby("state").agg(
        err_filter_on=("err_filter_on", "mean"),
        err_filter_off=("err_filter_off", "mean"),
        n_records=("err_filter_on", "size"),
    )
    n_states = len(per_state)
    w_state, p_state = stats.wilcoxon(per_state["err_filter_on"], per_state["err_filter_off"], method="approx")
    # z/sqrt(n) is the Wilcoxon/Rosenthal standardized effect size r, not the matched-pairs
    # rank-biserial correlation r_rb = (W_plus - W_minus)/(W_plus + W_minus) - the two are
    # numerically different statistics; this field and its label were corrected 2026-08-09
    # (Codex review, ai2ai.md) after being reported under the wrong name.
    #
    # z comes from src.benchmarking.signed_wilcoxon_z, not scipy's .zstatistic: under a
    # two-sided test scipy derives z from min(W+, W-), so its sign is always negative and
    # carries no direction. Every seed here happens to point the same way, so the previously
    # reported signs were right by coincidence rather than by construction; a seed favouring
    # the filter would have been reported with the same negative sign as one opposing it.
    # The difference is taken as (off - on) to match mean_paired_diff_off_minus_on below, so
    # a negative z/r means disabling the filter lowers error - the same direction as the mean.
    z_state, n_nonzero = signed_wilcoxon_z((per_state["err_filter_off"] - per_state["err_filter_on"]).to_numpy())
    r_effect = z_state / np.sqrt(n_nonzero)
    logger.info(
        f"State-level (n={n_states}, {n_nonzero} nonzero): W={w_state:.1f}, p={p_state:.3e}, "
        f"z={z_state:.3f}, Wilcoxon effect size r={r_effect:.3f}"
    )

    # --- State-resampling bootstrap CI on the mean paired MAE/RMSE difference ---
    rng = np.random.default_rng(SEED)
    state_ids = per_state.index.to_numpy()
    state_diff_mean = (per_state["err_filter_off"] - per_state["err_filter_on"]).to_numpy()

    boot_mean_diff = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(len(state_ids), size=len(state_ids), replace=True)
        boot_mean_diff[b] = state_diff_mean[sampled].mean()

    ci_low = float(np.percentile(boot_mean_diff, CI_LOW))
    ci_high = float(np.percentile(boot_mean_diff, CI_HIGH))
    point_mean_diff = float(state_diff_mean.mean())
    logger.info(
        f"State-resampling bootstrap ({N_BOOTSTRAP} reps) on mean(err_off - err_on): "
        f"point={point_mean_diff:.5f}, 95% CI=[{ci_low:.5f}, {ci_high:.5f}]"
    )

    # --- Tail-quantile comparison on the same matched records (both arms) ---
    quantiles = [0.90, 0.95, 0.99, 1.0]
    tail_on = df["err_filter_on"].quantile(quantiles)
    tail_off = df["err_filter_off"].quantile(quantiles)
    logger.info("Tail quantiles, filter ON:\n" + tail_on.to_string())
    logger.info("Tail quantiles, filter OFF:\n" + tail_off.to_string())

    rows = [
        {"quantity": "record_level_p_value_original", "value": p_record},
        {"quantity": "n_records", "value": len(df)},
        {"quantity": "n_states", "value": n_states},
        # Per-arm mean absolute error on the matched set. The paired difference below is the
        # inferential quantity, but without the levels a reader cannot judge the effect size
        # against the error it sits on (Reviewer 2, 2026-08-14).
        {"quantity": "mean_err_filter_on", "value": float(df["err_filter_on"].mean())},
        {"quantity": "mean_err_filter_off", "value": float(df["err_filter_off"].mean())},
        {"quantity": "state_level_wilcoxon_W", "value": w_state},
        {"quantity": "state_level_p_value", "value": p_state},
        {"quantity": "state_level_z", "value": z_state},
        {"quantity": "state_level_wilcoxon_effect_size_r", "value": r_effect},
        {"quantity": "mean_paired_diff_off_minus_on_point", "value": point_mean_diff},
        {"quantity": "mean_paired_diff_ci_low", "value": ci_low},
        {"quantity": "mean_paired_diff_ci_high", "value": ci_high},
        {"quantity": "q90_filter_on", "value": tail_on[0.90]},
        {"quantity": "q90_filter_off", "value": tail_off[0.90]},
        {"quantity": "q95_filter_on", "value": tail_on[0.95]},
        {"quantity": "q95_filter_off", "value": tail_off[0.95]},
        {"quantity": "q99_filter_on", "value": tail_on[0.99]},
        {"quantity": "q99_filter_off", "value": tail_off[0.99]},
        {"quantity": "max_filter_on", "value": tail_on[1.0]},
        {"quantity": "max_filter_off", "value": tail_off[1.0]},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
