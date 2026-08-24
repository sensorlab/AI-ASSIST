"""Compares screening methods at matched referral load on ELES.

The main text scores the retrieval screen against no-skill referral, which shows the screen
beats picking states at random for the same workload but not whether a supervised model would
do better at that workload. This closes that gap: each method is put at retrieval's referral
load, so recall can be read at equal cost rather than at each method's own operating point.

A state is unsafe when the smallest true CCT among its evaluated contingencies falls below the
threshold, and a screen refers a state when the smallest value it produces for that state falls
below its own cut. Retrieval carries one extra constraint the supervised models do not: where
any contingency of a state has no compatible support it abstains, and an abstention must be
referred, so retrieval cannot operate below the referral load those abstentions already force.
That floor is a property of the archive, not of the cut, and it is the quantity the comparison
is really about.

Threshold selection is cross-fitted. Retrieval sits at the pre-specified tau and selects
nothing. A comparator's cut is chosen on the folds a state does not belong to and then applied
unchanged to that state, so no cut is ever picked by looking at the outcomes it is scored
against. An earlier version swept cuts on the scored population and kept the best recall under
the load constraint, which handed the comparator a post-hoc operating point while retrieval
kept a fixed one, and biased the comparison toward the conclusion drawn from it.

Uncertainty is a paired bootstrap on the recall difference, since both arms score the same
states. It is reported clustered by state and again by topology group: states sharing a
switching configuration arrive in short bursts and retrieve each other, so they are not
independent, and the group is the more conservative unit.

Reported at the paper's operating threshold of 200 ms, on the five-fold population Table 3 uses
so that retrieval and the supervised models see the same records.

Run from the repository root:
    uv run python scripts/paper/screening_matched_workload.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "screening_matched_workload.csv"
RETRIEVAL: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_group_kfold.parquet"
SUPERVISED: Final[Path] = TMP_DIR / "ml_benchmark_predictions-eles-2026-06.parquet"
TAU: Final[float] = 0.20
# The location median takes only a handful of distinct values, so its attainable loads are a few
# coarse steps and it cannot be placed at an arbitrary workload; reported for completeness.
MODELS: Final[tuple[str, ...]] = ("hist_gradient_boosting", "random_forest", "location_median")
N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42


def _state_scores(df: pd.DataFrame, pred: str) -> pd.DataFrame:
    """Per state: the smallest predicted CCT, the smallest true CCT, and whether any is missing."""
    g = df.groupby("state", sort=False)
    return pd.DataFrame(
        {
            "score": g[pred].min(),
            "true_min": g["cct_true"].min(),
            "any_missing": g[pred].apply(lambda s: bool(s.isna().any())),
            "fold": g["fold"].first(),
        }
    )


def _cut_for_load(scores: np.ndarray, target_load: float) -> float:
    """Largest cut whose referral rate on `scores` does not exceed `target_load`."""
    if len(scores) == 0:
        return -np.inf
    return float(np.quantile(scores, min(max(target_load, 0.0), 1.0), method="lower"))


def _crossfit_referred(ss: pd.DataFrame, target_load: float) -> np.ndarray:
    """Refer decisions where each state's cut came from the folds it does not belong to."""
    referred = np.zeros(len(ss), dtype=bool)
    for f in sorted(ss["fold"].unique()):
        held = (ss["fold"] == f).to_numpy()
        cut = _cut_for_load(ss.loc[~held, "score"].to_numpy(), target_load)
        referred[held] = ss.loc[held, "score"].to_numpy() < cut
    return referred


def _paired_ci(
    unsafe: np.ndarray, a: np.ndarray, b: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile CI on recall(a) - recall(b), resampling whole clusters of states."""
    codes = pd.factorize(clusters, sort=False)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    parts = np.split(order, bounds)
    draws = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        pick = np.concatenate([parts[j] for j in rng.integers(0, len(parts), len(parts))])
        u = unsafe[pick]
        n_u = int(u.sum())
        draws[i] = np.nan if n_u == 0 else (a[pick] & u).sum() / n_u - (b[pick] & u).sum() / n_u
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main() -> None:
    configure_logging()
    r = pd.read_parquet(RETRIEVAL)
    r["state"] = r["state"].astype(str)
    r["pred"] = r["cct_weighted_per_location"].where(r["covered"])
    rs = _state_scores(r, "pred")
    forced = rs["any_missing"]
    floor = float(forced.mean())
    logger.info(f"{len(rs):,} ELES states; retrieval must refer {floor:.2%} of them for want of an estimate")

    # Topology group per state, as the conservative clustering unit.
    lf = pd.read_pickle(DATASET_DIR / "interim/lf.pkl")
    lines = [c for c in json.loads((DATASET_DIR / "processed/topology_cols_lines_only.json").read_text()) if c in lf]
    gmap = dict(
        zip((str(i) for i in lf.index), pd.factorize(lf[lines].astype(str).agg("|".join, axis=1))[0], strict=True)
    )

    m = pd.read_parquet(SUPERVISED)
    m["state"] = m["state"].astype(str)

    unsafe = (rs["true_min"] < TAU).to_numpy()
    unsafe_n = int(unsafe.sum())
    ret_ref = (forced | (rs["score"] < TAU)).to_numpy()
    ref_at_tau = float(ret_ref.mean())
    rec_at_tau = float((ret_ref & unsafe).sum() / unsafe_n)
    logger.info(f"retrieval at tau: refers {ref_at_tau:.2%}, recall {rec_at_tau:.2%} ({unsafe_n:,} unsafe states)")
    logger.info(f"perfect-screen floor (share of states genuinely unsafe): {unsafe.mean():.2%}")

    rng = np.random.default_rng(SEED)
    st_cl = np.asarray(rs.index, dtype=str)
    gp_cl = np.array([gmap.get(s, -1) for s in st_cl])

    rows = [{"method": "retrieval", "referral_load": ref_at_tau, "recall": rec_at_tau, "at": "tau=200ms"}]
    for model in MODELS:
        sub = m[m["model"] == model].rename(columns={"cct_pred": "pred"})
        ss = _state_scores(sub, "pred").reindex(rs.index)
        if ss["score"].isna().any():
            raise ValueError(f"{model}: does not cover every state; populations must match")
        referred = _crossfit_referred(ss, ref_at_tau)
        load, recall = float(referred.mean()), float((referred & unsafe).sum() / unsafe_n)
        lo_s, hi_s = _paired_ci(unsafe, ret_ref, referred, st_cl, rng)
        lo_g, hi_g = _paired_ci(unsafe, ret_ref, referred, gp_cl, rng)
        logger.info(
            f"{model}: cross-fitted to retrieval's load -> refers {load:.2%}, recall {recall:.2%}; "
            f"retrieval minus {model} = {rec_at_tau - recall:+.2%} "
            f"(state-clustered CI {lo_s:+.2%},{hi_s:+.2%}; group-clustered CI {lo_g:+.2%},{hi_g:+.2%})"
        )
        rows += [
            {
                "method": model,
                "referral_load": load,
                "recall": recall,
                "at": "cross-fitted to retrieval load",
                "recall_diff_retrieval_minus_model": rec_at_tau - recall,
                "diff_ci_low_state": lo_s,
                "diff_ci_high_state": hi_s,
                "diff_ci_low_group": lo_g,
                "diff_ci_high_group": hi_g,
            },
            {
                "method": model,
                "referral_load": float((ss["score"] < TAU).mean()),
                "recall": float(((ss["score"] < TAU).to_numpy() & unsafe).sum() / unsafe_n),
                "at": "tau=200ms",
            },
        ]

    rows += [
        {"method": "retrieval_abstention_floor", "referral_load": floor, "at": "forced"},
        {"method": "perfect_screen_floor", "referral_load": float(unsafe.mean()), "at": "share genuinely unsafe"},
        {"method": "n_states", "referral_load": float(len(rs)), "recall": float(unsafe_n), "at": "unsafe count"},
    ]
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
