"""Referral decomposition: does archive coverage constrain screening more than conditional accuracy? (R2.1b)

Pre-specified in the manuscript repository's revision/analyses/01-referral-decomposition-and-frontier.md, committed
before any number here was computed. The screening rule is the paper's: a state is referred when any recorded
contingency lacks an estimate (abstention) or its minimum screening estimate is below tau (threshold); it is unsafe when
its minimum true CCT is below tau.

C1, static share (retrospective, BUS39 and ELES): among referred states, the share referred because of abstention
(abstention only plus both), next to the share of states observed unsafe, which full recall must refer anyway.

C2, attribution of the causal change (ELES): moving from retrospective (leave-one-state-out) to causal availability
(lag 0) changes the referral load R. Every causally covered record is also covered retrospectively, so the change splits
sequentially:

    dR_coverage = R(causal coverage, retrospective estimates) - R(retrospective)
    dR_error    = R(causal) - R(causal coverage, retrospective estimates)

The reverse order is undefined because records that lose support have no causal estimate. Coverage loss can only add
referrals, so missed unsafe states (FN) can change only in the error step; dFN_error is reported, never netted.

Decision rule (fixed in advance): the comparative claim is retained only if, at tau = 200 ms, dR_coverage - dR_error > 0
with its 95% interval excluding zero under the most conservative (widest) of the state, topology-group and calendar-day
cluster bootstraps. Calendar week is reported for sensitivity and does not decide. 2,000 resamples, seed 42.

Inputs: tmp/harness_records_exact_lines_{loso,causal0h}.parquet from retrieval_harness_validation.py (which must have
passed), and tmp/generator_deoracled_records_loso.parquet for BUS39.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/referral_decomposition.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_eles_state_timestamps  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
ELES_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
RAW_ZIP: Final[Path] = ELES_DIR / "raw" / "Podatki_DSA.zip"

ELES_LOSO: Final[Path] = TMP_DIR / "harness_records_exact_lines_loso.parquet"
ELES_CAUSAL0: Final[Path] = TMP_DIR / "harness_records_exact_lines_causal0h.parquet"
BUS39_LOSO: Final[Path] = TMP_DIR / "generator_deoracled_records_loso.parquet"
VALIDATION: Final[Path] = PAPER_DATA_DIR / "retrieval_harness_validation.csv"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "referral_decomposition.csv"

THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)
DECISION_TAU: Final[float] = 0.20
DECISION_UNITS: Final[tuple[str, ...]] = ("state", "topology_group", "calendar_day")
SENSITIVITY_UNITS: Final[tuple[str, ...]] = ("calendar_week",)
N_BOOTSTRAP: Final[int] = 2000
SEED: Final[int] = 42


def _state_frame(records: pd.DataFrame, estimate: str, covered: str, tau: float) -> pd.DataFrame:
    """Per state: unsafe, abstention referral, threshold referral, referred. Uncovered records carry no estimate."""
    est = records[estimate].where(records[covered])
    g = records.assign(_est=est).groupby("state", sort=False)
    unsafe = g["cct_true"].min() < tau
    abstain = ~g[covered].all()
    threshold = (g["_est"].min() < tau).fillna(False)
    return pd.DataFrame({"unsafe": unsafe, "abstain": abstain, "threshold": threshold, "referred": abstain | threshold})


# A statistic maps per-state indicator arrays to a number; the cluster bootstrap reweights states, so every statistic
# is written as a ratio of weighted sums.
Stat = Callable[[dict[str, np.ndarray], np.ndarray], float]


def _mean(col: str) -> Stat:
    return lambda a, w: float(np.dot(w, a[col]) / w.sum())


def _ratio(num: str, den: str) -> Stat:
    def f(a: dict[str, np.ndarray], w: np.ndarray) -> float:
        d = float(np.dot(w, a[den]))
        return float(np.dot(w, a[num]) / d) if d > 0 else float("nan")

    return f


def _count(col: str) -> Stat:
    return lambda a, w: float(np.dot(w, a[col]))


def _bootstrap(
    arrays: dict[str, np.ndarray], stats: dict[str, Stat], clusters: np.ndarray | None, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    """Percentile 95% intervals. clusters=None resamples states; otherwise whole clusters (state cluster codes)."""
    n = len(next(iter(arrays.values())))
    codes = np.arange(n) if clusters is None else clusters
    n_clusters = int(codes.max()) + 1
    samples = {name: np.empty(N_BOOTSTRAP) for name in stats}
    for b in range(N_BOOTSTRAP):
        draws = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        w = draws[codes].astype(np.float64)
        for name, stat in stats.items():
            samples[name][b] = stat(arrays, w)
    out = {}
    for name, s in samples.items():
        valid = s[~np.isnan(s)]
        out[name] = (
            (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))) if len(valid) else (np.nan,) * 2
        )
    return out


def _rows(
    dataset: str,
    analysis: str,
    tau: float,
    arrays: dict[str, np.ndarray],
    stats: dict[str, Stat],
    units: dict[str, np.ndarray | None],
) -> list[dict[str, Any]]:
    ones = np.ones(len(next(iter(arrays.values()))))
    point = {name: stat(arrays, ones) for name, stat in stats.items()}
    rows = []
    for unit, clusters in units.items():
        rng = np.random.default_rng(SEED)
        ci = _bootstrap(arrays, stats, clusters, rng)
        n_units = len(ones) if clusters is None else int(clusters.max()) + 1
        for name in stats:
            rows.append(
                {
                    "dataset": dataset,
                    "analysis": analysis,
                    "tau_s": tau,
                    "quantity": name,
                    "resampling_unit": unit,
                    "n_units": n_units,
                    "decides": unit in DECISION_UNITS,
                    "estimate": point[name],
                    "ci_low": ci[name][0],
                    "ci_high": ci[name][1],
                    "n_resamples": N_BOOTSTRAP,
                }
            )
    return rows


def _c1_stats() -> dict[str, Stat]:
    return {
        "referral_load": _mean("referred"),
        "abstention_share_of_referrals": _ratio("abstain", "referred"),
        "abstention_only_share_of_referrals": _ratio("abstain_only", "referred"),
        "unsafe_share_of_states": _mean("unsafe"),
        "recall": _ratio("tp", "unsafe"),
        "n_missed_unsafe": _count("fn"),
    }


def _c1_arrays(s: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "referred": s["referred"].to_numpy(float),
        "abstain": s["abstain"].to_numpy(float),
        "abstain_only": (s["abstain"] & ~s["threshold"]).to_numpy(float),
        "unsafe": s["unsafe"].to_numpy(float),
        "tp": (s["unsafe"] & s["referred"]).to_numpy(float),
        "fn": (s["unsafe"] & ~s["referred"]).to_numpy(float),
    }


def _eles_clusters(states: pd.Index) -> dict[str, np.ndarray | None]:
    lf = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl")
    lf.index = lf.index.astype(str)
    cols = json.loads((ELES_DIR / "processed" / "topology_cols_lines_only.json").read_text())
    key = lf.loc[states, cols].ne(0).astype(np.uint8).astype(str).agg("".join, axis=1)
    ts = load_eles_state_timestamps(RAW_ZIP).reindex(states)
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} states have no timestamp")
    return {
        "state": None,
        "topology_group": pd.factorize(key)[0],
        "calendar_day": pd.factorize(ts.dt.floor("D"))[0],
        "calendar_week": pd.factorize(ts.dt.to_period("W"))[0],
    }


def main() -> None:
    configure_logging()
    validation = pd.read_csv(VALIDATION)
    if not validation["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")

    rows: list[dict[str, Any]] = []

    bus39 = pd.read_parquet(BUS39_LOSO)
    for tau in THRESHOLDS:
        s = _state_frame(bus39, "pred_screening_min", "covered", tau)
        rows += _rows("BUS39", "C1", tau, _c1_arrays(s), _c1_stats(), {"state": None})

    loso = pd.read_parquet(ELES_LOSO)
    causal = pd.read_parquet(ELES_CAUSAL0)
    key = ["state", "record_ordinal"]
    merged = loso.merge(
        causal[[*key, "covered", "pred_screening_min"]], on=key, suffixes=("", "_causal"), validate="1:1"
    )
    if (merged["covered_causal"] & ~merged["covered"]).any():
        raise ValueError("a record is covered causally but not retrospectively; the sequential split is undefined")
    logger.info(
        f"ELES records {len(merged):,}: retrospective covered {int(merged['covered'].sum()):,}, "
        f"causal covered {int(merged['covered_causal'].sum()):,}"
    )

    states = pd.Index(merged["state"].drop_duplicates())
    units = _eles_clusters(states)
    for tau in THRESHOLDS:
        retro = _state_frame(merged, "pred_screening_min", "covered", tau).loc[states]
        rows += _rows("ELES", "C1", tau, _c1_arrays(retro), _c1_stats(), units)

        # Causal coverage with the retrospective estimate on the records that keep support.
        mix = _state_frame(merged, "pred_screening_min", "covered_causal", tau).loc[states]
        caus = _state_frame(merged, "pred_screening_min_causal", "covered_causal", tau).loc[states]
        if (retro["referred"] & ~mix["referred"]).any():
            raise ValueError("coverage loss removed a referral; the coverage step should only add referrals")
        arrays = {
            "r_retro": retro["referred"].to_numpy(float),
            "r_mix": mix["referred"].to_numpy(float),
            "r_causal": caus["referred"].to_numpy(float),
            "fn_retro": (retro["unsafe"] & ~retro["referred"]).to_numpy(float),
            "fn_mix": (mix["unsafe"] & ~mix["referred"]).to_numpy(float),
            "fn_causal": (caus["unsafe"] & ~caus["referred"]).to_numpy(float),
        }
        arrays["d_cov"] = arrays["r_mix"] - arrays["r_retro"]
        arrays["d_err"] = arrays["r_causal"] - arrays["r_mix"]
        arrays["d_diff"] = arrays["d_cov"] - arrays["d_err"]
        arrays["dfn_cov"] = arrays["fn_mix"] - arrays["fn_retro"]
        arrays["dfn_err"] = arrays["fn_causal"] - arrays["fn_mix"]
        stats: dict[str, Stat] = {
            "referral_load_retrospective": _mean("r_retro"),
            "referral_load_causal_coverage_retrospective_estimates": _mean("r_mix"),
            "referral_load_causal": _mean("r_causal"),
            "delta_referral_coverage": _mean("d_cov"),
            "delta_referral_error": _mean("d_err"),
            "delta_referral_coverage_minus_error": _mean("d_diff"),
            "missed_unsafe_retrospective": _count("fn_retro"),
            "missed_unsafe_causal": _count("fn_causal"),
            "delta_missed_unsafe_coverage": _count("dfn_cov"),
            "delta_missed_unsafe_error": _count("dfn_err"),
        }
        rows += _rows("ELES", "C2", tau, arrays, stats, units)

    out = pd.DataFrame(rows)

    # The pre-specified decision.
    d = out[
        (out["dataset"] == "ELES")
        & (out["analysis"] == "C2")
        & (out["tau_s"] == DECISION_TAU)
        & (out["quantity"] == "delta_referral_coverage_minus_error")
        & out["decides"]
    ].assign(width=lambda x: x["ci_high"] - x["ci_low"])
    widest = d.loc[d["width"].idxmax()]
    retained = bool(widest["estimate"] > 0 and widest["ci_low"] > 0)
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {
                        "dataset": "ELES",
                        "analysis": "decision",
                        "tau_s": DECISION_TAU,
                        "quantity": "claim_retained",
                        "resampling_unit": widest["resampling_unit"],
                        "n_units": widest["n_units"],
                        "decides": True,
                        "estimate": float(retained),
                        "ci_low": widest["ci_low"],
                        "ci_high": widest["ci_high"],
                        "n_resamples": N_BOOTSTRAP,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        f"Decision at tau={DECISION_TAU:g}: coverage minus error {widest['estimate']:.4f} "
        f"[{widest['ci_low']:.4f}, {widest['ci_high']:.4f}] under {widest['resampling_unit']} -> "
        f"claim {'retained' if retained else 'withdrawn'}"
    )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
