"""Screening reliability: distance-based rejection and calibrated margins (R2.12b, R2.13a-d).

Pre-specified in the manuscript repository's revision/analyses/02-screening-reliability.md, committed before any number
here was computed. The screening rule is the paper's (P0): a state is referred when any recorded contingency lacks an
estimate (abstention) or its minimum screening estimate yhat is below tau; it is unsafe when its minimum true CCT y is
below tau. Every policy below refers what P0 refers, plus the states it adds.

    P1  fixed margin          yhat < tau + m
    P2  constant conformal    yhat < tau + q,        q from residuals r = yhat - y of calibration states
    P3  scaled conformal      yhat < tau + q * s,    s = dispersion at the deciding record, floored at 0.01 s;
                                                     single-neighbour deciding records form a Mondrian class with
                                                     their own constant margin
    P4  distance rejection    D > d,                 D = largest nearest-neighbour distance among the state's records
    Pall                      every state

Calibration uses fully covered states only. Retrospective: cross-conformal over five folds grouped by topology group
(ELES) or state (BUS39). Causal ELES: online, from states with a strictly earlier timestamp.

Q1 validity: missed share among fully covered states against alpha. Q2 efficiency: P3 and P4 matched to P2's referral
load using estimates, scales and distances only (never outcomes), missed counts compared paired. Q3 missed unsafe states
under P0. Q4 threshold curve and the P1 grid. Q5 expected cost against referring every state.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/screening_reliability.py
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_eles_state_timestamps  # noqa: E402
from referral_decomposition import (  # noqa: E402
    DECISION_UNITS,
    RAW_ZIP,
    VALIDATION,
    Stat,
    _bootstrap,
    _count,
    _eles_clusters,
    _mean,
    _ratio,
)

from src.benchmarking import group_k_fold_indices  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"

ELES_INPUTS: Final[dict[str, tuple[Path, Path]]] = {
    "retrospective": (
        TMP_DIR / "harness_records_exact_lines_loso.parquet",
        TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet",
    ),
    "causal_lag0": (
        TMP_DIR / "harness_records_exact_lines_causal0h.parquet",
        TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_causal0h.parquet",
    ),
}
BUS39_RECORDS: Final[Path] = TMP_DIR / "generator_deoracled_records_loso.parquet"
BUS39_DIAGNOSTICS: Final[Path] = TMP_DIR / "generator_diagnostics_selected_bus39_loso.parquet"
BUS39_TSA: Final[Path] = PROJECT_DIR / "datasets" / "bus39" / "interim" / "tsa.pkl"

RELIABILITY_PATH: Final[Path] = PAPER_DATA_DIR / "screening_reliability.csv"
CURVE_PATH: Final[Path] = PAPER_DATA_DIR / "screening_threshold_curve.csv"
MISSED_PATH: Final[Path] = PAPER_DATA_DIR / "screening_missed_unsafe.csv"

RESOLUTION: Final[float] = 0.01
THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)
MISSED_THRESHOLDS: Final[tuple[float, ...]] = (0.15, 0.20)
PRIMARY_TAU: Final[float] = 0.20
ALPHAS: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05)
PRIMARY_ALPHA: Final[float] = 0.01
FIXED_MARGINS: Final[tuple[float, ...]] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10)
COST_RATIOS: Final[tuple[float, ...]] = (1, 3, 10, 30, 100, 300, 1000, 3000, 10000)
TAU_CURVE: Final[np.ndarray] = np.round(np.arange(0.06, 0.30 + 1e-9, 0.01), 2)
N_FOLDS: Final[int] = 5
# Rounding guard for ceil((1 - alpha)(n + 1)), so e.g. alpha = 0.01, n = 99 gives rank 99, not 100.
RANK_EPS: Final[float] = 1e-9


# --- conformal margins -------------------------------------------------------------------------------------------


def conformal_margin(scores: np.ndarray, alpha: float) -> float:
    """The ceil((1 - alpha)(n + 1))-th smallest score; +inf when that rank exceeds n."""
    n = len(scores)
    k = math.ceil((1 - alpha) * (n + 1) - RANK_EPS)
    return math.inf if k > n else float(np.partition(scores, k - 1)[k - 1])


def _class_margins(r: np.ndarray, s: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """(P2 margin, P3 scaled-class factor, P3 single-neighbour-class margin) from calibration residuals."""
    single = np.isnan(s)
    return (
        conformal_margin(r, alpha),
        conformal_margin(r[~single] / s[~single], alpha),
        conformal_margin(r[single], alpha),
    )


def _apply(q: tuple[float, float, float], s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-state margins for P2 and P3 given calibrated (q_const, q_scaled, q_single)."""
    p2 = np.full(len(s), q[0])
    p3 = np.where(np.isnan(s), q[2], q[1] * np.nan_to_num(s, nan=1.0))
    return p2, p3


def cross_fit_margins(
    r: np.ndarray, s: np.ndarray, full: np.ndarray, fold: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each state's margins from the fully covered states of the other folds."""
    m2, m3 = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for f in np.unique(fold):
        cal = full & (fold != f)
        test = fold == f
        m2[test], m3[test] = _apply(_class_margins(r[cal], s[cal], alpha), s[test])
    return m2, m3


def online_margins(
    r: np.ndarray, s: np.ndarray, full: np.ndarray, ts: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each state's margins from the fully covered states with a strictly earlier timestamp."""
    m2, m3 = np.full(len(r), np.nan), np.full(len(r), np.nan)
    for t in np.unique(ts):
        cal = full & (ts < t)
        test = ts == t
        m2[test], m3[test] = _apply(_class_margins(r[cal], s[cal], alpha), s[test])
    return m2, m3


def match_load(base: np.ndarray, score: np.ndarray, target: int) -> np.ndarray:
    """Add the highest-scoring unreferred states to base, as many as fit within target referrals.

    A tie group at the cut is added whole or not at all, so the result never exceeds target. Uses no outcomes.
    """
    refer = base.copy()
    extra = target - int(base.sum())
    if extra <= 0:
        return refer
    cand = np.flatnonzero(~base)
    if np.isnan(score[cand]).any():
        raise ValueError("an unreferred state has no score")
    if extra >= len(cand):
        refer[cand] = True
        return refer
    v = np.sort(score[cand])[::-1][extra - 1]
    ge, gt = score[cand] >= v, score[cand] > v
    refer[cand[ge if ge.sum() <= extra else gt]] = True
    return refer


# --- inputs ------------------------------------------------------------------------------------------------------


def _ordinal(df: pd.DataFrame) -> pd.Series:
    return df.groupby("state", sort=False).cumcount()


def _join_diagnostics(records: pd.DataFrame, diagnostics: pd.DataFrame, check: list[str]) -> pd.DataFrame:
    """Join per-record diagnostics on (state, order within state), after checking the shared columns agree."""
    records = records.assign(_ord=_ordinal(records))
    diagnostics = diagnostics.assign(_ord=_ordinal(diagnostics))
    keep = ["state", "_ord", "n_neighbors", "cct_weighted_std", "distance_min", "gen_true_is_selected"]
    merged = records.merge(
        diagnostics[keep + check], on=["state", "_ord"], suffixes=("", "_diag"), validate="1:1", how="left"
    )
    for col in check:
        a, b = merged[col], merged[f"{col}_diag"]
        # A blank is accepted only on uncovered records, where the diagnostics leave the generator unset.
        if not ((a == b) | (a.isna() & b.isna()) | ((a.isna() | b.isna()) & ~merged["covered"])).all():
            raise ValueError(f"records and diagnostics disagree on {col}")
    covered = merged["covered"]
    if (covered & (merged["n_neighbors"] < 2)).ne(covered & merged["cct_weighted_std"].isna()).any():
        raise ValueError("dispersion is missing on records that do not have exactly one neighbour")
    return merged.drop(columns=[f"{c}_diag" for c in check])


def _load_eles(availability: str) -> pd.DataFrame:
    harness, diagnostics = ELES_INPUTS[availability]
    records = pd.read_parquet(harness)
    records = records.sort_values(["state", "record_ordinal"], kind="stable").drop(columns="record_ordinal")
    diag = pd.read_parquet(diagnostics)
    diag = diag.assign(_o=np.arange(len(diag))).sort_values(["state", "_o"], kind="stable").drop(columns="_o")
    return _join_diagnostics(records.reset_index(drop=True), diag, ["cct_true", "loc_true", "gen_true", "covered"])


def _load_bus39() -> pd.DataFrame:
    records = pd.read_parquet(BUS39_RECORDS)
    diag = pd.read_parquet(BUS39_DIAGNOSTICS)
    merged = _join_diagnostics(records, diag, ["cct_true", "gen_true", "covered"])
    # Records follow tsa.pkl grouped by state; the location comes from there, checked on CCT and generator
    # (uncovered records carry no generator).
    tsa = pd.read_pickle(BUS39_TSA)
    tsa = tsa.assign(state=tsa["state"].astype(str), _o=np.arange(len(tsa)))
    tsa = tsa.sort_values(["state", "_o"], kind="stable")
    tsa = tsa.assign(_ord=_ordinal(tsa))[["state", "_ord", "CCT", "Crit_gen", "Location"]]
    merged = merged.merge(tsa, on=["state", "_ord"], validate="1:1", how="left")
    gen = merged["Crit_gen"].astype(str).str.strip().str.lower()
    if not (
        np.isclose(merged["CCT"].astype(float), merged["cct_true"]).all()
        and (gen.eq(merged["gen_true"]) | merged["gen_true"].isna()).all()
    ):
        raise ValueError("BUS39 records do not follow tsa.pkl order")
    return merged.assign(loc_true=merged["Location"].astype(str)).drop(columns=["CCT", "Crit_gen", "Location"])


def state_frame(records: pd.DataFrame, estimate: str) -> pd.DataFrame:
    """Per state: y, yhat, full coverage, residual, support scale s, distance score D, and the y-setting record."""
    records = records.assign(_est=records[estimate].where(records["covered"]))
    g = records.groupby("state", sort=False)
    out = pd.DataFrame(
        {
            "y": g["cct_true"].min(),
            "yhat": g["_est"].min(),
            "full": g["covered"].all(),
            "D": records["distance_min"].where(records["covered"]).groupby(records["state"], sort=False).max(),
        }
    )
    covered = records[records["covered"]]
    deciding = covered.sort_values(["state", "_est", "_ord"], kind="stable").groupby("state", sort=False).head(1)
    s = deciding.set_index("state")["cct_weighted_std"].clip(lower=RESOLUTION)
    out["s"] = s.reindex(out.index)
    out["r"] = (out["yhat"] - out["y"]).where(out["full"])
    setting = records.sort_values(["state", "cct_true", "_ord"], kind="stable").groupby("state", sort=False).head(1)
    setting = setting.set_index("state")
    for col in ("loc_true", "gen_true_is_selected", "distance_min", "cct_weighted_std"):
        out[f"ysetting_{col}"] = setting[col].reindex(out.index)
    return out


# --- evaluation --------------------------------------------------------------------------------------------------


def _policy_arrays(st: pd.DataFrame, tau: float, refer: np.ndarray) -> dict[str, np.ndarray]:
    unsafe = (st["y"] < tau).to_numpy()
    return {
        "referred": refer.astype(float),
        "missed": (unsafe & ~refer).astype(float),
        "caught": (unsafe & refer).astype(float),
        "unsafe": unsafe.astype(float),
        "full": st["full"].to_numpy(float),
    }


POLICY_STATS: Final[dict[str, Stat]] = {
    "referral_load": _mean("referred"),
    "missed_unsafe": _count("missed"),
    "missed_share_of_fully_covered": _ratio("missed", "full"),
    "recall": _ratio("caught", "unsafe"),
}


def _rows(
    ctx: dict[str, Any], arrays: dict[str, np.ndarray], stats: dict[str, Stat], units: dict[str, np.ndarray | None]
) -> list[dict[str, Any]]:
    ones = np.ones(len(next(iter(arrays.values()))))
    point = {name: stat(arrays, ones) for name, stat in stats.items()}
    rows = []
    for unit, clusters in units.items():
        ci = _bootstrap(arrays, stats, clusters, np.random.default_rng(42))
        for name in stats:
            rows.append(
                {
                    **ctx,
                    "quantity": name,
                    "resampling_unit": unit,
                    "n_units": len(ones) if clusters is None else int(clusters.max()) + 1,
                    "decides": unit in DECISION_UNITS or ctx["dataset"] == "BUS39",
                    "estimate": point[name],
                    "ci_low": ci[name][0],
                    "ci_high": ci[name][1],
                }
            )
    return rows


def _margins(
    st: pd.DataFrame, alpha: float, fold: np.ndarray | None, ts: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    r, s, full = st["r"].to_numpy(), st["s"].to_numpy(), st["full"].to_numpy()
    r = np.nan_to_num(r)  # only fully covered states are ever used for calibration
    if ts is not None:
        return online_margins(r, s, full, ts, alpha)
    return cross_fit_margins(r, s, full, fold, alpha)


def _p3_score(st: pd.DataFrame, tau: float, m3: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Rank for scaling P3's per-state margins by one multiplier: refer first the states needing the smallest one."""
    cand = ~base
    if (m3[cand] <= 0).any():
        raise ValueError("a non-positive P3 margin; the multiplier family is not monotone (record a deviation)")
    return -(st["yhat"].to_numpy() - tau) / m3


def evaluate(
    dataset: str,
    availability: str,
    st: pd.DataFrame,
    units: dict[str, np.ndarray | None],
    fold: np.ndarray | None,
    ts: np.ndarray | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    yhat = st["yhat"].to_numpy()
    full = st["full"].to_numpy()
    n = len(st)
    margins = {alpha: _margins(st, alpha, fold, ts) for alpha in ALPHAS}
    r = st["r"].to_numpy()
    for alpha, (m2, m3) in margins.items():
        logger.info(
            f"{dataset} {availability} alpha={alpha:g}: P2 margin median {np.median(m2[full & np.isfinite(m2)]):.4f} s, "
            f"{int((full & np.isinf(m2)).sum())} fully covered states with no finite margin (burn-in)"
        )
        for name, m in (("P2", m2), ("P3", m3)):
            finite = full & np.isfinite(m)
            # Not pre-specified (Deviation 1): the event the margin bounds directly, y < yhat - q, among calibrated states.
            for quantity, value in (
                ("states_without_finite_margin", int((full & np.isinf(m)).sum())),
                ("median_margin_s", float(np.median(m[finite]))),
                ("exceedance_rate", float((r[finite] > m[finite]).mean())),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "availability": availability,
                        "analysis": "margin",
                        "policy": name,
                        "alpha": alpha,
                        "quantity": quantity,
                        "estimate": value,
                    }
                )

    for tau in THRESHOLDS:
        if dataset == "ELES" and tau == 0.10:
            base = ~full | (yhat < tau)
            rows.append(
                {
                    "dataset": dataset,
                    "availability": availability,
                    "analysis": "Q1",
                    "policy": "P0",
                    "tau_s": tau,
                    "quantity": "referral_load",
                    "estimate": base.mean(),
                }
            )
            continue
        ctx = {"dataset": dataset, "availability": availability, "tau_s": tau}
        base = ~full | (yhat < tau)
        rows += _rows({**ctx, "analysis": "Q1", "policy": "P0"}, _policy_arrays(st, tau, base), POLICY_STATS, units)
        refers: dict[str, np.ndarray] = {"P0": base, "Pall": np.ones(n, bool)}
        for m in FIXED_MARGINS:
            refer = base | (yhat < tau + m)
            refers[f"P1_m{m:g}"] = refer
            rows += _rows(
                {**ctx, "analysis": "Q4", "policy": "P1", "param": m},
                _policy_arrays(st, tau, refer),
                POLICY_STATS,
                units,
            )
        for alpha, (m2, m3) in margins.items():
            for name, m in (("P2", m2), ("P3", m3)):
                refer = base | (yhat < tau + m)
                refers[f"{name}_a{alpha:g}"] = refer
                rows += _rows(
                    {**ctx, "analysis": "Q1", "policy": name, "alpha": alpha},
                    _policy_arrays(st, tau, refer),
                    POLICY_STATS,
                    units,
                )

        # Q2: P3 and P4 at P2's load, matched on estimates, scales and distances only.
        m3 = margins[PRIMARY_ALPHA][1]
        p2 = refers[f"P2_a{PRIMARY_ALPHA:g}"]
        target = int(p2.sum())
        p3_score = _p3_score(st, tau, m3, base)
        d_score = st["D"].to_numpy()
        p3m = match_load(base, p3_score, target)
        p4m = match_load(base, d_score, target)
        refers["P3_matched"], refers["P4_matched"] = p3m, p4m
        arrays: dict[str, np.ndarray] = {}
        for name, refer in (("P2", p2), ("P3m", p3m), ("P4m", p4m)):
            for k, v in _policy_arrays(st, tau, refer).items():
                arrays[f"{k}_{name}"] = v
        arrays["dmiss_P3m"] = arrays["missed_P3m"] - arrays["missed_P2"]
        arrays["dmiss_P4m"] = arrays["missed_P4m"] - arrays["missed_P2"]
        q2_stats: dict[str, Stat] = {
            "referral_load_P2": _mean("referred_P2"),
            "referral_load_P3_matched": _mean("referred_P3m"),
            "referral_load_P4_matched": _mean("referred_P4m"),
            "missed_unsafe_P2": _count("missed_P2"),
            "missed_unsafe_P3_matched": _count("missed_P3m"),
            "missed_unsafe_P4_matched": _count("missed_P4m"),
            "missed_P3_matched_minus_P2": _count("dmiss_P3m"),
            "missed_P4_matched_minus_P2": _count("dmiss_P4m"),
        }
        q2 = _rows({**ctx, "analysis": "Q2", "policy": "matched", "alpha": PRIMARY_ALPHA}, arrays, q2_stats, units)
        rows += q2
        rows += _q2_decisions(ctx, q2)

        # Trade-off curves: each family's missed count at loads from P0's to every state.
        families = {"P1": -yhat, "P3": p3_score, "P4": d_score}
        start = math.ceil(100 * base.mean())
        for pct in range(start, 101):
            k = math.floor(pct / 100 * n)
            for fam, score in families.items():
                refer = match_load(base, score, k)
                a = _policy_arrays(st, tau, refer)
                curve.append(
                    {
                        **ctx,
                        "family": fam,
                        "target_load_pct": pct,
                        "referral_load": a["referred"].mean(),
                        "missed_unsafe": int(a["missed"].sum()),
                    }
                )

        # Q5: expected cost per state, referral = 1, missed unsafe = c.
        for name, refer in refers.items():
            a = _policy_arrays(st, tau, refer)
            load, miss = a["referred"].mean(), a["missed"].mean()
            for c in COST_RATIOS:
                rows.append(
                    {
                        **ctx,
                        "analysis": "Q5",
                        "policy": name,
                        "param": c,
                        "quantity": "expected_cost_per_state",
                        "estimate": load + c * miss,
                    }
                )
            rows.append(
                {
                    **ctx,
                    "analysis": "Q5",
                    "policy": name,
                    "quantity": "break_even_cost_ratio",
                    "estimate": (1 - load) / miss if miss > 0 else math.inf,
                }
            )
    return rows, curve


def _q2_decisions(ctx: dict[str, Any], q2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    df = pd.DataFrame(q2)
    for policy in ("P3_matched", "P4_matched"):
        d = df[(df["quantity"] == f"missed_{policy}_minus_P2") & df["decides"]]
        widest = d.loc[(d["ci_high"] - d["ci_low"]).idxmax()]
        out.append(
            {
                **ctx,
                "analysis": "Q2",
                "policy": policy,
                "alpha": PRIMARY_ALPHA,
                "quantity": "improves_on_P2",
                "resampling_unit": widest["resampling_unit"],
                "decides": True,
                "estimate": float(widest["estimate"] < 0 and widest["ci_high"] < 0),
                "ci_low": widest["ci_low"],
                "ci_high": widest["ci_high"],
            }
        )
    return out


def missed_profile(dataset: str, availability: str, st: pd.DataFrame, covered: pd.DataFrame) -> list[dict[str, Any]]:
    """Q3: who the unmodified screen misses. ELES locations are counted, never named."""
    rows = []
    dist = np.sort(covered["distance_min"].to_numpy())
    disp = np.sort(covered["cct_weighted_std"].dropna().to_numpy())
    for tau in MISSED_THRESHOLDS:
        base = ~st["full"] | (st["yhat"] < tau)
        unsafe = st["y"] < tau
        m = st[unsafe & ~base]
        locs = m["ysetting_loc_true"].value_counts()
        row: dict[str, Any] = {
            "dataset": dataset,
            "availability": availability,
            "tau_s": tau,
            "n_states": len(st),
            "n_unsafe": int(unsafe.sum()),
            "n_referred": int(base.sum()),
            "n_missed": len(m),
            "n_missed_within_label_resolution": int((tau - m["y"] <= RESOLUTION + 1e-9).sum()),
            "n_missed_with_true_generator_selected": int(m["ysetting_gen_true_is_selected"].astype(bool).sum()),
            "n_missed_locations": len(locs),
            "largest_location_share": locs.iloc[0] / len(m) if len(m) else np.nan,
        }
        if dataset == "BUS39" and len(m):
            row["largest_location"] = locs.index[0]
        for name, x in (("depth_s", tau - m["y"]), ("passed_by_s", m["yhat"] - tau)):
            for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                row[f"{name}_q{int(q * 100)}"] = float(x.quantile(q)) if len(m) else np.nan
        pct_d = np.searchsorted(dist, m["ysetting_distance_min"].to_numpy(), side="right") / len(dist)
        sd = m["ysetting_cct_weighted_std"].dropna().to_numpy()
        pct_s = np.searchsorted(disp, sd, side="right") / len(disp)
        row["distance_percentile_median"] = float(np.median(pct_d)) if len(m) else np.nan
        row["dispersion_percentile_median"] = float(np.median(pct_s)) if len(sd) else np.nan
        row["n_missed_single_neighbour"] = int(m["ysetting_cct_weighted_std"].isna().sum())
        rows.append(row)
    return rows


def threshold_curve(dataset: str, availability: str, st: pd.DataFrame) -> list[dict[str, Any]]:
    """Q4: P0 over a continuous threshold grid."""
    rows = []
    for tau in TAU_CURVE:
        base = (~st["full"] | (st["yhat"] < tau)).to_numpy()
        a = _policy_arrays(st, float(tau), base)
        n_unsafe = a["unsafe"].sum()
        rows.append(
            {
                "dataset": dataset,
                "availability": availability,
                "family": "P0",
                "tau_s": float(tau),
                "referral_load": a["referred"].mean(),
                "unsafe_share": a["unsafe"].mean(),
                "missed_unsafe": int(a["missed"].sum()),
                "recall": a["caught"].sum() / n_unsafe if n_unsafe else np.nan,
            }
        )
    return rows


def _eles_inputs() -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    for availability in ELES_INPUTS:
        records = _load_eles(availability)
        yield availability, state_frame(records, "pred_screening_min"), records[records["covered"]]


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")

    rows: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []

    bus39 = _load_bus39()
    st = state_frame(bus39, "pred_screening_min")
    fold = np.empty(len(st), int)
    for i, (_, test) in enumerate(group_k_fold_indices(st.index, n_splits=N_FOLDS)):
        fold[test] = i
    r, c = evaluate("BUS39", "retrospective", st, {"state": None}, fold, None)
    rows += r
    curve += c + threshold_curve("BUS39", "retrospective", st)
    missed += missed_profile("BUS39", "retrospective", st, bus39[bus39["covered"]])
    del bus39

    ts_all = load_eles_state_timestamps(RAW_ZIP)
    for availability, st, covered in _eles_inputs():
        states = pd.Index(st.index)
        units = _eles_clusters(states)
        ts = ts_all.reindex(states)
        if ts.isna().any():
            raise ValueError(f"{int(ts.isna().sum())} ELES states have no timestamp")
        logger.info(f"ELES timestamps: {len(ts):,} states, {ts.nunique():,} distinct")
        if availability == "causal_lag0":
            r, c = evaluate("ELES", availability, st, units, None, ts.to_numpy())
        else:
            fold = np.empty(len(st), int)
            for i, (_, test) in enumerate(group_k_fold_indices(units["topology_group"], n_splits=N_FOLDS)):
                fold[test] = i
            r, c = evaluate("ELES", availability, st, units, fold, None)
        rows += r
        curve += c + threshold_curve("ELES", availability, st)
        missed += missed_profile("ELES", availability, st, covered)

    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RELIABILITY_PATH, index=False)
    pd.DataFrame(curve).to_csv(CURVE_PATH, index=False)
    pd.DataFrame(missed).to_csv(MISSED_PATH, index=False)
    logger.info(f"Saved {RELIABILITY_PATH}, {CURVE_PATH}, {MISSED_PATH}")


if __name__ == "__main__":
    main()
