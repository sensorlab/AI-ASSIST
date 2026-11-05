"""Headline ELES intervals under state, topology-group, calendar-day and calendar-week resampling (R2.4b, R2.4c).

Pre-specified in the manuscript repository's revision/analyses/04-clustered-intervals-and-label-steps.md, committed
before any number here was computed. Each headline result is recomputed from the records its published script uses,
its point estimate is checked against the published value, and its interval is drawn by resampling whole clusters of
states. State, topology group and calendar day decide (the reported interval is the widest of the three); calendar week
is sensitivity only. Resample counts follow each published analysis; seed 42.

    H1  Table 1, ELES: recall, false-alarm rate and state referral at 100, 150, 200 ms        2,000 resamples
    H2  Table 2, ELES: MAE of the oracle and the highest-support generator                    1,000
    H3  Retrieval minus gradient-boosting MAE on the matched five-fold records                1,000
    H4  Causal minus retrospective paired MAE                                                  1,000
    H5  Table 4, ELES: nAURC of the seven diagnostics, and the count diagnostics on all records 1,000
    H6  Matched-workload recall, retrieval minus gradient boosting                             1,000

Run from the repository root:
    nice -n 19 uv run python scripts/paper/clustered_intervals.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import n_neighbors_selected_naurc as nn  # noqa: E402  (pins TOPOLOGY_VARIANT and adds scripts/evaluation to the path)
from referral_decomposition import (  # noqa: E402
    DECISION_UNITS,
    ELES_LOSO,
    VALIDATION,
    Stat,
    _bootstrap,
    _eles_clusters,
    _mean,
    _ratio,
    _state_frame,
)
from screening_matched_workload import _crossfit_referred, _state_scores  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = DATA_DIR / "clustered_intervals.csv"
DEORACLED: Final[Path] = TMP_DIR / "eles_deoracled_records_eles-2026-06_lines_only.parquet"
KFOLD: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_group_kfold.parquet"
SUPERVISED: Final[Path] = TMP_DIR / "ml_benchmark_predictions-eles-2026-06.parquet"
BASELINE: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"
CAUSAL: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_causal0h.parquet"
GB: Final[str] = "hist_gradient_boosting"
SEED: Final[int] = 42
UNITS: Final[tuple[str, ...]] = ("state", "topology_group", "calendar_day", "calendar_week")
POINT_ATOL: Final[float] = 1e-9

# Table 1 ELES rows as published, to the precision printed; the harness reproduces them (validation gate).
TABLE1: Final[dict[tuple[float, str], float]] = {
    (0.10, "false_alarm"): 0.4031,
    (0.10, "referral"): 0.4031,
    (0.15, "recall"): 0.9249,
    (0.15, "false_alarm"): 0.4371,
    (0.15, "referral"): 0.5110,
    (0.20, "recall"): 0.9752,
    (0.20, "false_alarm"): 0.5433,
    (0.20, "referral"): 0.8127,
}


class Units:
    """Cluster codes for every ELES state, restricted and re-coded for any subset of states."""

    def __init__(self, states: pd.Index) -> None:
        self.states = states
        self.codes = _eles_clusters(states)

    def for_states(self, subset: pd.Index) -> dict[str, np.ndarray | None]:
        pos = self.states.get_indexer(subset)
        if (pos < 0).any():
            raise ValueError("a state has no cluster codes")
        out: dict[str, np.ndarray | None] = {"state": None}
        for unit in UNITS[1:]:
            out[unit] = pd.factorize(self.codes[unit][pos])[0]
        return out


def _check(analysis: str, quantity: str, value: float, published: float, atol: float = POINT_ATOL) -> None:
    if not abs(value - published) <= atol:
        raise ValueError(f"{analysis} {quantity}: {value!r} does not reproduce the published {published!r}")


def _rows(
    analysis: str,
    arrays: dict[str, np.ndarray],
    stats: dict[str, Stat],
    units: dict[str, np.ndarray | None],
    n_resamples: int,
    reference: dict[str, float | None],
) -> list[dict[str, Any]]:
    ones = np.ones(len(next(iter(arrays.values()))))
    point = {name: stat(arrays, ones) for name, stat in stats.items()}
    rows = []
    for unit in UNITS:
        ci = _bootstrap(arrays, stats, units[unit], np.random.default_rng(SEED), n_resamples)
        for name in stats:
            rows.append(
                {
                    "analysis": analysis,
                    "quantity": name,
                    "resampling_unit": unit,
                    "n_units": len(ones) if units[unit] is None else int(units[unit].max()) + 1,
                    "decides": unit in DECISION_UNITS,
                    "estimate": point[name],
                    "ci_low": ci[name][0],
                    "ci_high": ci[name][1],
                    "n_resamples": n_resamples,
                    "reference": reference.get(name),
                }
            )
    return rows


def _diff(a: Stat, b: Stat) -> Stat:
    return lambda x, w: a(x, w) - b(x, w)


def _per_state(df: pd.DataFrame, cols: dict[str, pd.Series]) -> tuple[pd.Index, dict[str, np.ndarray]]:
    g = df.assign(**cols, _n=1.0).groupby("state", sort=False)[[*cols, "_n"]].sum()
    return g.index, {**{c: g[c].to_numpy(float) for c in cols}, "n": g["_n"].to_numpy(float)}


def h1(units: Units) -> list[dict[str, Any]]:
    records = pd.read_parquet(ELES_LOSO, columns=["state", "covered", "cct_true", "pred_screening_min"])
    rows = []
    for tau in (0.10, 0.15, 0.20):
        sf = _state_frame(records, "pred_screening_min", "covered", tau)
        safe = ~sf["unsafe"]
        arrays = {
            "referred": sf["referred"].to_numpy(float),
            "unsafe": sf["unsafe"].to_numpy(float),
            "caught": (sf["unsafe"] & sf["referred"]).to_numpy(float),
            "safe": safe.to_numpy(float),
            "false_alarm": (safe & sf["referred"]).to_numpy(float),
        }
        stats: dict[str, Stat] = {
            f"false_alarm_{round(1000 * tau)}ms": _ratio("false_alarm", "safe"),
            f"referral_{round(1000 * tau)}ms": _mean("referred"),
        }
        if sf["unsafe"].any():
            stats[f"recall_{round(1000 * tau)}ms"] = _ratio("caught", "unsafe")
        ones = np.ones(len(sf))
        for name, stat in stats.items():
            _check("H1", name, stat(arrays, ones), TABLE1[(tau, name.rsplit("_", 1)[0])], atol=5e-5)
        rows += _rows("H1", arrays, stats, units.for_states(sf.index), 2000, {})
    return rows


def h2(units: Units) -> list[dict[str, Any]]:
    df = pd.read_parquet(DEORACLED)
    published = pd.read_csv(DATA_DIR / "bootstrap_ci_deoracling.csv").query("dataset == 'eles/2026-06'")
    published = published.set_index("level")["mae_point"]
    rows = []
    for level, covered, col in (
        ("oracle", "full_deoracled_covered", "pred_oracle_loc_and_gen"),
        ("generator_only_selection", "gen_deoracled_covered", "pred_gen_deoracled_selection"),
    ):
        sub = df[df[covered] & df[col].notna()]
        states, arrays = _per_state(sub, {"ae": (sub["cct_true"] - sub[col]).abs()})
        stats: dict[str, Stat] = {f"mae_{level}": _ratio("ae", "n")}
        _check("H2", level, stats[f"mae_{level}"](arrays, np.ones(len(states))), float(published[level]))
        rows += _rows("H2", arrays, stats, units.for_states(states), 1000, {})
    return rows


def h3(units: Units) -> list[dict[str, Any]]:
    r = pd.read_parquet(KFOLD)
    r["record_ordinal"] = r.groupby("state", sort=False).cumcount()
    r = r[r["covered"] & r["cct_weighted_per_location"].notna()]
    m = pd.read_parquet(SUPERVISED)
    m = m[m["model"] == GB].copy()
    m["record_ordinal"] = m.groupby("state", sort=False).cumcount()
    for f in (r, m):
        f["state"] = f["state"].astype(str)
    j = r.merge(
        m[["state", "record_ordinal", "cct_true", "cct_pred"]], on=["state", "record_ordinal"], suffixes=("", "_ml")
    )
    if not np.allclose(j["cct_true"], j["cct_true_ml"], atol=1e-12):
        raise ValueError("H3: targets disagree after the record-key join")
    states, arrays = _per_state(
        j,
        {
            "ae_ret": (j["cct_true"] - j["cct_weighted_per_location"]).abs(),
            "ae_gb": (j["cct_true"] - j["cct_pred"]).abs(),
        },
    )
    stats: dict[str, Stat] = {
        "mae_retrieval_minus_gradient_boosting": _diff(_ratio("ae_ret", "n"), _ratio("ae_gb", "n"))
    }
    published = (
        pd.read_csv(DATA_DIR / "label_resolution_check.csv").set_index("comparison").loc["raw", "mean_paired_diff"]
    )
    _check(
        "H3",
        "difference",
        stats["mae_retrieval_minus_gradient_boosting"](arrays, np.ones(len(states))),
        float(published),
    )
    return _rows("H3", arrays, stats, units.for_states(states), 1000, {"mae_retrieval_minus_gradient_boosting": 0.0})


def h4(units: Units) -> list[dict[str, Any]]:
    def arm(path: Path) -> pd.DataFrame:
        f = pd.read_parquet(path, columns=["state", "covered", "cct_true", "cct_weighted_per_location"])
        f["record_ordinal"] = f.groupby("state", sort=False).cumcount()
        f["state"] = f["state"].astype(str)
        return f[f["covered"] & f["cct_weighted_per_location"].notna()]

    j = arm(BASELINE).merge(arm(CAUSAL), on=["state", "record_ordinal"], suffixes=("_b", "_c"), validate="1:1")
    states, arrays = _per_state(
        j,
        {
            "ae_b": (j["cct_true_b"] - j["cct_weighted_per_location_b"]).abs(),
            "ae_c": (j["cct_true_c"] - j["cct_weighted_per_location_c"]).abs(),
        },
    )
    stats: dict[str, Stat] = {"mae_causal_minus_retrospective": _diff(_ratio("ae_c", "n"), _ratio("ae_b", "n"))}
    published = pd.read_csv(DATA_DIR / "eles_causal_evaluation.csv").query("lag_hours == 0")["mae_diff"].iloc[0]
    _check("H4", "difference", stats["mae_causal_minus_retrospective"](arrays, np.ones(len(states))), float(published))
    return _rows("H4", arrays, stats, units.for_states(states), 1000, {"mae_causal_minus_retrospective": 0.0})


def _naurc_rows(
    population: str,
    df: pd.DataFrame,
    metrics: tuple[tuple[str, bool], ...],
    units: Units,
    published: dict[str, float],
) -> list[dict[str, Any]]:
    states = pd.Index(df["state"].astype(str).drop_duplicates())
    codes = units.for_states(states)
    state_pos = states.get_indexer(df["state"].astype(str))
    err = df["err"].to_numpy(float)
    values = {name: df[name].to_numpy(float) for name, _ in metrics}
    point = {name: nn._naurc_of(values[name], err, higher_is_better=hib) for name, hib in metrics}
    for name in point:
        _check("H5", f"{population} {name}", point[name], published[name])
    rows = []
    for unit in UNITS:
        row_code = state_pos if codes[unit] is None else codes[unit][state_pos]
        order = np.argsort(row_code, kind="stable")
        parts = np.split(order, np.flatnonzero(np.diff(row_code[order])) + 1)
        rng = np.random.default_rng(SEED)
        draws = {name: np.empty(1000) for name in point}
        for b in range(1000):
            idx = np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))])
            for name, hib in metrics:
                draws[name][b] = nn._naurc_of(values[name][idx], err[idx], higher_is_better=hib)
        for name in point:
            rows.append(
                {
                    "analysis": "H5",
                    "quantity": f"naurc_{name}_{population}",
                    "resampling_unit": unit,
                    "n_units": len(parts),
                    "decides": unit in DECISION_UNITS,
                    "estimate": point[name],
                    "ci_low": float(np.percentile(draws[name], 2.5)),
                    "ci_high": float(np.percentile(draws[name], 97.5)),
                    "n_resamples": 1000,
                    "reference": 1.0,
                }
            )
        logger.info(f"H5 {population}: {unit} done")
    return rows


def h5(units: Units) -> list[dict[str, Any]]:
    covered = nn._load_frame("eles/2026-06")
    seven = (*nn.METRIC_SETS["main"], *nn.CANDIDATES)
    common = covered[nn._common_support_mask(covered, seven)]
    main = pd.read_csv(DATA_DIR / "bootstrap_ci_selected_eles-2026-06.csv").query("quantity == 'nAURC(MAE)'")
    counts = pd.read_csv(DATA_DIR / "n_neighbors_selected_naurc.csv").query(
        "dataset == 'eles/2026-06' and quantity == 'nAURC(MAE)'"
    )
    pub_common = {
        **main.set_index("metric")["point_estimate"].to_dict(),
        **counts.query("population == 'table3_common'").set_index("metric")["point_estimate"].to_dict(),
    }
    pub_own = counts.query("population == 'own_support'").set_index("metric")["point_estimate"].to_dict()
    rows = _naurc_rows("common", common, seven, units, pub_common)
    own = covered[nn._common_support_mask(covered, nn.CANDIDATES)]
    return rows + _naurc_rows("all_covered", own, nn.CANDIDATES, units, pub_own)


def h6(units: Units) -> list[dict[str, Any]]:
    r = pd.read_parquet(KFOLD)
    r["state"] = r["state"].astype(str)
    r["pred"] = r["cct_weighted_per_location"].where(r["covered"])
    rs = _state_scores(r, "pred")
    unsafe = (rs["true_min"] < 0.20).to_numpy()
    ret_ref = (rs["any_missing"] | (rs["score"] < 0.20)).to_numpy()
    m = pd.read_parquet(SUPERVISED)
    m["state"] = m["state"].astype(str)
    ss = _state_scores(m[m["model"] == GB].rename(columns={"cct_pred": "pred"}), "pred").reindex(rs.index)
    gb_ref = _crossfit_referred(ss, float(ret_ref.mean()))
    arrays = {
        "unsafe": unsafe.astype(float),
        "c_ret": (ret_ref & unsafe).astype(float),
        "c_gb": (gb_ref & unsafe).astype(float),
    }
    stats: dict[str, Stat] = {
        "recall_retrieval_minus_gradient_boosting": _diff(_ratio("c_ret", "unsafe"), _ratio("c_gb", "unsafe"))
    }
    published = (
        pd.read_csv(DATA_DIR / "screening_matched_workload.csv")
        .query("method == @GB and at == 'cross-fitted to retrieval load'")["recall_diff_retrieval_minus_model"]
        .iloc[0]
    )
    _check(
        "H6",
        "difference",
        stats["recall_retrieval_minus_gradient_boosting"](arrays, np.ones(len(rs))),
        float(published),
    )
    return _rows(
        "H6",
        arrays,
        stats,
        units.for_states(pd.Index(rs.index)),
        1000,
        {"recall_retrieval_minus_gradient_boosting": 0.0},
    )


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    states = pd.Index(pd.read_parquet(ELES_LOSO, columns=["state"])["state"].astype(str).drop_duplicates())
    units = Units(states)
    analyses: list[tuple[str, Callable[[Units], list[dict[str, Any]]]]] = [
        ("H1", h1),
        ("H2", h2),
        ("H3", h3),
        ("H4", h4),
        ("H6", h6),
        ("H5", h5),
    ]
    rows: list[dict[str, Any]] = []
    for name, fn in analyses:
        rows += fn(units)
        logger.info(f"{name} done")
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)  # partial results survive a later failure
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
