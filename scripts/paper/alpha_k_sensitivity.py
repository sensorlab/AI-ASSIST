"""Kernel bandwidth and neighbour count: swept on every state and selected out of fold (analysis 06, R2.8a-b).

Pre-specified in the manuscript repository's revision/analyses/06-alpha-k-sensitivity.md, committed before any number
here was computed. Every setting runs through the validated harness, leave one state out, on the full population.

Q1 (R2.8a), descriptive: per setting, error, generator top-1, coverage and screening at 150 and 200 ms; the range over
the neighbourhood alpha in [0.5, 2], K in [50, 200]; the hindsight best point, labelled optimistic.

Q2 (R2.8b): five outer group folds (states on BUS39, exact line-status topology groups on ELES). For fold f the setting
with the lowest MAE on the states outside f, over the records every setting covers there, is chosen; ties go to the
setting nearer the deployed one. Its out-of-fold records are the leave-one-state-out records of the states in f. The
selection records use no label in f: on BUS39 every inner query is rerun with fold f excluded from the database; on
ELES retrieval never leaves a topology group, so the leave-one-state-out records of the inner states already exclude
f, which is asserted. Out-of-fold selected against deployed MAE, paired on the records both cover, decides: deployed
materially worse at a reduction of at least 0.010 s with the 95% interval excluding zero (BUS39 states resampled, ELES
the widest of state, topology group and calendar day), negligible if the interval excludes zero below 0.010 s,
otherwise no material difference.

Each harness run is stored under tmp/alpha_k/ and reused if present, so an interrupted run resumes.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/alpha_k_sensitivity.py [n_jobs]
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import LINE_KEY, Rule, causal_exclusions, run_rule  # noqa: E402
from eles_uniform_weights import THRESHOLDS, compare  # noqa: E402
from referral_decomposition import ELES_DIR, ELES_LOSO, RAW_ZIP, VALIDATION, _eles_clusters, _state_frame  # noqa: E402
from retrieval_harness_validation_bus39 import HARNESS_RECORDS as BUS39_LOSO  # noqa: E402
from retrieval_harness_validation_bus39 import OUTPUT_PATH as VALIDATION_BUS39  # noqa: E402

from src.benchmarking import group_k_fold_test_groups  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
RUN_DIR: Final[Path] = PROJECT_DIR / "tmp" / "alpha_k"
SWEEP_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "alpha_k_sensitivity.csv"
NESTED_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "alpha_k_nested.csv"
ALPHAS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
BUS39_KS: Final[tuple[int, ...]] = (25, 50, 100, 200, 400)
ELES_K: Final[int] = 100
DEPLOYED: Final[tuple[int, float]] = (100, 1.0)
N_FOLDS: Final[int] = 5
QUANTILES: Final[tuple[float, ...]] = (0.50, 0.95, 0.99)
SELECTION_COLUMNS: Final[list[str]] = ["alpha", "state", "record_ordinal", "covered", "cct_true", "pred_selection"]
VERDICTS: Final[dict[str, str]] = {
    "decision_kernel_improves": "decision_selected_worse",  # not foreseen by the pre-specification
    "decision_uniform_improves": "decision_deployed_materially_worse",
    "decision_negligible": "decision_negligible",
    "decision_no_difference": "decision_no_material_difference",
}


def nearness(k: int, alpha: float) -> tuple[float, float]:
    """Distance of a setting from the deployed one for tie-breaking: alpha first, then K; alpha = 0 is farthest."""
    return (math.inf if alpha == 0 else abs(math.log(alpha)), abs(math.log(k / DEPLOYED[0])))


def common_support(records: pd.DataFrame) -> pd.DataFrame:
    """Records (state, record_ordinal) covered under every setting present, with each setting's selected estimate."""
    key = ["state", "record_ordinal"]
    n_settings = records.groupby(["K", "alpha"]).ngroups
    covered = records[records["covered"].astype(bool)]
    n = covered.groupby(key)["covered"].transform("size")
    return covered[n == n_settings]


def select(records: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    """The setting with the lowest MAE over the common support, nearest the deployed one on exact ties."""
    support = common_support(records)
    crit = (
        support.assign(ae=(support["pred_selection"] - support["cct_true"]).abs())
        .groupby(["K", "alpha"])["ae"]
        .agg(mae="mean", n_records="size")
        .reset_index()
    )
    crit["near"] = [nearness(int(k), float(a)) for k, a in zip(crit["K"], crit["alpha"], strict=True)]
    best = crit.sort_values(["mae", "near"], kind="stable").iloc[0]
    return int(best["K"]), float(best["alpha"]), crit.drop(columns="near")


def assert_folds_closed(states: pd.DataFrame, group_of: pd.Series, folds: list[frozenset[str]]) -> None:
    """ELES: no topology group spans two folds and no retrieved neighbour differs from its query in line status, so
    an inner query's neighbours never lie in the outer fold. `group_of` maps state to topology group."""
    fold_of = pd.Series({s: f for f, members in enumerate(folds) for s in members})
    per_group = fold_of.reindex(group_of.index).groupby(group_of).nunique()
    if (per_group > 1).any():
        raise AssertionError(f"{int((per_group > 1).sum())} topology groups span more than one fold")
    if (states["n_retrieved_line_mismatch"] > 0).any():
        raise AssertionError("a retrieved neighbour differs from its query in line status")


def describe(records: pd.DataFrame, states: pd.DataFrame) -> dict[str, float]:
    """Q1 quantities for one setting."""
    cov = records[records["covered"].astype(bool)]
    err = (cov["pred_selection"] - cov["cct_true"]).abs()
    out: dict[str, float] = {
        "n_records": len(records),
        "record_coverage": float(records["covered"].mean()),
        "complete_state_coverage": float(records.groupby("state")["covered"].all().mean()),
        "mae": float(err.mean()),
        **{f"abs_error_q{int(100 * q)}": float(err.quantile(q)) for q in QUANTILES},
        "top1": float((cov["gen_selected"] == cov["gen_true"]).mean()),
        "n_fallback_groups": float(states["n_fallback_groups"].sum()),
        "n_query_errors": float(states["query_error"].notna().sum()),
    }
    for tau in THRESHOLDS:
        t = round(1000 * tau)
        sf = _state_frame(records.rename(columns={"pred_screening_min": "est", "covered": "cov"}), "est", "cov", tau)
        out[f"referral_load_{t}ms"] = float(sf["referred"].mean())
        out[f"missed_unsafe_{t}ms"] = float((sf["unsafe"] & ~sf["referred"]).sum())
        out[f"recall_{t}ms"] = float((sf["unsafe"] & sf["referred"]).sum() / sf["unsafe"].sum())
    return out


def _run(name: str, rule: Rule, exclusions: dict[str, list[str]] | None, n_jobs: int, columns: list[str] | None = None):
    """One harness run at every alpha, stored under RUN_DIR and reused if present."""
    records_path, states_path = RUN_DIR / f"{name}.parquet", RUN_DIR / f"{name}_states.parquet"
    if not records_path.exists():
        logger.info(f"running {name}")
        records, states, _ = run_rule(rule, exclusions, n_jobs=n_jobs, alphas=ALPHAS)
        states.to_parquet(states_path, index=False)
        (records if columns is None else records[columns]).to_parquet(records_path, index=False)
    return pd.read_parquet(records_path), pd.read_parquet(states_path)


def _check_deployed(records: pd.DataFrame, validated: Path, label: str) -> None:
    """The deployed point of a sweep must equal the validated harness records."""
    mine = records[records["alpha"] == DEPLOYED[1]].drop(columns="alpha").reset_index(drop=True)
    ref = pd.read_parquet(validated)
    pd.testing.assert_frame_equal(mine[ref.columns], ref, check_exact=True, check_dtype=False)
    logger.info(f"{label}: deployed point equals {validated.name}")


def sweep_rows(dataset: str, availability: str, k: int, records: pd.DataFrame, states: pd.DataFrame) -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        d = describe(records[records["alpha"] == alpha], states[states["alpha"] == alpha])
        rows += [
            {"dataset": dataset, "availability": availability, "K": k, "alpha": alpha, "quantity": q, "estimate": v}
            for q, v in d.items()
        ]
    return rows


def summary_rows(sweep: pd.DataFrame) -> list[dict[str, Any]]:
    """Neighbourhood ranges and the hindsight best point, per dataset and availability."""
    rows = []
    for (dataset, availability), g in sweep.groupby(["dataset", "availability"]):
        wide = g.pivot_table(index=["K", "alpha"], columns="quantity", values="estimate")
        near = wide[
            (wide.index.get_level_values("alpha").isin([0.5, 1.0, 2.0]))
            & wide.index.get_level_values("K").isin([50, 100, 200, ELES_K])
        ]
        best = wide["mae"].idxmin()
        deployed = wide.loc[(DEPLOYED[0] if dataset == "bus39" else ELES_K, DEPLOYED[1])]
        facts = {
            "neighbourhood_mae_min": near["mae"].min(),
            "neighbourhood_mae_max": near["mae"].max(),
            "neighbourhood_missed_unsafe_200ms_min": near["missed_unsafe_200ms"].min(),
            "neighbourhood_missed_unsafe_200ms_max": near["missed_unsafe_200ms"].max(),
            "hindsight_best_K": best[0],
            "hindsight_best_alpha": best[1],
            "hindsight_best_mae": wide.loc[best, "mae"],
            "hindsight_best_minus_deployed_mae": wide.loc[best, "mae"] - deployed["mae"],
        }
        rows += [
            {"dataset": dataset, "availability": availability, "quantity": q, "estimate": float(v)}
            for q, v in facts.items()
        ]
    return rows


def assemble(sweep_records: dict[int, pd.DataFrame], choice: dict[int, tuple[int, float]], folds: list[frozenset[str]]):
    """Out-of-fold records: for each outer fold, its states' leave-one-state-out records at that fold's choice."""
    parts = []
    for f, members in enumerate(folds):
        k, alpha = choice[f]
        r = sweep_records[k]
        parts.append(r[(r["alpha"] == alpha) & r["state"].isin(members)].drop(columns="alpha"))
    return pd.concat(parts, ignore_index=True)


def nested(
    dataset: str,
    sweep_records: dict[int, pd.DataFrame],
    inner: dict[int, pd.DataFrame],
    folds: list[frozenset[str]],
    units: dict[str, np.ndarray | None],
) -> pd.DataFrame:
    """Q2 for one dataset: per-fold choices, then out-of-fold selected against deployed."""
    rows: list[dict[str, Any]] = []
    choice: dict[int, tuple[int, float]] = {}
    for f in range(len(folds)):
        k, alpha, crit = select(inner[f])
        choice[f] = (k, alpha)
        dep = crit[(crit["K"] == (DEPLOYED[0] if dataset == "bus39" else ELES_K)) & (crit["alpha"] == DEPLOYED[1])]
        rows += [
            {"dataset": dataset, "fold": f, "quantity": "chosen_K", "estimate": k},
            {"dataset": dataset, "fold": f, "quantity": "chosen_alpha", "estimate": alpha},
            {"dataset": dataset, "fold": f, "quantity": "inner_mae_chosen", "estimate": float(crit["mae"].min())},
            {"dataset": dataset, "fold": f, "quantity": "inner_mae_deployed", "estimate": float(dep["mae"].iloc[0])},
            {
                "dataset": dataset,
                "fold": f,
                "quantity": "inner_common_records",
                "estimate": int(crit["n_records"].iloc[0]),
            },
            {"dataset": dataset, "fold": f, "quantity": "n_outer_states", "estimate": len(folds[f])},
        ]
        logger.info(f"{dataset} fold {f}: chose K={k}, alpha={alpha}")
    deployed_k = DEPLOYED[0] if dataset == "bus39" else ELES_K
    deployed = sweep_records[deployed_k]
    deployed = deployed[deployed["alpha"] == DEPLOYED[1]].drop(columns="alpha")
    selected = assemble(sweep_records, choice, folds)
    out = compare(deployed, selected, units, "retrospective", same_coverage=False)
    q = out["quantity"].map(lambda s: VERDICTS.get(s, s.replace("uniform", "selected").replace("kernel", "deployed")))
    out = out.assign(quantity=q, dataset=dataset)
    same = sum(choice[f] == (deployed_k, DEPLOYED[1]) for f in choice)
    rows.append({"dataset": dataset, "quantity": "n_folds_choosing_deployed", "estimate": same})
    return pd.concat([pd.DataFrame(rows), out], ignore_index=True)


def main() -> None:
    configure_logging()
    for path in (VALIDATION, VALIDATION_BUS39):
        if not pd.read_csv(path)["passed"].all():
            raise RuntimeError(f"{path.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sweep: list[dict[str, Any]] = []

    # ELES: alpha only, retrospective and causal lag 0.
    state_ids = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl").index.astype(str)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(state_ids)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")
    eles_rule = Rule("eles_alpha", LINE_KEY, 0)
    eles, eles_states = _run("eles_retrospective", eles_rule, None, n_jobs)
    _check_deployed(eles, ELES_LOSO, "ELES retrospective")
    sweep += sweep_rows("eles", "retrospective", ELES_K, eles, eles_states)
    causal, causal_states = _run("eles_causal_lag0", eles_rule, causal_exclusions(timestamps, 0.0), n_jobs)
    sweep += sweep_rows("eles", "causal_lag0", ELES_K, causal, causal_states)

    # BUS39: alpha and K, leave one state out.
    bus39: dict[int, pd.DataFrame] = {}
    for k in BUS39_KS:
        rule = Rule(f"bus39_k{k}", LINE_KEY, 0, n_neighbors=k, dataset="bus39")
        bus39[k], states = _run(f"bus39_k{k}", rule, None, n_jobs)
        sweep += sweep_rows("bus39", "retrospective", k, bus39[k], states)
    _check_deployed(bus39[DEPLOYED[0]], BUS39_LOSO, "BUS39")
    sweep_frame = pd.DataFrame(sweep)
    pd.concat([sweep_frame, pd.DataFrame(summary_rows(sweep_frame))], ignore_index=True).to_csv(SWEEP_PATH, index=False)
    logger.info(f"Saved {SWEEP_PATH}")

    # Q2, ELES: topology-group folds; inner records are the sweep's own.
    eles_order = pd.Index(eles["state"].drop_duplicates())
    eles_units = _eles_clusters(eles_order)
    group_of = pd.Series(eles_units["topology_group"].astype(str), index=eles_order)
    # group_k_fold_test_groups returns the test folds' group labels; map them back to states.
    eles_folds = [
        frozenset(group_of.index[group_of.isin(g)]) for g in group_k_fold_test_groups(group_of, n_splits=N_FOLDS)
    ]
    assert_folds_closed(eles_states[eles_states["alpha"] == DEPLOYED[1]], group_of, eles_folds)
    eles_inner = {
        f: eles[~eles["state"].isin(members)].assign(K=ELES_K)[[*SELECTION_COLUMNS, "K"]]
        for f, members in enumerate(eles_folds)
    }
    results = [nested("eles", {ELES_K: eles}, eles_inner, eles_folds, eles_units)]

    # Q2, BUS39: state folds; every inner query rerun with the outer fold excluded.
    bus39_states = pd.Index(bus39[DEPLOYED[0]]["state"].drop_duplicates())
    bus39_folds = group_k_fold_test_groups(bus39_states, n_splits=N_FOLDS)
    bus39_inner: dict[int, pd.DataFrame] = {}
    for f, members in enumerate(bus39_folds):
        held_out = sorted(members)
        exclusions = {s: [*held_out, s] for s in bus39_states if s not in members}
        parts = []
        for k in BUS39_KS:
            rule = Rule(f"bus39_k{k}", LINE_KEY, 0, n_neighbors=k, dataset="bus39")
            records, _ = _run(f"bus39_fold{f}_k{k}", rule, exclusions, n_jobs, columns=SELECTION_COLUMNS)
            parts.append(records.assign(K=k))
        bus39_inner[f] = pd.concat(parts, ignore_index=True)
    results.append(nested("bus39", bus39, bus39_inner, bus39_folds, {"state": None}))

    out = pd.concat(results, ignore_index=True)
    out.to_csv(NESTED_PATH, index=False)
    for row in out[out["quantity"].str.startswith("decision_")].itertuples():
        logger.info(
            f"{row.dataset}: selected - deployed MAE {row.estimate:+.5f} s [{row.ci_low:+.5f}, {row.ci_high:+.5f}] "
            f"under {row.resampling_unit} -> {row.quantity}"
        )
    logger.info(f"Saved {NESTED_PATH}")


if __name__ == "__main__":
    main()
