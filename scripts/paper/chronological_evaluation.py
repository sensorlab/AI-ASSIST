"""Strictly chronological evaluation with historically fitted preprocessing (analysis 09, R2.6 and R2.7c).

Pre-specified in the manuscript repository's revision/analyses/09-chronological-evaluation.md, committed before any
number here was computed. ELES only. Six expanding origins (1 August 2024 to 1 January 2025); at each, the production
scaler is fitted on the states recorded before the origin and the following month is the test block.

    R-cont    retrieval, causal lag 0 (every earlier state, including earlier states of the block), scaler at origin
    R-frozen  retrieval from the states before the origin only, scaler at origin
    R-global  the published causal lag-0 harness run, restricted to the test blocks (scaler fitted on all states)
    S-chrono  Table 3's ELES configurations (sqrt(p) ensembles, boosting, location median) trained on the history
    S-fold    Table 3's published five-fold predictions, restricted to the test blocks

Q1: R-cont against R-global (the scaler's effect). Q2: R-cont against each S-chrono model; R-frozen against each as
symmetric sensitivity, unread below 1,000 records. Q3: S-chrono against S-fold. Every comparison is paired, with
1,000 resamples under the widest of state, topology group and calendar day; material at 0.010 s with the interval
excluding zero, negligible below that, otherwise no material difference.

Each harness run and supervised fit is stored under tmp/chronological/ and reused if present.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/chronological_evaluation.py [n_jobs]
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import LINE_KEY, Rule, causal_exclusions, fit_scaler, run_rule  # noqa: E402
from alpha_k_sensitivity import describe  # noqa: E402
from matched_extratrees_comparison import _load_ml  # noqa: E402
from ml_benchmark import build_record_table, location_median_baseline, make_models, make_preprocessor  # noqa: E402
from referral_decomposition import ELES_CAUSAL0, ELES_DIR, RAW_ZIP, VALIDATION, _eles_clusters  # noqa: E402
from tuned_comparators import paired_rows  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import _dataset_paths, _make_scaler_for_dataset  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
RUN_DIR: Final[Path] = PROJECT_DIR / "tmp" / "chronological"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "chronological_evaluation.csv"
DATASET: Final[str] = "eles/2026-06"
ORIGINS: Final[tuple[pd.Timestamp, ...]] = tuple(pd.date_range("2024-08-01", "2025-01-01", freq="MS"))
MIN_FROZEN_RECORDS: Final[int] = 1000
SUPERVISED: Final[dict[str, str]] = {  # S-chrono model -> its Table 3 (S-fold) label
    "extra_trees": "extra_trees@sqrt",
    "random_forest": "random_forest@sqrt",
    "hist_gradient_boosting": "hist_gradient_boosting",
    "location_median": "location_median",
}
KEY: Final[list[str]] = ["state", "record_ordinal"]


def blocks(timestamps: pd.Series) -> list[tuple[pd.Timestamp, list[str], list[str]]]:
    """(origin, history states, test states) per origin; the last block runs to the end of the campaign."""
    out = []
    for k, origin in enumerate(ORIGINS):
        end = ORIGINS[k + 1] if k + 1 < len(ORIGINS) else None
        history = timestamps.index[timestamps < origin].astype(str).tolist()
        in_block = (timestamps >= origin) & (timestamps < end) if end is not None else timestamps >= origin
        out.append((origin, history, timestamps.index[in_block].astype(str).tolist()))
    return out


class HistoryScaler:
    """The production scaler, fitted on the history states only; build_record_table() calls fit_transform(lf)."""

    def __init__(self, history: list[str]):
        self.scaler = _make_scaler_for_dataset(DATASET)
        self.history = history

    def fit_transform(self, lf: pd.DataFrame) -> pd.DataFrame:
        return fit_scaler(self.scaler, lf, self.history)


def retrieval_arm(
    name: str, history: list[str], exclusions: dict[str, list[str]], n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-record output and per-state query errors (production raises when every kernel weight underflows)."""
    path, states_path = RUN_DIR / f"{name}.parquet", RUN_DIR / f"{name}_states.parquet"
    if not (path.exists() and states_path.exists()):
        logger.info(f"running {name} ({len(exclusions):,} queries, scaler fitted on {len(history):,} states)")
        rule = Rule(name, LINE_KEY, 0, scaler_fit_states=tuple(history))
        records, states, _ = run_rule(rule, exclusions, n_jobs=n_jobs)
        records.to_parquet(path, index=False)
        states[["state", "query_error"]].to_parquet(states_path, index=False)
    return pd.read_parquet(path), pd.read_parquet(states_path)


def coverage_loss(m: pd.DataFrame, errors: pd.DataFrame) -> tuple[int, int]:
    """(records, states) R-global covers and R-cont does not; stops unless R-cont's coverage is a subset of R-global's
    and every state that loses coverage carries a production query error (deviation 1)."""
    if (m["covered"] & ~m["covered_global"]).any():
        raise ValueError("R-cont covers records the published causal run does not")
    lost = m[~m["covered"] & m["covered_global"]]
    failed = set(errors.loc[errors["query_error"].notna(), "state"].astype(str))
    unexplained = set(lost["state"].astype(str)) - failed
    if unexplained:
        raise ValueError(
            f"{len(unexplained)} states lose coverage without a query error, e.g. {sorted(unexplained)[:3]}"
        )
    return len(lost), int(lost["state"].nunique())


def supervised_origin(k: int, history: list[str], test: list[str], lf: pd.DataFrame, tsa: pd.DataFrame, n_jobs: int):
    path = RUN_DIR / f"supervised_origin{k}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    logger.info(f"supervised origin {k}: training on {len(history):,} states")
    X, y, groups = build_record_table(lf, tsa, scaler=HistoryScaler(history), contingency_cols=("Location",))
    ordinal = groups.groupby(groups, sort=False).cumcount()
    train, test_mask = groups.isin(set(history)).to_numpy(), groups.isin(set(test)).to_numpy()
    out = pd.DataFrame({"state": groups[test_mask].to_numpy(), "record_ordinal": ordinal[test_mask].to_numpy()})
    out["location_median"] = location_median_baseline(X[train], y[train], X[test_mask])
    pre = make_preprocessor(["Location"])
    X_train, X_test = pre.fit_transform(X[train]), pre.transform(X[test_mask])
    models = make_models(max_features="sqrt", n_jobs=n_jobs)
    for name in ("extra_trees", "random_forest", "hist_gradient_boosting"):
        out[name] = models[name].fit(X_train, y[train]).predict(X_test)
        logger.info(f"supervised origin {k}: {name} fitted")
    out.to_parquet(path, index=False)
    return out


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    lf_path, tsa_path, _ = _dataset_paths(PROJECT_DIR / "datasets", DATASET)
    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)
    states = pd.Index(lf.index.astype(str))
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(states)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")
    causal = causal_exclusions(timestamps, 0.0)

    cont, cont_errors, frozen, supervised = [], [], [], []
    for k, (origin, history, test) in enumerate(blocks(timestamps)):
        later = timestamps.index[timestamps >= origin].astype(str).tolist()  # includes every test state itself
        records, errors = retrieval_arm(f"cont_origin{k}", history, {s: causal[s] for s in test}, n_jobs)
        cont.append(records.assign(origin=k))
        cont_errors.append(errors)
        frozen.append(
            retrieval_arm(f"frozen_origin{k}", history, dict.fromkeys(test, later), n_jobs)[0].assign(origin=k)
        )
        supervised.append(supervised_origin(k, history, test, lf, tsa, n_jobs).assign(origin=k))
    cont_r, frozen_r, sup = (pd.concat(x, ignore_index=True) for x in (cont, frozen, supervised))
    sup["state"] = sup["state"].astype(str)

    test_states = set(cont_r["state"])
    glob = pd.read_parquet(ELES_CAUSAL0)
    glob = glob[glob["state"].isin(test_states)]
    m = cont_r.merge(glob, on=KEY, suffixes=("", "_global"), validate="1:1")
    lost_records, lost_states = coverage_loss(m, pd.concat(cont_errors, ignore_index=True))
    logger.info(
        f"R-cont covers {int(m['covered'].sum()):,} of R-global's {int(m['covered_global'].sum()):,} test-block records; "
        f"{lost_records:,} records in {lost_states:,} states lost to kernel-weight underflow"
    )

    rows: list[dict[str, Any]] = [
        {"arm": "R-cont", "quantity": "records_lost_to_underflow", "estimate": lost_records},
        {"arm": "R-cont", "quantity": "states_lost_to_underflow", "estimate": lost_states},
    ]
    for arm, records in (("R-cont", cont_r), ("R-frozen", frozen_r), ("R-global", glob)):
        rows += [{"arm": arm, "quantity": q, "estimate": v} for q, v in describe(records).items()]
        for k, g in records.groupby(records["state"].map(dict(zip(cont_r["state"], cont_r["origin"], strict=True)))):
            rows += [
                {"arm": arm, "origin": int(k), "quantity": "record_coverage", "estimate": float(g["covered"].mean())},
                {
                    "arm": arm,
                    "origin": int(k),
                    "quantity": "mae",
                    "estimate": float(
                        (g.loc[g["covered"], "pred_selection"] - g.loc[g["covered"], "cct_true"]).abs().mean()
                    ),
                },
            ]

    fold = _load_ml("eles-2026-06")[0]
    fold["state"] = fold["state"].astype(str)
    common = (
        m[m["covered"]][[*KEY, "cct_true", "pred_selection", "pred_selection_global"]]
        .rename(columns={"pred_selection": "R_cont", "pred_selection_global": "R_global"})
        .merge(sup.drop(columns="origin"), on=KEY, validate="1:1")
        .merge(
            fold[[*KEY, "cct_true", *SUPERVISED.values()]].rename(
                columns={"cct_true": "cct_true_fold", **{v: f"{k}_fold" for k, v in SUPERVISED.items()}}
            ),
            on=KEY,
            validate="1:1",
        )
    )
    # Harness and supervised tables number each state's records independently; equal labels show they align.
    if not np.allclose(common["cct_true"], common["cct_true_fold"], rtol=0.0, atol=1e-12):
        raise ValueError("record keys do not align between the harness and the supervised predictions")
    units = _eles_clusters(pd.Index(common["state"].drop_duplicates()))
    paired = [paired_rows("ELES", common, "R_cont", "R_global", units, decide=True)]
    paired += [paired_rows("ELES", common, "R_cont", model, units, decide=True) for model in SUPERVISED]
    paired += [paired_rows("ELES", common, model, f"{model}_fold", units, decide=True) for model in SUPERVISED]

    fz = frozen_r[frozen_r["covered"]][[*KEY, "cct_true", "pred_selection"]].rename(
        columns={"pred_selection": "R_frozen"}
    )
    fz = fz.merge(sup.drop(columns="origin"), on=KEY, validate="1:1")
    rows.append({"arm": "R-frozen", "quantity": "n_covered_records", "estimate": len(fz)})
    if len(fz) >= MIN_FROZEN_RECORDS:
        fz_units = _eles_clusters(pd.Index(fz["state"].drop_duplicates()))
        paired += [paired_rows("ELES", fz, "R_frozen", model, fz_units, decide=True) for model in SUPERVISED]
    else:
        logger.info(f"R-frozen covers {len(fz):,} records, below {MIN_FROZEN_RECORDS:,}: reported without a reading")

    out = pd.concat([pd.DataFrame(rows), *paired], ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)
    for row in out[out["quantity"].astype(str).str.startswith("decision_")].itertuples():
        logger.info(
            f"{row.quantity}: {row.estimate:+.5f} [{row.ci_low:+.5f}, {row.ci_high:+.5f}] under {row.resampling_unit}"
        )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
