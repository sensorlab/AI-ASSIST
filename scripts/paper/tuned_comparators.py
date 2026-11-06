"""Systematically tuned comparators under one feature-budget rule (analysis 08, R2.7a-b).

Pre-specified in the manuscript repository's revision/analyses/08-tuned-comparators.md, committed before any number
here was computed. Outer folds are Table 3's five GroupKFold folds by pre-fault state (asserted against the stored
predictions). Within each outer training set one inner split holds out 20% of the training states (GroupShuffleSplit,
seed 42); every grid configuration is fitted on the inner training part (tree ensembles with 100 trees) and scored by
MAE on the inner held-out records; the best, first-listed on ties, is refitted on the outer training set (300 trees)
and predicts the outer fold. Features, preprocessing and the location median are scripts/evaluation/ml_benchmark.py's.

Modes:
    timing   time the tree ensembles on ELES outer fold 0, run its boosting tuning for real, project the full run
    run      fit everything (resumable, per dataset, model and fold under tmp/tuned_comparators/) and write the CSV

Run from the repository root:
    nice -n 19 uv run python scripts/paper/tuned_comparators.py timing|run [n_jobs] [n_budgets_dropped]
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eles_uniform_weights import decision  # noqa: E402
from matched_extratrees_comparison import CONFIGS, _load_ml, _load_retrieval, _summary_row  # noqa: E402
from ml_benchmark import build_record_table, location_median_baseline, make_preprocessor  # noqa: E402
from referral_decomposition import DECISION_UNITS, _bootstrap, _eles_clusters, _ratio  # noqa: E402

from src.benchmarking import CONTINGENCY_CATEGORICAL_COLUMNS, group_k_fold_indices  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import _dataset_paths, _make_scaler_for_dataset  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
RUN_DIR: Final[Path] = PROJECT_DIR / "tmp" / "tuned_comparators"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "tuned_comparators.csv"
DATASETS: Final[dict[str, str]] = {"BUS39": "bus39", "ELES": "eles/2026-06"}
N_SPLITS: Final[int] = 5
INNER_HOLDOUT: Final[float] = 0.2
SEED: Final[int] = 42
TUNING_TREES: Final[int] = 100
FINAL_TREES: Final[int] = 300
N_BOOTSTRAP: Final[int] = 1000
LABEL_RESOLUTION: Final[float] = 0.010
BUDGETS: Final[tuple[str | float, ...]] = ("sqrt", 0.1, 0.3, 1.0)  # dropped from the end by the fallback
LEAVES: Final[tuple[int, ...]] = (1, 5)
# Deviation 1: iterations are tuned by early stopping on the inner held-out states, up to HGB_MAX_ITER.
HGB_GRID: Final[list[dict[str, Any]]] = [
    {"learning_rate": lr, "max_leaf_nodes": leaves} for lr in (0.05, 0.1, 0.2) for leaves in (31, 127, 255)
]
HGB_MAX_ITER: Final[int] = 1000
HGB_PATIENCE: Final[int] = 10
TREE_MODELS: Final[tuple[str, ...]] = ("extra_trees", "random_forest")
MODELS: Final[tuple[str, ...]] = (*TREE_MODELS, "hist_gradient_boosting")
TIMING_SUBSAMPLE: Final[float] = 0.1
PROJECTION_LIMIT_H: Final[float] = 48.0
# The untuned Table 3 row each tuned model is compared against, per dataset (the rows the paper headlines).
UNTUNED: Final[dict[str, dict[str, str]]] = {
    "BUS39": {
        "extra_trees": "extra_trees@1.0",
        "random_forest": "random_forest@1.0",
        "hist_gradient_boosting": "hist_gradient_boosting",
    },
    "ELES": {
        "extra_trees": "extra_trees@sqrt",
        "random_forest": "random_forest@sqrt",
        "hist_gradient_boosting": "hist_gradient_boosting",
    },
}
VERDICTS: Final[dict[str, str]] = {  # decision()'s labels, read for the first arm of a paired comparison
    "kernel_improves": "better",
    "uniform_improves": "worse",
    "negligible": "negligible",
    "no_difference": "no_material_difference",
}


def grid(model: str, n_dropped: int = 0) -> list[dict[str, Any]]:
    """The fixed grid for a model, with the `n_dropped` largest feature budgets removed (tractability fallback)."""
    if model == "hist_gradient_boosting":
        return HGB_GRID
    budgets = BUDGETS[: len(BUDGETS) - n_dropped]
    if not budgets:
        raise ValueError("the fallback removed every feature budget")
    return [{"max_features": mf, "min_samples_leaf": leaf} for mf in budgets for leaf in LEAVES]


def make(model: str, config: dict[str, Any], *, n_estimators: int, n_jobs: int) -> Any:
    if model == "hist_gradient_boosting":
        params = {"max_iter": HGB_MAX_ITER, "early_stopping": False, **config}
        return HistGradientBoostingRegressor(
            loss="absolute_error", l2_regularization=1e-3, n_iter_no_change=HGB_PATIENCE, random_state=SEED, **params
        )
    cls = ExtraTreesRegressor if model == "extra_trees" else RandomForestRegressor
    return cls(n_estimators=n_estimators, n_jobs=n_jobs, random_state=SEED, **config)


def tune_one(model: str, config: dict[str, Any], X_tr, y_tr, X_va, y_va, n_jobs: int) -> tuple[dict[str, Any], float]:
    """Fit one grid configuration on the inner training part; return the configuration to refit and its inner MAE.
    Boosting stops early on the inner held-out states and the refit takes the iteration count it reached."""
    if model == "hist_gradient_boosting":
        est = make(model, {**config, "early_stopping": True}, n_estimators=0, n_jobs=n_jobs)
        est.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        config = {**config, "max_iter": int(est.n_iter_)}
    else:
        est = make(model, config, n_estimators=TUNING_TREES, n_jobs=n_jobs).fit(X_tr, y_tr)
    return config, float(np.mean(np.abs(est.predict(X_va) - np.asarray(y_va))))


def select(scores: list[float]) -> int:
    """Index of the lowest inner MAE; the first-listed configuration wins an exact tie."""
    return min(range(len(scores)), key=lambda i: (scores[i], i))


def check_folds(groups: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]], stored: pd.DataFrame) -> None:
    """Every state's outer fold must equal the fold stored with Table 3's supervised predictions."""
    mine = pd.Series(-1, index=pd.Index(groups.astype(str).unique()))
    for f, (_, test_idx) in enumerate(folds):
        mine[groups.iloc[test_idx].astype(str).unique()] = f
    theirs = stored.drop_duplicates("state").set_index("state")["fold"]
    theirs.index = theirs.index.astype(str)
    joined = pd.concat([mine.rename("mine"), theirs.rename("theirs")], axis=1)
    bad = int((joined["mine"] != joined["theirs"]).sum())
    if bad:
        raise AssertionError(f"{bad} states are in a different outer fold than Table 3's")


def load(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    lf_path, tsa_path, _ = _dataset_paths(PROJECT_DIR / "datasets", dataset)
    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)
    return build_record_table(lf, tsa, scaler=_make_scaler_for_dataset(dataset), contingency_cols=("Location",))


def _categorical(X: pd.DataFrame) -> list[str]:
    return [c for c in CONTINGENCY_CATEGORICAL_COLUMNS if c in X.columns]


def inner_split(groups: pd.Series, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=INNER_HOLDOUT, random_state=SEED)
    a, b = next(splitter.split(train_idx, groups=groups.iloc[train_idx]))
    return train_idx[a], train_idx[b]


def fit_fold(
    name: str, model: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, train_idx, test_idx, configs, n_jobs: int
) -> None:
    """Tune on the inner split, refit the choice on the outer training set, store predictions and the tuning log."""
    pred_path, log_path = RUN_DIR / f"{name}.parquet", RUN_DIR / f"{name}.json"
    if pred_path.exists():
        return
    inner_tr, inner_va = inner_split(groups, train_idx)
    pre = make_preprocessor(_categorical(X))
    X_tr, X_va = pre.fit_transform(X.iloc[inner_tr]), pre.transform(X.iloc[inner_va])
    tuned, scores, seconds = [], [], []
    for config in configs:
        t = time.time()
        fitted, score = tune_one(model, config, X_tr, y.iloc[inner_tr], X_va, y.iloc[inner_va], n_jobs)
        tuned.append(fitted)
        scores.append(score)
        seconds.append(time.time() - t)
        logger.info(f"{name}: {fitted} inner MAE {scores[-1]:.5f} ({seconds[-1]:.0f} s)")
    del X_tr, X_va
    best = select(scores)
    pre = make_preprocessor(_categorical(X))
    X_train, X_test = pre.fit_transform(X.iloc[train_idx]), pre.transform(X.iloc[test_idx])
    t = time.time()
    est = make(model, tuned[best], n_estimators=FINAL_TREES, n_jobs=n_jobs).fit(X_train, y.iloc[train_idx])
    refit_s = time.time() - t
    pd.DataFrame(
        {
            "state": groups.iloc[test_idx].astype(str).to_numpy(),
            "record_ordinal": groups.groupby(groups, sort=False).cumcount().iloc[test_idx].to_numpy(),
            "cct_true": y.iloc[test_idx].to_numpy(),
            "cct_pred": est.predict(X_test),
        }
    ).to_parquet(pred_path, index=False)
    log_path.write_text(
        json.dumps(
            {"configs": tuned, "inner_mae": scores, "seconds": seconds, "chosen": best, "refit_seconds": refit_s}
        )
    )


def timing(n_jobs: int) -> None:
    """Time the tree ensembles on a subsample of ELES outer fold 0's inner training set, run the real boosting tuning
    and refit on that fold (deviation 1; reused by the full run), and project the full run."""
    X, y, groups = load(DATASETS["ELES"])
    slug = CONFIGS["ELES"]["slug"]
    folds = list(group_k_fold_indices(groups, n_splits=N_SPLITS))
    check_folds(groups, folds, pd.read_parquet(PROJECT_DIR / "tmp" / f"ml_benchmark_predictions-{slug}.parquet"))
    train_idx, test_idx = folds[0]
    inner_tr, _ = inner_split(groups, train_idx)
    states = groups.iloc[inner_tr].unique()
    keep = set(np.random.default_rng(SEED).choice(states, size=int(TIMING_SUBSAMPLE * len(states)), replace=False))
    sub = inner_tr[groups.iloc[inner_tr].isin(keep).to_numpy()]
    pre = make_preprocessor(_categorical(X))
    X_sub = pre.fit_transform(X.iloc[sub])
    p = X_sub.shape[1]
    n_full, n_sub = len(inner_tr), len(sub)
    scale = (n_full / n_sub) * math.log(n_full) / math.log(n_sub)
    logger.info(f"timing on {n_sub:,} of {n_full:,} inner training records, {p:,} features after preprocessing")
    per_fit: dict[str, dict[str, float]] = {}
    for model in TREE_MODELS:
        measured = {}
        for mf in ("sqrt", 0.1):
            t = time.time()
            make(model, {"max_features": mf, "min_samples_leaf": 1}, n_estimators=TUNING_TREES, n_jobs=n_jobs).fit(
                X_sub, y.iloc[sub]
            )
            measured[mf] = (time.time() - t) * scale
        per_feature = measured[0.1] / (0.1 * p)  # cost per feature evaluated per split, from the 0.1 budget
        per_fit[model] = {
            "sqrt": measured["sqrt"],
            0.1: measured[0.1],
            0.3: per_feature * 0.3 * p,
            1.0: per_feature * p,
        }
    del X_sub
    name = f"{slug}_hist_gradient_boosting_fold0"
    fit_fold(name, "hist_gradient_boosting", X, y, groups, train_idx, test_idx, HGB_GRID, n_jobs)
    log = json.loads((RUN_DIR / f"{name}.json").read_text())
    hgb_fold = sum(log["seconds"]) + log["refit_seconds"]
    rows = []
    for n_dropped in range(len(BUDGETS)):
        total = 0.0
        for model in TREE_MODELS:
            fits = sum(per_fit[model][c["max_features"]] for c in grid(model, n_dropped))
            refit = max(per_fit[model][c["max_features"]] for c in grid(model, n_dropped)) * 3 / (1 - INNER_HOLDOUT)
            total += N_SPLITS * (fits + refit)
        total += N_SPLITS * hgb_fold
        rows.append(
            {
                "n_budgets_dropped": n_dropped,
                "budgets": str(BUDGETS[: len(BUDGETS) - n_dropped]),
                "eles_hours": total / 3600,
            }
        )
    out = pd.DataFrame(rows)
    print(json.dumps({m: {str(k): round(v) for k, v in d.items()} for m, d in per_fit.items()}, indent=1))
    print(
        f"hist_gradient_boosting, outer fold 0 tuning and refit: {hgb_fold:.0f} s; chosen {log['configs'][log['chosen']]}"
    )
    print(out.to_string(index=False))
    fits = out[out["eles_hours"] <= PROJECTION_LIMIT_H]
    print(
        "pre-specified fallback:",
        "stop, sqrt alone does not fit" if fits.empty else f"drop {int(fits['n_budgets_dropped'].min())} budget(s)",
    )
    out.to_csv(RUN_DIR / "timing.csv", index=False)


def paired_rows(
    dataset: str,
    common: pd.DataFrame,
    a: str,
    b: str,
    units: dict[str, np.ndarray | None],
    decide: bool | None = None,
) -> pd.DataFrame:
    """MAE(a) - MAE(b) on the common records, per resampling unit, plus a decision row reading the rule for `a`
    (by default only when `a` is retrieval, analysis 08's rule; analysis 09 decides other pairs too)."""
    states = pd.Index(common["state"].drop_duplicates())
    g = (
        common.assign(ae_a=(common[a] - common["cct_true"]).abs(), ae_b=(common[b] - common["cct_true"]).abs(), n=1.0)
        .groupby("state", sort=False)[["ae_a", "ae_b", "n"]]
        .sum()
        .reindex(states)
    )
    arrays = {c: g[c].to_numpy() for c in g.columns}
    stats = {"diff": lambda x, w: _ratio("ae_a", "n")(x, w) - _ratio("ae_b", "n")(x, w)}
    est = stats["diff"](arrays, np.ones(len(states)))
    rows = []
    for unit, clusters in units.items():
        lo, hi = _bootstrap(arrays, stats, clusters, np.random.default_rng(SEED), n_resamples=N_BOOTSTRAP)["diff"]
        rows.append(
            {
                "dataset": dataset,
                "quantity": f"mae_{a}_minus_{b}",
                "resampling_unit": unit,
                "decides": unit in DECISION_UNITS,
                "estimate": est,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    out = pd.DataFrame(rows)
    if decide if decide is not None else a == "retrieval":
        # decision() credits the second arm when (second - first) is positive; give it (b - a) so that a positive
        # value means `a` has the lower error.
        flipped = out.assign(estimate=-out["estimate"], ci_low=-out["ci_high"], ci_high=-out["ci_low"])
        d = decision(flipped)
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "dataset": dataset,
                            "quantity": f"decision_{a}_vs_{b}_{VERDICTS[d['verdict']]}",
                            "resampling_unit": d["resampling_unit"],
                            "decides": True,
                            "estimate": est,
                            "ci_low": -d["ci_high"],
                            "ci_high": -d["ci_low"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return out


def analyse(dataset: str, slug: str) -> pd.DataFrame:
    config = CONFIGS[dataset]
    ml, _, budgets = _load_ml(config["slug"])
    retrieval = _load_retrieval(config["retrieval_path"], config["retrieval_prediction"])
    tuned = pd.concat(
        [
            pd.concat([pd.read_parquet(RUN_DIR / f"{slug}_{model}_fold{f}.parquet") for f in range(N_SPLITS)])
            .rename(columns={"cct_pred": f"{model}_tuned"})[["state", "record_ordinal", f"{model}_tuned"]]
            .set_index(["state", "record_ordinal"])
            for model in MODELS
        ],
        axis=1,
    ).reset_index()
    m = retrieval.merge(ml.drop(columns=["fold", "cct_true"]), on=["state", "record_ordinal"], validate="1:1")
    m = m.merge(tuned, on=["state", "record_ordinal"], validate="1:1")
    if len(m) != len(retrieval):
        raise ValueError(f"{dataset}: record populations differ")
    common = m[m["covered"] & m["retrieval"].notna()].copy()
    coverage = len(common) / len(m)
    rows = [
        _summary_row(
            common,
            dataset=dataset,
            model=col,
            population="common_retrieval_covered",
            prediction_col=col,
            coverage=coverage,
            max_features=budgets.get(col, "tuned" if col.endswith("_tuned") else "not applicable"),
        )
        for col in ["retrieval", *[f"{x}_tuned" for x in MODELS], *UNTUNED[dataset].values(), "location_median"]
    ]
    units = {"state": None} if dataset == "BUS39" else _eles_clusters(pd.Index(common["state"].drop_duplicates()))
    paired = [paired_rows(dataset, common, "retrieval", f"{x}_tuned", units) for x in MODELS]
    paired.append(paired_rows(dataset, common, "retrieval", "location_median", units))
    paired += [paired_rows(dataset, common, f"{x}_tuned", UNTUNED[dataset][x], units) for x in MODELS]
    chosen = []
    for model in MODELS:
        for f in range(N_SPLITS):
            log = json.loads((RUN_DIR / f"{slug}_{model}_fold{f}.json").read_text())
            chosen.append(
                {
                    "dataset": dataset,
                    "quantity": f"chosen_{model}",
                    "fold": f,
                    "config": json.dumps(log["configs"][log["chosen"]]),
                    "estimate": log["inner_mae"][log["chosen"]],
                }
            )
    return pd.concat([pd.DataFrame(rows).assign(quantity="metrics"), *paired, pd.DataFrame(chosen)], ignore_index=True)


def run(n_jobs: int, n_dropped: int) -> None:
    for dataset, name in DATASETS.items():
        slug = CONFIGS[dataset]["slug"]
        X, y, groups = load(name)
        folds = list(group_k_fold_indices(groups, n_splits=N_SPLITS))
        check_folds(groups, folds, pd.read_parquet(PROJECT_DIR / "tmp" / f"ml_benchmark_predictions-{slug}.parquet"))
        logger.info(f"{dataset}: outer folds equal Table 3's")
        for model in MODELS:
            for f, (train_idx, test_idx) in enumerate(folds):
                fit_fold(
                    f"{slug}_{model}_fold{f}", model, X, y, groups, train_idx, test_idx, grid(model, n_dropped), n_jobs
                )
        del X, y, groups
    out = pd.concat([analyse(d, CONFIGS[d]["slug"]) for d in DATASETS], ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)
    for row in out[out["quantity"].astype(str).str.startswith("decision_")].itertuples():
        logger.info(
            f"{row.dataset}: {row.quantity} {row.estimate:+.5f} [{row.ci_low:+.5f}, {row.ci_high:+.5f}] under {row.resampling_unit}"
        )
    logger.info(f"Saved {OUTPUT_PATH}")


def main() -> None:
    configure_logging()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    if mode == "timing":
        timing(n_jobs)
    elif mode == "run":
        run(n_jobs, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        raise ValueError(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
