"""Chronologically trained comparators at analysis 08's tuned configurations (analysis 10, R2.7c after R2.7a).

Pre-specified in the manuscript repository's revision/analyses/10-chronological-tuned.md, committed before any number
here was computed. ELES only. Analysis 09's six monthly origins, history-fitted scaler and cached retrieval arms
(R-cont, R-frozen) are reused unchanged; the new arm S-tuned trains extra trees, random forest and gradient boosting on
each origin's history at the configuration analysis 08 chose in most outer folds.

Q1 (decides): R-cont against each S-tuned model on the records R-cont covers. Q2 (reading): S-tuned against 08's tuned
five-fold predictions. Q3 (reading): S-tuned against 09's untuned S-chrono. Sensitivity (reading): R-frozen against
S-tuned. Paired, 1,000 resamples under the widest of state, topology group and calendar day; material at 0.010 s with
the interval excluding zero, negligible below that, otherwise no material difference.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/chronological_tuned.py [n_jobs]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from chronological_evaluation import DATASET, KEY, RUN_DIR, HistoryScaler, blocks  # noqa: E402
from ml_benchmark import build_record_table, make_preprocessor  # noqa: E402
from referral_decomposition import RAW_ZIP, _eles_clusters  # noqa: E402
from tuned_comparators import FINAL_TREES, MODELS, make, paired_rows  # noqa: E402
from tuned_comparators import OUTPUT_PATH as TUNED_CSV  # noqa: E402
from tuned_comparators import RUN_DIR as TUNED_DIR  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import _dataset_paths  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "chronological_tuned.csv"
N_ORIGINS: Final[int] = 6
N_FOLDS: Final[int] = 5
CONFIGS: Final[dict[str, dict[str, Any]]] = {  # fixed by the pre-specification; checked against 08's CSV
    "extra_trees": {"max_features": 0.1, "min_samples_leaf": 1},
    "random_forest": {"max_features": 0.1, "min_samples_leaf": 1},
    "hist_gradient_boosting": {"learning_rate": 0.1, "max_leaf_nodes": 255, "max_iter": 1000},
}
GRID_ORDER: Final[dict[str, tuple[Any, ...]]] = {  # 08's grid order, for the first-listed tie-break
    "max_features": ("sqrt", 0.1, 0.3, 1.0),
    "min_samples_leaf": (1, 5),
    "learning_rate": (0.05, 0.1, 0.2),
    "max_leaf_nodes": (31, 127, 255),
}


def majority_config(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """Per hyperparameter, the value chosen in most folds; a tie goes to the value listed first in 08's grid
    (iteration counts, which have no grid, to the larger)."""
    out = {}
    for name in chosen[0]:
        counts = Counter(c[name] for c in chosen)
        top = max(counts.values())
        tied = [v for v, n in counts.items() if n == top]
        order = GRID_ORDER.get(name)
        out[name] = min(tied, key=order.index) if order else max(tied)
    return out


def check_configs() -> None:
    rows = pd.read_csv(TUNED_CSV)
    for model, expected in CONFIGS.items():
        chosen = rows[(rows["dataset"] == "ELES") & (rows["quantity"] == f"chosen_{model}")]["config"]
        if len(chosen) != N_FOLDS:
            raise ValueError(f"{model}: {len(chosen)} chosen configurations in {TUNED_CSV.name}, expected {N_FOLDS}")
        got = majority_config([json.loads(c) for c in chosen])
        if got != expected:
            raise ValueError(f"{model}: majority configuration {got} differs from the pre-specified {expected}")


def tuned_origin(k: int, history: list[str], test: list[str], lf: pd.DataFrame, tsa: pd.DataFrame, n_jobs: int):
    path = RUN_DIR / f"tuned_origin{k}.parquet"
    if path.exists():
        return pd.read_parquet(path), []
    logger.info(f"tuned origin {k}: training on {len(history):,} states")
    X, y, groups = build_record_table(lf, tsa, scaler=HistoryScaler(history), contingency_cols=("Location",))
    ordinal = groups.groupby(groups, sort=False).cumcount()
    train, test_mask = groups.isin(set(history)).to_numpy(), groups.isin(set(test)).to_numpy()
    out = pd.DataFrame({"state": groups[test_mask].to_numpy(), "record_ordinal": ordinal[test_mask].to_numpy()})
    out["cct_true"] = y[test_mask].to_numpy()
    pre = make_preprocessor(["Location"])
    X_train, X_test = pre.fit_transform(X[train]), pre.transform(X[test_mask])
    timings = []
    for model in MODELS:
        t = time.time()
        est = make(model, CONFIGS[model], n_estimators=FINAL_TREES, n_jobs=n_jobs).fit(X_train, y[train])
        out[f"{model}_tuned"] = est.predict(X_test)
        timings.append({"origin": k, "model": model, "quantity": "fit_seconds", "estimate": time.time() - t})
        logger.info(f"tuned origin {k}: {model} fitted ({timings[-1]['estimate']:.0f} s)")
    out.to_parquet(path, index=False)
    return out, timings


def load_fold_predictions() -> pd.DataFrame:
    """08's tuned five-fold predictions, one column per model."""
    out = None
    for model in MODELS:
        p = pd.concat(
            [pd.read_parquet(TUNED_DIR / f"eles-2026-06_{model}_fold{f}.parquet") for f in range(N_FOLDS)],
            ignore_index=True,
        ).rename(columns={"cct_pred": f"{model}_fold", "cct_true": f"cct_true_{model}"})
        p["state"] = p["state"].astype(str)
        out = p if out is None else out.merge(p, on=KEY, validate="1:1")
    return out


def compared(retrieval: pd.DataFrame, column: str, tuned, chrono, fold=None) -> pd.DataFrame:
    """Covered retrieval records joined 1:1 to every prediction; stops if a record is missing or labels disagree."""
    r = retrieval[retrieval["covered"]][[*KEY, "cct_true", "pred_selection"]].rename(columns={"pred_selection": column})
    r["state"] = r["state"].astype(str)
    out = r.merge(tuned, on=KEY, suffixes=("", "_tuned"), validate="1:1").merge(chrono, on=KEY, validate="1:1")
    labels = ["cct_true_tuned"]
    if fold is not None:
        out = out.merge(fold, on=KEY, validate="1:1")
        labels += [f"cct_true_{m}" for m in MODELS]
    if len(out) != len(r):
        raise ValueError(f"{len(r) - len(out):,} covered {column} records have no prediction")
    for c in labels:  # harness and record tables number each state's records independently
        if not np.allclose(out["cct_true"], out[c], rtol=0.0, atol=1e-12):
            raise ValueError(f"record keys do not align between {column} and {c}")
    return out


def main() -> None:
    configure_logging()
    check_configs()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    lf_path, tsa_path, _ = _dataset_paths(PROJECT_DIR / "datasets", DATASET)
    lf = pd.read_pickle(lf_path)
    with sqlite3.connect(tsa_path) as conn:
        tsa = pd.read_sql_query("SELECT * FROM tsa", conn)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(pd.Index(lf.index.astype(str)))
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")

    tuned, rows = [], []
    for k, (_, history, test) in enumerate(blocks(timestamps)):
        preds, timings = tuned_origin(k, history, test, lf, tsa, n_jobs)
        tuned.append(preds)
        rows += timings
    if len(tuned) != N_ORIGINS:
        raise ValueError(f"{len(tuned)} origins, expected {N_ORIGINS}")
    tuned_p = pd.concat(tuned, ignore_index=True)
    tuned_p["state"] = tuned_p["state"].astype(str)
    chrono = pd.concat([pd.read_parquet(RUN_DIR / f"supervised_origin{k}.parquet") for k in range(N_ORIGINS)])
    chrono["state"] = chrono["state"].astype(str)
    chrono = chrono[[*KEY, *MODELS]].rename(columns={m: f"{m}_chrono" for m in MODELS})
    cont = pd.concat([pd.read_parquet(RUN_DIR / f"cont_origin{k}.parquet") for k in range(N_ORIGINS)])
    frozen = pd.concat([pd.read_parquet(RUN_DIR / f"frozen_origin{k}.parquet") for k in range(N_ORIGINS)])

    common = compared(cont, "R_cont", tuned_p, chrono, load_fold_predictions())
    logger.info(f"{len(common):,} R-cont covered records compared (analysis 09 Q2: 128,614)")
    units = _eles_clusters(pd.Index(common["state"].drop_duplicates()))
    rows += [{"arm": "R-cont", "quantity": "n_records", "estimate": len(common)}]
    for c in ["R_cont", *(f"{m}_{s}" for m in MODELS for s in ("tuned", "chrono", "fold"))]:
        rows.append({"arm": c, "quantity": "mae", "estimate": float((common[c] - common["cct_true"]).abs().mean())})
    paired = [paired_rows("ELES", common, "R_cont", f"{m}_tuned", units, decide=True) for m in MODELS]
    paired += [paired_rows("ELES", common, f"{m}_tuned", f"{m}_fold", units, decide=False) for m in MODELS]
    paired += [paired_rows("ELES", common, f"{m}_tuned", f"{m}_chrono", units, decide=False) for m in MODELS]

    fz = compared(frozen, "R_frozen", tuned_p, chrono)
    rows.append({"arm": "R-frozen", "quantity": "n_records", "estimate": len(fz)})
    fz_units = _eles_clusters(pd.Index(fz["state"].drop_duplicates()))
    for m in MODELS:  # a reading under the rule, not a decision (pre-specification)
        p = paired_rows("ELES", fz, "R_frozen", f"{m}_tuned", fz_units, decide=True)
        p["decides"] = False
        paired.append(p)

    out = pd.concat([pd.DataFrame(rows), *paired], ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)
    for row in out[out["quantity"].astype(str).str.startswith("decision_")].itertuples():
        logger.info(
            f"{row.quantity}: {row.estimate:+.5f} [{row.ci_low:+.5f}, {row.ci_high:+.5f}] under {row.resampling_unit}"
        )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
