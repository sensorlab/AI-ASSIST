"""Generator-de-oracled BUS39 benchmark.

Every error figure in the paper conditions on the true critical generator, which is a
time-domain-simulation outcome the supervised baselines never receive. This script removes that
conditioning while keeping the fault location, which the baselines *do* receive as an input feature
and which a deployed screening procedure enumerates.

Two targets are computed per record, both at the record's true location:

* selection - pick the generator group with the largest retrieved weight mass, score its estimate.
  Directly comparable to the supervised baselines.
* screening - take the minimum estimate across generator groups as a conservative stability margin.
  Matches the deployment story (operators inspect low-CCT slices) and needs no generator identification.

Also records the rank of the true critical generator in the weight-mass ordering, which explains
whatever the selection number turns out to be.

Runs the real service in-process, in parallel across worker *processes* (not threads, see
scripts/evaluation/full_deoracled_bound.py's docstring for why): each worker builds its own
EstimationService and embedded :memory: Qdrant collection once and processes its share of
states sequentially.

Run from the repository root:
    uv run python scripts/evaluation/generator_deoracled_bound.py [limit_per_fold] [n_jobs]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("DATASET_NAME", "bus39")
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("DATA_DIR", "./datasets")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmarking import group_k_fold_test_groups  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import build_estimation_service  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib, .parquet) go to
# tmp/, CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
LF_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/lf.pkl"
TSA_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/tsa.pkl"
# BUS39_BENCHMARK_SPLIT mirrors ELES_BENCHMARK_SPLIT in eles_generator_diagnostics_selected.py,
# so both datasets can be evaluated under the same protocol. "group-k-fold" withholds the query's
# whole fold; "leave-one-state-out" withholds only the query state, which is the protocol the
# paper reports for ELES. Leaving BUS39 on folds while ELES ran leave-one-state-out meant the two
# headline accuracies were not measured the same way. Artifacts take a suffix so a run under one
# protocol cannot overwrite the other.
SPLIT: Final[str] = os.environ.get("BUS39_BENCHMARK_SPLIT", "group-k-fold")
if SPLIT not in {"group-k-fold", "leave-one-state-out"}:
    raise ValueError(f"BUS39_BENCHMARK_SPLIT must be group-k-fold or leave-one-state-out, got {SPLIT!r}")
_SUFFIX: Final[str] = "" if SPLIT == "group-k-fold" else "_loso"
OUT_CSV: Final[Path] = PAPER_DATA_DIR / f"generator_deoracled_bound{_SUFFIX}.csv"
OUT_RECORDS: Final[Path] = TMP_DIR / f"generator_deoracled_records{_SUFFIX}.parquet"
ALPHA: Final[float] = 1.0
N_SPLITS: Final[int] = 5


def _norm(value: Any) -> str:
    return str(value).strip().lower()


WorkItem = tuple[int, str, dict[str, Any], pd.DataFrame, list[str]]


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    """One worker process's share of work. Builds its own EstimationService exactly once,
    not once per state - process-parallel rather than thread-parallel for the same reason
    as scripts/evaluation/full_deoracled_bound.py: sharing one in-process Qdrant client across
    threads is untested in this codebase, and BUS39 is small enough (~21.8k points) that
    duplicating the embedded collection per worker is cheap."""
    svc = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for fold, uid, state, subset, excluded_sorted in items:
        out = svc.estimate_by_location(state=state, exclude_uids=excluded_sorted, alpha=ALPHA)
        by_loc = {_norm(k): v for k, v in out.items()}

        for _, rec in subset.iterrows():
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            location_group = by_loc.get(loc_true)
            if location_group is None:
                rows.append({"state": uid, "fold": fold, "cct_true": float(rec["CCT"]), "covered": False})
                continue

            # crit_gen_likelihood is the service's own raw (never-renormalized) kernel
            # mass per generator, comparable across generators - exactly what this used
            # to hand-recompute locally via a since-removed _group_mass() helper.
            masses: dict[str, float] = {}
            estimates: dict[str, float] = {}
            for gen_key, report in location_group.per_crit_gen.items():
                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                gen_norm = _norm(gen_key)
                masses[gen_norm] = location_group.crit_gen_likelihood.get(gen_key, 0.0)
                estimates[gen_norm] = float(est)
            if not estimates:
                rows.append({"state": uid, "fold": fold, "cct_true": float(rec["CCT"]), "covered": False})
                continue

            order = sorted(masses, key=masses.get, reverse=True)
            sel_gen = order[0]
            rank = order.index(gen_true) + 1 if gen_true in order else -1

            rows.append(
                {
                    "state": uid,
                    "fold": fold,
                    "covered": True,
                    "cct_true": float(rec["CCT"]),
                    "gen_true": gen_true,
                    "n_candidate_gens": len(estimates),
                    "gen_true_rank": rank,
                    "pred_oracle_gen": estimates.get(gen_true),
                    "pred_selection": estimates[sel_gen],
                    "pred_screening_min": min(estimates.values()),
                }
            )
    return rows


def main() -> None:
    configure_logging()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    lf: pd.DataFrame = pd.read_pickle(LF_PATH)
    tsa: pd.DataFrame = pd.read_pickle(TSA_PATH)
    tsa_by_state = {str(s): sub for s, sub in tsa.groupby("state", observed=True)}

    if SPLIT == "group-k-fold":
        folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)
    else:
        # One "fold" per state, excluding only that state: the query cannot retrieve itself,
        # but every other state stays available, as in the ELES leave-one-state-out arm.
        folds = [{uid} for uid in dict.fromkeys(str(s) for s in tsa["state"])]
    logger.info(f"split={SPLIT}, {len(folds)} exclusion sets")

    # Index the state rows once. The previous form rescanned every state for each fold, which is
    # cheap at five folds and quadratic at leave-one-state-out's one fold per state (21,783 folds
    # x 21,783 rows). Build a lookup instead and walk each fold's own members.
    state_rows = {str(state_id): row for state_id, row in lf.iterrows()}

    work: list[WorkItem] = []
    for fold, excluded in enumerate(folds):
        excluded_sorted = sorted(excluded)
        n_fold = 0
        for uid in excluded_sorted:
            state_row = state_rows.get(uid)
            if state_row is None:
                continue
            subset = tsa_by_state.get(uid)
            if subset is None or subset.empty:
                continue

            # `limit` is per fold, so every fold is represented in a subsampled run. Under
            # leave-one-state-out a fold holds one state, so `limit` caps folds instead: the
            # outer loop is truncated below rather than each fold being trimmed.
            n_fold += 1
            if limit and n_fold > limit:
                break

            state = {k: (None if pd.isna(v) else v) for k, v in state_row.items()}
            work.append((fold, uid, state, subset, excluded_sorted))
        if limit and SPLIT == "leave-one-state-out" and len(work) >= limit:
            break

    n_states = len(work)
    logger.info(f"{n_states} (fold, state) tasks queued across {n_jobs} worker processes")

    chunks: list[list[WorkItem]] = [[] for _ in range(n_jobs)]
    for i, item in enumerate(work):
        chunks[i % n_jobs].append(item)

    t0 = time.time()
    chunk_results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_process_chunk)(chunk) for chunk in chunks if chunk)
    rows: list[dict[str, Any]] = [row for chunk_rows in chunk_results for row in chunk_rows]
    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    logger.info(f"{n_states} states, {len(df)} records, {elapsed:.1f}s ({elapsed / max(n_states, 1):.3f}s/state)")

    cov = df[df["covered"]].copy()
    summary: list[dict[str, Any]] = []
    for label, col in (
        ("oracle_generator_and_location", "pred_oracle_gen"),
        ("deoracled_generator_selection", "pred_selection"),
        ("deoracled_generator_screening_min", "pred_screening_min"),
    ):
        sub = cov[cov[col].notna()]
        err = (sub["cct_true"] - sub[col]).abs().to_numpy(dtype=float)
        summary.append(
            {
                "variant": label,
                "mae": float(err.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "n": int(len(sub)),
            }
        )

    ranked = cov[cov["gen_true_rank"] > 0]
    gen_stats = {
        "n_states": n_states,
        "n_records": int(len(df)),
        "coverage": float(cov.shape[0] / max(len(df), 1)),
        "gen_top1": float((ranked["gen_true_rank"] == 1).mean()),
        "gen_top3": float((ranked["gen_true_rank"] <= 3).mean()),
        "mean_candidate_gens": float(cov["n_candidate_gens"].mean()),
        "uniform_top1": float((1.0 / cov["n_candidate_gens"]).mean()),
        "elapsed_s": elapsed,
    }

    out = pd.DataFrame(summary)
    out.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_RECORDS, index=False)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        logger.warning(f"parquet write skipped: {exc}")

    print(out.to_string(index=False))
    print(json.dumps(gen_stats, indent=2))


if __name__ == "__main__":
    main()
