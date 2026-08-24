"""Full de-oracled BUS39 benchmark: removes both the critical-generator and the
fault-location oracle.

generator_deoracled_bound.py removes only the generator oracle, keeping the true location as
given - matching a deployed screen that enumerates candidate locations but has no way to know
the outcome generator in advance. This script goes one step further and also removes the
location oracle, so no part of the contingency is disclosed to the retrieval step: for each
pre-fault state it pools every retrieved (location, generator) group and scores across the
full pool, not just the true location's slice.

Two targets are computed per record, at the record's true (location, generator):

* selection - pick the (location, generator) group with the largest raw kernel mass across the
  full pool, score its estimate. Directly comparable to a supervised baseline given only the
  pre-fault state, no contingency descriptors at all.
* screening - take the minimum estimate across every (location, generator) group, the
  conservative deployment rule: enumerate everything retrieved, flag the worst case.

Mass is compared using EstimationService._raw_kernel_mass (sum of K(distance) over each
group's own raw neighbor distances) - the same quantity crit_gen_likelihood is built from
inside estimate_by_location, and the one mass measure in this codebase that's actually
comparable across different (location, generator) groups. LocationReportStats.weight_mass is
NOT usable here: it's normalized within each generator's own neighbor set (see
LocationGroupReport's docstring in src/domain/estimation/models.py), so it isn't comparable
across groups the way raw kernel mass is.

Also records the joint rank of the true (location, generator) pair in that pooled mass
ordering, generalizing generator_deoracled_bound.py's gen_true_rank to both axes. The
"oracle_location_and_generator" summary row is a sanity check against the main benchmark's
service_location_strict numbers (scripts/evaluation/benchmark.py) - both score the same
(true location, true generator) target, just through different code paths, so they should
agree closely; a large discrepancy would flag a bug in one of the two.

Runs the real service in-process, in parallel across worker *processes* (not threads): each
worker builds its own EstimationService and embedded :memory: Qdrant collection once and
processes its share of states sequentially. BUS39 is small (~21.8k points), so duplicating the
collection across workers is cheap - deliberately safer than sharing one in-process Qdrant
client across threads, whose concurrency-safety for this codebase's local-mode client is
undocumented and untested; no other script here does that either (eles_benchmark.py's
equivalent loop is sequential, and benchmark.py's parallelism goes through the HTTP API, not a
shared in-process client).

Run from the repository root:
    uv run python scripts/evaluation/full_deoracled_bound.py [limit_per_fold] [n_jobs]
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
# tmp/, CSV summaries the manuscript reports go to results/data/ (tracked).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
LF_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/lf.pkl"
TSA_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/tsa.pkl"
OUT_CSV: Final[Path] = PAPER_DATA_DIR / "full_deoracled_bound.csv"
OUT_RECORDS: Final[Path] = TMP_DIR / "full_deoracled_records.parquet"
ALPHA: Final[float] = 1.0
N_SPLITS: Final[int] = 5

WorkItem = tuple[int, str, dict[str, Any], pd.DataFrame, list[str]]


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    """One worker process's share of work. Builds its own EstimationService exactly once,
    not once per state - see the module docstring for why this is process-, not
    thread-, parallel."""
    svc = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for fold, uid, state, subset, excluded_sorted in items:
        by_loc = svc.estimate_by_location(state=state, exclude_uids=excluded_sorted, alpha=ALPHA)

        # Pool every (location, generator) group into one flat, cross-location-comparable
        # ranking.
        pool: dict[tuple[str, str], tuple[float, float]] = {}  # (loc, gen) -> (mass, estimate)
        for loc_key, loc_group in by_loc.items():
            loc_norm = _norm(loc_key)
            for gen_key, report in loc_group.per_crit_gen.items():
                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                mass = svc._raw_kernel_mass(report.per_neighbor, alpha=ALPHA)
                pool[(loc_norm, _norm(gen_key))] = (mass, float(est))

        for _, rec in subset.iterrows():
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            true_key = (loc_true, gen_true)

            if not pool:
                rows.append({"state": uid, "fold": fold, "cct_true": float(rec["CCT"]), "covered": False})
                continue

            order = sorted(pool, key=lambda k: pool[k][0], reverse=True)
            sel_key = order[0]
            rank = order.index(true_key) + 1 if true_key in pool else -1

            rows.append(
                {
                    "state": uid,
                    "fold": fold,
                    "covered": True,
                    "cct_true": float(rec["CCT"]),
                    "loc_true": loc_true,
                    "gen_true": gen_true,
                    "n_candidate_pairs": len(pool),
                    "true_pair_rank": rank,
                    "pred_oracle_loc_gen": pool.get(true_key, (None, None))[1],
                    "pred_selection": pool[sel_key][1],
                    "pred_screening_min": min(v[1] for v in pool.values()),
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

    folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)

    # Build the flat work list first (cheap, no service needed), then split it into n_jobs
    # chunks so each worker process builds EstimationService exactly once, not once per state.
    work: list[WorkItem] = []
    for fold, excluded in enumerate(folds):
        excluded_sorted = sorted(excluded)
        n_fold = 0
        for state_id, state_row in lf.iterrows():
            uid = str(state_id)
            if uid not in excluded:
                continue
            subset = tsa_by_state.get(uid)
            if subset is None or subset.empty:
                continue

            # `limit` is per fold, so every fold is represented in a subsampled run.
            n_fold += 1
            if limit and n_fold > limit:
                break

            state = {k: (None if pd.isna(v) else v) for k, v in state_row.items()}
            work.append((fold, uid, state, subset, excluded_sorted))

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
        ("oracle_location_and_generator", "pred_oracle_loc_gen"),
        ("deoracled_full_selection", "pred_selection"),
        ("deoracled_full_screening_min", "pred_screening_min"),
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

    ranked = cov[cov["true_pair_rank"] > 0]
    stats = {
        "n_states": n_states,
        "n_records": int(len(df)),
        "coverage": float(cov.shape[0] / max(len(df), 1)),
        "pair_top1": float((ranked["true_pair_rank"] == 1).mean()),
        "pair_top3": float((ranked["true_pair_rank"] <= 3).mean()),
        "mean_candidate_pairs": float(cov["n_candidate_pairs"].mean()),
        "uniform_top1": float((1.0 / cov["n_candidate_pairs"]).mean()),
        "elapsed_s": elapsed,
    }

    out = pd.DataFrame(summary)
    out.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_RECORDS, index=False)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        logger.warning(f"parquet write skipped: {exc}")

    print(out.to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
