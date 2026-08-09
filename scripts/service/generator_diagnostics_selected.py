"""Retrieval-support diagnostics for BUS39 under the *selected* (non-oracle) generator,
not the recorded true one - paper-sr issue raised by Codex review (ai2ai.md, 2026-08-09):
Table 2's diagnostics (nAURC per candidate diagnostic) were computed from
scripts/service/benchmark.py's per-record reports, which select the report for the recorded
true critical generator (`outputs_by_crit_gen.get(crit_gen_true)`). That is the same
oracle-conditioning already disclosed for Table 1's oracle row - appropriate for studying
the oracle-vs-selected contrast there, but Table 2's diagnostics are advertised as
characterizing the method generally, not that one conditioning. This script computes the
same LocationReportStats diagnostic bundle for the highest-support (largest raw
kernel-support mass) candidate generator instead - the same population and CCT estimate as
generator_deoracled_bound.py's `deoracled_generator_selection` row - so
bootstrap_risk_coverage.py's nAURC methodology can be rerun against it unchanged.

Column-for-column mirrors benchmark.py's flattening of LocationReportStats
(location_weight_mass, n_neighbors, n_eff, n_eff_fraction, neighborhood_compactness,
n_unique_states, cct_weighted_std, cct_distance_correlation, distance_min/mean/median/
spread/norm) so the downstream nAURC script needs no field-name translation.

Runs the real service in-process, in parallel across worker *processes*, same rationale as
generator_deoracled_bound.py (each worker builds its own EstimationService once).

Run from the repository root:
    uv run python scripts/service/generator_diagnostics_selected.py [limit_per_fold] [n_jobs]
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
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
LF_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/lf.pkl"
TSA_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/tsa.pkl"
OUT_RECORDS: Final[Path] = TMP_DIR / "generator_diagnostics_selected_bus39.parquet"
ALPHA: Final[float] = 1.0
N_SPLITS: Final[int] = 5


def _norm(value: Any) -> str:
    return str(value).strip().lower()


WorkItem = tuple[int, str, dict[str, Any], pd.DataFrame, list[str]]


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    svc = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for fold, uid, state, subset, excluded_sorted in items:
        out = svc.estimate_by_location(state=state, exclude_uids=excluded_sorted, alpha=ALPHA)
        by_loc = {_norm(k): v for k, v in out.items()}

        for _, rec in subset.iterrows():
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            cct_true = float(rec["CCT"])
            row: dict[str, Any] = {"state": uid, "fold": fold, "cct_true": cct_true, "covered": False}

            location_group = by_loc.get(loc_true)
            if location_group is None:
                rows.append(row)
                continue

            masses: dict[str, float] = {}
            reports: dict[str, Any] = {}
            for gen_key, report in location_group.per_crit_gen.items():
                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                gen_norm = _norm(gen_key)
                masses[gen_norm] = location_group.crit_gen_likelihood.get(gen_key, 0.0)
                reports[gen_norm] = report

            if not reports:
                rows.append(row)
                continue

            sel_gen = max(masses, key=masses.get)
            sel_report = reports[sel_gen]
            stats = sel_report.summary.stats
            distances = stats.distances

            row.update(
                {
                    "covered": True,
                    "gen_true": gen_true,
                    "gen_selected": sel_gen,
                    "gen_true_is_selected": gen_true == sel_gen,
                    "n_candidate_gens": len(reports),
                    "cct_weighted_per_location": float(sel_report.summary.cct_weighted),
                    "location_weight_mass": stats.weight_mass,
                    "n_neighbors": stats.n,
                    "n_eff": stats.n_eff,
                    "n_eff_fraction": (stats.n_eff / stats.n) if stats.n > 0 else None,
                    "neighborhood_compactness": stats.neighborhood_compactness,
                    "n_unique_states": stats.n_unique_states,
                    "cct_weighted_std": stats.cct_weighted_std,
                    "cct_distance_correlation": stats.cct_distance_correlation,
                    "distance_min": distances.get("min"),
                    "distance_mean": distances.get("mean"),
                    "distance_median": distances.get("median"),
                    "distance_spread": distances.get("spread"),
                    "distance_norm": distances.get("norm"),
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    configure_logging()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    lf: pd.DataFrame = pd.read_pickle(LF_PATH)
    tsa: pd.DataFrame = pd.read_pickle(TSA_PATH)
    tsa_by_state = {str(s): sub for s, sub in tsa.groupby("state", observed=True)}

    folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)

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

    cov = df[df["covered"]]
    logger.info(
        f"Covered: {len(cov):,}/{len(df):,} ({len(cov) / max(len(df), 1):.4%}); "
        f"gen_true_is_selected rate: {cov['gen_true_is_selected'].mean():.4%}"
    )

    df.to_parquet(OUT_RECORDS, index=False)
    logger.info(f"Saved {OUT_RECORDS}")


if __name__ == "__main__":
    main()
