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

Runs the real service in-process. Set QDRANT_URL=:memory: for a self-contained run, or point it at a
Qdrant instance for speed.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmarking import group_k_fold_test_groups  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import build_estimation_service  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
LF_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/lf.pkl"
TSA_PATH: Final[Path] = PROJECT_DIR / "datasets/bus39/interim/tsa.pkl"
OUT_CSV: Final[Path] = PROJECT_DIR / "generator_deoracled_bound.csv"
OUT_RECORDS: Final[Path] = PROJECT_DIR / "generator_deoracled_records.parquet"
ALPHA: Final[float] = 1.0
N_SPLITS: Final[int] = 5


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def main() -> None:
    configure_logging()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    lf: pd.DataFrame = pd.read_pickle(LF_PATH)
    tsa: pd.DataFrame = pd.read_pickle(TSA_PATH)
    tsa_by_state = {str(s): sub for s, sub in tsa.groupby("state", observed=True)}

    svc = build_estimation_service()
    logger.info("service built")

    folds = group_k_fold_test_groups(tsa["state"], n_splits=N_SPLITS)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    n_states = 0

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
            n_states += 1

            if n_states % 500 == 0:
                rate = (time.time() - t0) / n_states
                logger.info(f"{n_states} states, fold {fold}, {rate:.2f}s/state, {len(rows):,} records so far")

            state = {k: (None if pd.isna(v) else v) for k, v in state_row.items()}
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
