"""De-oracling ladder for ELES, matching scripts/evaluation/full_deoracled_bound.py and
generator_deoracled_bound.py's BUS39 analysis, adapted for ELES's dataset access pattern
(SqliteRecordStore-backed tsa.db, not a flat tsa.pkl) and topology-variant selection
(DATASET_NAME=eles/2026-06, TOPOLOGY_VARIANT=lines_only|slovenia_only).

This was deliberately deferred while BUS39's result was pending (DIRECTION.md Decisions:
"BUS39 first, revisit once #4's full BUS39 result is in") - that result is now in, so this
fills the gap. The old draft's own reasoning for skipping this on ELES was that its ~72
generator groups make the BUS39-style argument potentially inapplicable; this script tests
that rather than assuming it, and reports the actual candidate-pool sizes observed.

Both de-oracling levels are computed from a single service.estimate_by_location() call per
query state, avoiding two separate full passes:

* generator-only de-oracled (location kept, matching a screen that enumerates candidate
  locations but doesn't know the outcome generator in advance) - selection (largest raw
  kernel mass within the true location) and screening_min (minimum estimate within it).
* fully de-oracled (both location and generator removed) - pools every retrieved
  (location, generator) group for the query and does the same two aggregations across the
  full pool, using EstimationService._raw_kernel_mass (comparable across groups, unlike
  LocationReportStats.weight_mass - see models.py and full_deoracled_bound.py's docstring).

Runs the real service in-process, in parallel across worker *processes* (not threads), same
rationale as full_deoracled_bound.py: each worker builds its own EstimationService and
embedded :memory: Qdrant collection once.

Run from the repository root, e.g.:
    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only QDRANT_URL=":memory:" \\
        uv run python scripts/evaluation/eles_deoracled_bound.py [limit] [n_jobs]

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only QDRANT_URL=":memory:" \\
        ELES_BENCHMARK_SAMPLE_STATES=300 uv run python scripts/evaluation/eles_deoracled_bound.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("QDRANT_URL", ":memory:")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402
from src.config.settings import get_app_settings  # noqa: E402
from src.domain.estimation.service import _dataset_paths, build_estimation_service  # noqa: E402
from src.services.qdrant.config import get_qdrant_config  # noqa: E402
from src.services.sqlite_store import SqliteRecordStore  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHA: Final[float] = 1.0
SAMPLE_SEED: Final[int] = int(os.environ.get("ELES_BENCHMARK_SAMPLE_SEED", "42"))


def _norm(value: Any) -> str:
    return str(value).strip().lower()


WorkItem = tuple[str, dict[str, Any], pd.DataFrame]


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    """One worker process's share of work. Builds its own EstimationService exactly once -
    see the module docstring for why this is process-, not thread-, parallel."""
    service = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for uid, state, tsa_subset in items:
        by_loc = service.estimate_by_location(state=state, exclude_uids=[uid], alpha=ALPHA)

        # Pool every (location, generator) group into one flat, cross-location-comparable
        # ranking (full de-oracling), and index by location for the generator-only variant.
        pool: dict[tuple[str, str], tuple[float, float]] = {}  # (loc, gen) -> (mass, estimate)
        per_location_pool: dict[str, dict[str, tuple[float, float]]] = {}
        for loc_key, loc_group in by_loc.items():
            loc_norm = _norm(loc_key)
            per_location_pool[loc_norm] = {}
            for gen_key, report in loc_group.per_crit_gen.items():
                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                gen_norm = _norm(gen_key)
                mass = service._raw_kernel_mass(report.per_neighbor, alpha=ALPHA)
                pool[(loc_norm, gen_norm)] = (mass, float(est))
                per_location_pool[loc_norm][gen_norm] = (mass, float(est))

        for _, rec in tsa_subset.iterrows():
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            true_key = (loc_true, gen_true)

            row: dict[str, Any] = {
                "state": uid,
                "cct_true": float(rec["CCT"]),
                "loc_true": loc_true,
                "gen_true": gen_true,
            }

            # Generator-only de-oracled: true location given, pool restricted to it.
            loc_pool = per_location_pool.get(loc_true)
            if not loc_pool:
                row["gen_deoracled_covered"] = False
            else:
                order = sorted(loc_pool, key=lambda g: loc_pool[g][0], reverse=True)
                row["gen_deoracled_covered"] = True
                row["n_candidate_gens"] = len(loc_pool)
                row["gen_true_rank"] = order.index(gen_true) + 1 if gen_true in loc_pool else -1
                row["pred_oracle_gen_and_loc"] = loc_pool.get(gen_true, (None, None))[1]
                row["pred_gen_deoracled_selection"] = loc_pool[order[0]][1]
                row["pred_gen_deoracled_screening_min"] = min(v[1] for v in loc_pool.values())

            # Fully de-oracled: pool over every retrieved (location, generator) pair.
            if not pool:
                row["full_deoracled_covered"] = False
            else:
                order = sorted(pool, key=lambda k: pool[k][0], reverse=True)
                row["full_deoracled_covered"] = True
                row["n_candidate_pairs"] = len(pool)
                row["true_pair_rank"] = order.index(true_key) + 1 if true_key in pool else -1
                row["pred_oracle_loc_and_gen"] = pool.get(true_key, (None, None))[1]
                row["pred_full_deoracled_selection"] = pool[order[0]][1]
                row["pred_full_deoracled_screening_min"] = min(v[1] for v in pool.values())

            rows.append(row)
    return rows


def main() -> None:
    configure_logging()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    config = get_qdrant_config()
    app_settings = get_app_settings()
    lf_path, tsa_path, _ = _dataset_paths(
        app_settings.data_dir, config.dataset_name, topology_variant=config.topology_variant
    )
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}, topology_variant={config.topology_variant!r}")

    safe_dataset = config.dataset_name.replace("/", "-")
    sample_states = os.environ.get("ELES_BENCHMARK_SAMPLE_STATES")
    variant_tag = config.topology_variant
    if sample_states:
        variant_tag = f"{config.topology_variant}-sample{sample_states}"
        if SAMPLE_SEED != 42:
            variant_tag = f"{variant_tag}-seed{SAMPLE_SEED}"
    out_csv = PAPER_DATA_DIR / f"eles_deoracled_bound_{safe_dataset}_{variant_tag}.csv"
    out_records = TMP_DIR / f"eles_deoracled_records_{safe_dataset}_{variant_tag}.parquet"

    lf = pd.read_pickle(lf_path)
    if sample_states:
        original_n = len(lf)
        n = min(int(sample_states), original_n)
        rng = np.random.default_rng(SAMPLE_SEED)
        sampled_ids = set(rng.choice(sorted(lf.index.astype(str)), size=n, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(sampled_ids)]
        logger.info(f"Query-side subsample: {n} of {original_n} states as queries (seed={SAMPLE_SEED})")

    tsa_store = SqliteRecordStore(tsa_path, table="tsa")
    tsa = tsa_store.fetch(list(lf.index.astype(str)))
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}

    work: list[WorkItem] = []
    n_taken = 0
    for state_id, state_row in lf.iterrows():
        uid = str(state_id)
        tsa_subset = tsa_by_state.get(uid)
        if tsa_subset is None or tsa_subset.empty:
            continue
        if limit and n_taken >= limit:
            break
        n_taken += 1
        state = {k: (None if pd.isna(v) else v) for k, v in state_row.items()}
        work.append((uid, state, tsa_subset))

    n_states = len(work)
    logger.info(f"{n_states} states queued across {n_jobs} worker processes")

    chunks: list[list[WorkItem]] = [[] for _ in range(n_jobs)]
    for i, item in enumerate(work):
        chunks[i % n_jobs].append(item)

    t0 = time.time()
    chunk_results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_process_chunk)(chunk) for chunk in chunks if chunk)
    rows: list[dict[str, Any]] = [row for chunk_rows in chunk_results for row in chunk_rows]
    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    logger.info(f"{n_states} states, {len(df)} records, {elapsed:.1f}s ({elapsed / max(n_states, 1):.3f}s/state)")

    summary: list[dict[str, Any]] = []

    gen_cov = df[df["gen_deoracled_covered"]].copy()
    for label, col in (
        ("oracle_generator_and_location", "pred_oracle_gen_and_loc"),
        ("deoracled_generator_selection", "pred_gen_deoracled_selection"),
        ("deoracled_generator_screening_min", "pred_gen_deoracled_screening_min"),
    ):
        sub = gen_cov[gen_cov[col].notna()]
        err = (sub["cct_true"] - sub[col]).abs().to_numpy(dtype=float)
        summary.append(
            {
                "variant": label,
                "mae": float(err.mean()) if len(err) else float("nan"),
                "rmse": float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
                "n": int(len(sub)),
            }
        )

    full_cov = df[df["full_deoracled_covered"]].copy()
    for label, col in (
        ("oracle_location_and_generator", "pred_oracle_loc_and_gen"),
        ("deoracled_full_selection", "pred_full_deoracled_selection"),
        ("deoracled_full_screening_min", "pred_full_deoracled_screening_min"),
    ):
        sub = full_cov[full_cov[col].notna()]
        err = (sub["cct_true"] - sub[col]).abs().to_numpy(dtype=float)
        summary.append(
            {
                "variant": label,
                "mae": float(err.mean()) if len(err) else float("nan"),
                "rmse": float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
                "n": int(len(sub)),
            }
        )

    gen_ranked = gen_cov[gen_cov.get("gen_true_rank", -1) > 0] if "gen_true_rank" in gen_cov else gen_cov.iloc[0:0]
    full_ranked = (
        full_cov[full_cov.get("true_pair_rank", -1) > 0] if "true_pair_rank" in full_cov else full_cov.iloc[0:0]
    )

    stats = {
        "n_states": n_states,
        "n_records": int(len(df)),
        "gen_deoracled_coverage": float(gen_cov.shape[0] / max(len(df), 1)),
        "full_deoracled_coverage": float(full_cov.shape[0] / max(len(df), 1)),
        "gen_top1": float((gen_ranked["gen_true_rank"] == 1).mean()) if len(gen_ranked) else float("nan"),
        "gen_top3": float((gen_ranked["gen_true_rank"] <= 3).mean()) if len(gen_ranked) else float("nan"),
        "mean_candidate_gens": float(gen_cov["n_candidate_gens"].mean())
        if "n_candidate_gens" in gen_cov
        else float("nan"),
        "pair_top1": float((full_ranked["true_pair_rank"] == 1).mean()) if len(full_ranked) else float("nan"),
        "pair_top3": float((full_ranked["true_pair_rank"] <= 3).mean()) if len(full_ranked) else float("nan"),
        "mean_candidate_pairs": float(full_cov["n_candidate_pairs"].mean())
        if "n_candidate_pairs" in full_cov
        else float("nan"),
        "elapsed_s": elapsed,
    }

    out = pd.DataFrame(summary)
    out.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_records, index=False)
    except Exception as exc:  # pragma: no cover - parquet engine optional
        logger.warning(f"parquet write skipped: {exc}")

    print(out.to_string(index=False))
    print(json.dumps(stats, indent=2))
    logger.info(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
