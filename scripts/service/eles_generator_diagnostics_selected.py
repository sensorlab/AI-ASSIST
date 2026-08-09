"""ELES equivalent of generator_diagnostics_selected.py - retrieval-support diagnostics for
the highest-support (largest raw kernel-support mass) candidate generator at the recorded
true location, not the recorded true generator. Same motivation as the BUS39 script's
docstring (paper-sr issue raised by Codex review, ai2ai.md, 2026-08-09): Table 2's
diagnostics were computed from the oracle-conditioned report, same conditioning as Table 1's
oracle row, but advertised as characterizing the method generally.

Selection uses EstimationService._raw_kernel_mass, the same cross-group-comparable ranking
eles_deoracled_bound.py uses for its `pred_gen_deoracled_selection` column (not
LocationReportStats.weight_mass, which is normalized within each generator's own group and
not comparable across generators - see that script's and models.py's docstrings) - so the
"selected" generator here is identically defined to Table 1's ELES selected row.

Runs the real service in-process, in parallel across worker *processes*, same rationale as
eles_deoracled_bound.py.

Run from the repository root, e.g.:
    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only QDRANT_URL=":memory:" \\
        uv run python scripts/service/eles_generator_diagnostics_selected.py [limit] [n_jobs]
"""

from __future__ import annotations

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
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHA: Final[float] = 1.0


def _norm(value: Any) -> str:
    return str(value).strip().lower()


WorkItem = tuple[str, dict[str, Any], pd.DataFrame]


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    service = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for uid, state, tsa_subset in items:
        by_loc = service.estimate_by_location(state=state, exclude_uids=[uid], alpha=ALPHA)

        per_location_reports: dict[str, dict[str, Any]] = {}
        for loc_key, loc_group in by_loc.items():
            loc_norm = _norm(loc_key)
            per_location_reports[loc_norm] = {}
            for gen_key, report in loc_group.per_crit_gen.items():
                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                gen_norm = _norm(gen_key)
                mass = service._raw_kernel_mass(report.per_neighbor, alpha=ALPHA)
                per_location_reports[loc_norm][gen_norm] = (mass, report)

        for _, rec in tsa_subset.iterrows():
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            cct_true = float(rec["CCT"])
            row: dict[str, Any] = {"state": uid, "cct_true": cct_true, "loc_true": loc_true, "covered": False}

            loc_reports = per_location_reports.get(loc_true)
            if not loc_reports:
                rows.append(row)
                continue

            sel_gen = max(loc_reports, key=lambda g: loc_reports[g][0])
            _, sel_report = loc_reports[sel_gen]
            stats = sel_report.summary.stats
            distances = stats.distances

            row.update(
                {
                    "covered": True,
                    "gen_true": gen_true,
                    "gen_selected": sel_gen,
                    "gen_true_is_selected": gen_true == sel_gen,
                    "n_candidate_gens": len(loc_reports),
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

    config = get_qdrant_config()
    app_settings = get_app_settings()
    lf_path, tsa_path, _ = _dataset_paths(
        app_settings.data_dir, config.dataset_name, topology_variant=config.topology_variant
    )
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}, topology_variant={config.topology_variant!r}")

    safe_dataset = config.dataset_name.replace("/", "-")
    out_records = TMP_DIR / f"eles_generator_diagnostics_selected_{safe_dataset}_{config.topology_variant}.parquet"

    lf = pd.read_pickle(lf_path)
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

    cov = df[df["covered"]]
    logger.info(
        f"Covered: {len(cov):,}/{len(df):,} ({len(cov) / max(len(df), 1):.4%}); "
        f"gen_true_is_selected rate: {cov['gen_true_is_selected'].mean():.4%}"
    )

    df.to_parquet(out_records, index=False)
    logger.info(f"Saved {out_records}")


if __name__ == "__main__":
    main()
