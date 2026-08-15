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

For the matched five-fold comparison, add:
    ELES_BENCHMARK_SPLIT=group-k-fold
The grouped-fold artifact receives a `_group_kfold` filename suffix.

ELES_TEMPORAL_EXCLUSION_HOURS=H additionally excludes every state whose acquisition timestamp
is within H hours of the query's, not just the query itself. ELES's same-topology groups are
largely contiguous hourly bursts spanning one to three days, so leave-one-state-out leaves a
query's immediate temporal neighbours in the reference library and retrieval can return
essentially the same operating state an hour earlier. This option measures how much of the
reported accuracy depends on that near-duplicate recall. It applies to leave-one-group-out
only, since group-k-fold already removes a whole fold, and the artifact receives an
`_excl{H}h` filename suffix so a control run cannot overwrite the baseline.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_eles_state_timestamps  # noqa: E402

from src.benchmarking import group_k_fold_test_groups  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402
from src.config.settings import get_app_settings  # noqa: E402
from src.domain.estimation.service import _dataset_paths, build_estimation_service  # noqa: E402
from src.services.qdrant.config import get_qdrant_config  # noqa: E402
from src.services.sqlite_store import SqliteRecordStore  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
# Raw archive is the only persistent location for ELES state timestamps, needed by the
# optional temporal-exclusion control (ELES_TEMPORAL_EXCLUSION_HOURS).
RAW_ZIP: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06" / "raw" / "Podatki_DSA.zip"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHA: Final[float] = 1.0


def _norm(value: Any) -> str:
    return str(value).strip().lower()


WorkItem = tuple[int | None, str, dict[str, Any], pd.DataFrame, list[str]]


def _process_chunk(items: list[WorkItem]) -> list[dict[str, Any]]:
    service = build_estimation_service()
    rows: list[dict[str, Any]] = []

    for fold, uid, state, tsa_subset, excluded_uids in items:
        by_loc = service.estimate_by_location(state=state, exclude_uids=excluded_uids, alpha=ALPHA)
        excluded_norm = {_norm(excluded) for excluded in excluded_uids}

        per_location_reports: dict[str, dict[str, Any]] = {}
        for loc_key, loc_group in by_loc.items():
            loc_norm = _norm(loc_key)
            per_location_reports[loc_norm] = {}
            for gen_key, report in loc_group.per_crit_gen.items():
                leaked = excluded_norm & {_norm(included) for included in report.included_state_ids}
                if leaked:
                    raise RuntimeError(f"Data leakage: excluded states returned by retrieval: {sorted(leaked)}")

                est = getattr(report.summary, "cct_weighted", None)
                if est is None:
                    continue
                gen_norm = _norm(gen_key)
                mass = service._raw_kernel_mass(report.per_neighbor, alpha=ALPHA)
                per_location_reports[loc_norm][gen_norm] = (mass, report)

        for record_ordinal, (_, rec) in enumerate(tsa_subset.iterrows()):
            loc_true = _norm(rec["Location"])
            gen_true = _norm(rec["Crit_gen"])
            cct_true = float(rec["CCT"])
            row: dict[str, Any] = {
                "state": uid,
                "record_ordinal": record_ordinal,
                "cct_true": cct_true,
                "loc_true": loc_true,
                "covered": False,
            }
            if fold is not None:
                row["fold"] = fold

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
    split = os.environ.get("ELES_BENCHMARK_SPLIT", "leave-one-group-out")
    if split not in {"leave-one-group-out", "group-k-fold"}:
        raise ValueError(f"Unsupported ELES_BENCHMARK_SPLIT: {split!r}")

    config = get_qdrant_config()
    app_settings = get_app_settings()
    lf_path, tsa_path, _ = _dataset_paths(
        app_settings.data_dir, config.dataset_name, topology_variant=config.topology_variant
    )
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}, topology_variant={config.topology_variant!r}, split={split}")

    exclusion_hours = float(os.environ.get("ELES_TEMPORAL_EXCLUSION_HOURS", "0") or 0)
    causal_lag = os.environ.get("ELES_CAUSAL_LAG_HOURS")
    if exclusion_hours < 0:
        raise ValueError("ELES_TEMPORAL_EXCLUSION_HOURS must be non-negative.")
    if exclusion_hours and split == "group-k-fold":
        raise ValueError(
            "ELES_TEMPORAL_EXCLUSION_HOURS applies to leave-one-group-out only; "
            "group-k-fold already excludes the query's whole fold."
        )

    safe_dataset = config.dataset_name.replace("/", "-")
    split_suffix = "_group_kfold" if split == "group-k-fold" else ""
    excl_suffix = f"_excl{exclusion_hours:g}h" if exclusion_hours else ""
    if causal_lag is not None:
        excl_suffix = f"_causal{float(causal_lag):g}h"
    out_records = (
        TMP_DIR / f"eles_generator_diagnostics_selected_{safe_dataset}_"
        f"{config.topology_variant}{split_suffix}{excl_suffix}.parquet"
    )

    lf = pd.read_pickle(lf_path)
    tsa_store = SqliteRecordStore(tsa_path, table="tsa")
    tsa = tsa_store.fetch(list(lf.index.astype(str)))
    tsa_by_state = {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}
    fold_exclusions: dict[int, list[str]] = {}
    state_to_fold: dict[str, int] = {}
    if split == "group-k-fold":
        folds = group_k_fold_test_groups(tsa["state"], n_splits=5)
        fold_exclusions = {fold: sorted(excluded) for fold, excluded in enumerate(folds)}
        state_to_fold = {uid: fold for fold, excluded in enumerate(folds) for uid in excluded}

    if causal_lag is not None and exclusion_hours:
        raise ValueError("ELES_CAUSAL_LAG_HOURS and ELES_TEMPORAL_EXCLUSION_HOURS are alternatives, not a combination.")

    temporal_exclusions: dict[str, list[str]] = {}
    if causal_lag is not None:
        # One-sided: withhold every state whose outcomes a deployment could not yet hold at
        # query time, and nothing else. A state recorded earlier is legitimate reference data
        # once its contingencies have been simulated, so the symmetric sweep - which also
        # removes those - measures robustness to a hole in the archive rather than deployment.
        # The lag is the delay between a state being recorded and its outcomes being usable.
        lag_hours = float(causal_lag)
        if lag_hours < 0:
            raise ValueError("ELES_CAUSAL_LAG_HOURS must be non-negative.")
        timestamps = load_eles_state_timestamps(RAW_ZIP)
        known = timestamps.reindex([str(i) for i in lf.index])
        if known.isna().any():
            raise ValueError(f"{int(known.isna().sum())} of {len(known)} states have no timestamp in {RAW_ZIP}")
        order = known.sort_values()
        times = order.to_numpy()
        uids = order.index.to_numpy()
        lag = np.timedelta64(int(lag_hours * 3600), "s")
        # Usable for a query at time q: recorded at or before q - lag. Everything later is
        # withheld, the query itself included.
        # Usable for a query at time q: recorded at or before q - lag, and never the query
        # itself. side="right" keeps a state exactly `lag` old usable, but at lag 0 it would
        # also keep the query, which would let it retrieve itself.
        cutoff = np.searchsorted(times, times - lag, side="right")
        temporal_exclusions = {}
        for i, uid in enumerate(uids):
            withheld = list(uids[cutoff[i] :])
            if uid not in withheld:
                withheld.append(uid)
            temporal_exclusions[uid] = withheld
        available = np.array([cutoff[i] - (1 if i < cutoff[i] else 0) for i in range(len(uids))])
        logger.info(
            f"Causal evaluation, ingestion lag {lag_hours:g} h: a query may use only states recorded at or before "
            f"its own timestamp minus the lag. Median states available {np.median(available):.0f} "
            f"(min {available.min()}, max {available.max()}) of {len(uids)}"
        )
    elif exclusion_hours:
        timestamps = load_eles_state_timestamps(RAW_ZIP)
        known = timestamps.reindex([str(i) for i in lf.index])
        if known.isna().any():
            missing = int(known.isna().sum())
            raise ValueError(f"{missing} of {len(known)} states have no timestamp in {RAW_ZIP}")
        order = known.sort_values()
        times = order.to_numpy()
        uids = order.index.to_numpy()
        window = np.timedelta64(int(exclusion_hours * 3600), "s")
        # Both bounds are inclusive: a neighbour exactly H hours away is excluded, so H=0 would
        # still exclude the query itself and any exact-timestamp duplicate.
        lo = np.searchsorted(times, times - window, side="left")
        hi = np.searchsorted(times, times + window, side="right")
        temporal_exclusions = {uid: list(uids[lo[i] : hi[i]]) for i, uid in enumerate(uids)}
        sizes = np.array([hi[i] - lo[i] for i in range(len(uids))])
        logger.info(
            f"Temporal exclusion +/-{exclusion_hours:g} h: excluding a median of "
            f"{np.median(sizes):.0f} states per query (min {sizes.min()}, max {sizes.max()}), "
            f"against 1 under plain leave-one-state-out"
        )

    work: list[WorkItem] = []
    n_taken = 0
    for state_id, state_row in lf.iterrows():
        uid = str(state_id)
        fold = state_to_fold.get(uid)
        if split == "group-k-fold" and fold is None:
            continue
        if fold is None:
            use_temporal = exclusion_hours or causal_lag is not None
            excluded_uids = temporal_exclusions.get(uid, [uid]) if use_temporal else [uid]
        else:
            excluded_uids = fold_exclusions[fold]
        tsa_subset = tsa_by_state.get(uid)
        if tsa_subset is None or tsa_subset.empty:
            continue
        if limit and n_taken >= limit:
            break
        n_taken += 1
        state = {k: (None if pd.isna(v) else v) for k, v in state_row.items()}
        work.append((fold, uid, state, tsa_subset, excluded_uids))

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
