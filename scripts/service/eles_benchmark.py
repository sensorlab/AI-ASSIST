"""Production-path leave-one-group-out benchmark for an eles/* dataset.

Unlike scripts/service/eles_topology_candidate_eval.py (exploratory: hand-rolled its own
EstimationService construction with a hardcoded candidate topology column set, run against
eles/2026-01 before the eles/2026-06 + config-selectable-variant decision was made), this
script goes through the real production entry point, build_estimation_service(), which now
respects DATASET_NAME and TOPOLOGY_VARIANT (see src/services/qdrant/config.py and
datasets/eles/2026-06/README.md's "Topology Variants" section). Supersedes that script for
any further eles benchmark reruns.

Uses an embedded (:memory:) Qdrant instance rather than a live API server, for the same
reason the exploratory script did: no external services needed, and local-mode brute-force
search is fast enough at this dataset's scale (~4,400 states) to run a full leave-one-group-
out pass in-session - true under a real (non-degenerate) topology filter, which restricts
each query to a small same-topology subset. Under a variant whose filter is a no-op (e.g.
slovenia_only on the current data batch - see the README), every query aggregates over the
full ~4,400-state pool instead, which is far slower per query, not just a bigger loop; set
ELES_BENCHMARK_SAMPLE_STATES to a state count (e.g. 300) to subsample which states are used
as queries (retrieval still searches the full population - only the query side is sampled),
keeping runtime tractable for topology-variant comparisons. Off by default so full-population
runs used for paper numbers (e.g. lines_only) stay exactly reproducible.

Run from repository root, e.g.:
    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only QDRANT_URL=":memory:" \\
        uv run python scripts/service/eles_benchmark.py

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=slovenia_only QDRANT_URL=":memory:" \\
        ELES_BENCHMARK_SAMPLE_STATES=300 uv run python scripts/service/eles_benchmark.py
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("QDRANT_URL", ":memory:")

import joblib
import numpy as np
import pandas as pd

from scripts.service.benchmark import normalize_label
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import EstimationService, _dataset_paths, build_estimation_service
from src.services.qdrant.config import get_qdrant_config
from src.services.sqlite_store import SqliteRecordStore

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the paper actually consumes go to paper-sr/data/ (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
# Optional query-side subsampling (env ELES_BENCHMARK_SAMPLE_STATES), for topology-variant
# comparisons where the full leave-one-group-out pass is intractable - e.g. under a variant
# whose topology filter is a no-op, every query is scored against the full reference pool
# instead of a small same-topology subset, which is far more expensive per query (more
# distinct Crit_gen groups to aggregate), not just a bigger loop. Off by default so the
# full-population runs already used for paper numbers (e.g. lines_only) are unaffected and
# stay exactly reproducible. Same SAMPLE_SEED convention as scripts/service/alpha_k_sweep.py.
# Overridable via ELES_BENCHMARK_SAMPLE_SEED for multi-seed robustness checks (e.g. repeating
# the topology with/without-filter ablation across several draws); default 42 matches the
# seed used for the paper's reported ablation numbers.
SAMPLE_SEED: Final[int] = int(os.environ.get("ELES_BENCHMARK_SAMPLE_SEED", "42"))


def _process_state(
    service: EstimationService,
    state_id: Any,
    state: pd.Series,
    tsa_subset: pd.DataFrame,
) -> list[dict[str, Any]]:
    """In-process equivalent of scripts/service/benchmark.py::_process_state (leave-one-
    group-out variant: exclude only the query state itself, matching analyzer())."""
    state_id_norm = normalize_label(state_id)
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}

    reports = service.estimate_by_generator(state=state_dict, exclude_uids=[str(state_id)])
    outputs_by_crit_gen = {normalize_label(key): value for key, value in reports.items()}

    rows: list[dict[str, Any]] = []
    for _, row in tsa_subset.iterrows():
        location_true = normalize_label(row["Location"])
        crit_gen_true = normalize_label(row["Crit_gen"])

        pred = outputs_by_crit_gen.get(crit_gen_true)
        # Report is flat now (no group-level summary/stats blending every location together -
        # see Report's docstring in src/domain/estimation/models.py), so every diagnostic
        # below is sourced from the true location's own LocationReport within per_location.
        per_location = {normalize_label(k): v for k, v in pred.per_location.items()} if pred is not None else {}
        location_true_report = per_location.get(location_true)
        location_stats = location_true_report.summary.stats if location_true_report is not None else None
        distances = location_stats.distances if location_stats is not None else {}

        # No more group-level aggregate CCT to fall back on when the true location isn't
        # covered - fall back to the single most-likely location's CCT instead (the first
        # entry of location_likelihood, which is sorted descending by score).
        top_location_report = None
        if pred is not None and pred.location_likelihood:
            top_location = next(iter(pred.location_likelihood))
            top_location_report = pred.per_location.get(top_location)

        rows.append(
            {
                "state": state_id,
                "state_norm": state_id_norm,
                "cct_true": row["CCT"],
                "crit_gen_true": crit_gen_true,
                "location_true": location_true,
                "cct_weighted_per_location": (
                    location_true_report.summary.cct_weighted if location_true_report is not None else None
                ),
                "cct_weighted_global": (
                    top_location_report.summary.cct_weighted if top_location_report is not None else None
                ),
                "location_weight_mass": (location_stats.weight_mass if location_stats is not None else None),
                "n_neighbors": location_stats.n if location_stats is not None else 0,
                "n_eff": location_stats.n_eff if location_stats is not None else None,
                "neighborhood_compactness": (
                    location_stats.neighborhood_compactness if location_stats is not None else None
                ),
                "cct_weighted_std": (location_stats.cct_weighted_std if location_stats is not None else None),
                "distance_min": distances.get("min"),
                "distance_mean": distances.get("mean"),
                "distance_median": distances.get("median"),
                "distance_spread": distances.get("spread"),
                "distance_norm": distances.get("norm"),
            }
        )
    return rows


def _risk_coverage(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Reproduces reports/30_benchmark_results_analysis.ipynb::risk_coverage() exactly."""
    x = df.dropna(subset=[metric, "err"]).copy()
    x = x.sort_values(metric, ascending=not higher_is_better)

    rows = []
    n = len(x)
    for cov in COVERAGES:
        k = math.ceil(cov * n)
        kept = x.iloc[:k]
        rows.append(
            {
                "metric": metric,
                "coverage": cov,
                "n": k,
                "mae": kept["err"].mean(),
                "rmse": np.sqrt((kept["err"] ** 2).mean()),
                "q90": kept["err"].quantile(0.90),
                "q95": kept["err"].quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    app_settings = get_app_settings()
    lf_path, tsa_path, topo_path = _dataset_paths(
        app_settings.data_dir, config.dataset_name, topology_variant=config.topology_variant
    )
    logger.info(f"Dataset: lf={lf_path}, tsa={tsa_path}, topology_cols={topo_path}")
    logger.info(f"Collection: {config.collection_name} (topology_variant={config.topology_variant!r})")

    safe_dataset = config.dataset_name.replace("/", "-")
    sample_states = os.environ.get("ELES_BENCHMARK_SAMPLE_STATES")
    variant_tag = config.topology_variant
    if sample_states:
        variant_tag = f"{config.topology_variant}-sample{sample_states}"
        if SAMPLE_SEED != 42:
            variant_tag = f"{variant_tag}-seed{SAMPLE_SEED}"
    report_path = TMP_DIR / f"report-{safe_dataset}-{variant_tag}.joblib"
    risk_coverage_path = PAPER_DATA_DIR / f"risk_coverage_{safe_dataset}_{variant_tag}.csv"

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

    logger.info("Building EstimationService via build_estimation_service() (embedded Qdrant)...")
    t0 = time.monotonic()
    service = build_estimation_service()
    logger.info(
        f"Service ready in {time.monotonic() - t0:.1f}s "
        f"({len(service.db.significant_topology_cols)} significant topology columns)"
    )

    all_rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        tsa_subset = tsa_by_state.get(str(state_id))
        if tsa_subset is None or tsa_subset.empty:
            continue
        all_rows.extend(_process_state(service, state_id, state, tsa_subset))
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(lf)} states ({time.monotonic() - t0:.1f}s elapsed)")

    logger.info(f"Done: {len(all_rows)} rows from {len(lf)} states in {time.monotonic() - t0:.1f}s")

    df = pd.DataFrame(all_rows)
    coverage_has_location = df["cct_weighted_per_location"].notna().mean()
    coverage_has_global = df["cct_weighted_global"].notna().mean()
    logger.info(
        f"Coverage: has_location_prediction={coverage_has_location:.4f}, has_any_prediction={coverage_has_global:.4f}"
    )

    joblib.dump(all_rows, report_path)
    logger.info(f"Saved raw per-record report to {report_path}")

    _df = df.dropna(subset=["cct_weighted_per_location"]).copy()
    _df["err"] = (_df["cct_true"] - _df["cct_weighted_per_location"]).abs()

    metrics = {
        "location_weight_mass": True,
        "n_eff": True,
        "n_neighbors": True,
        "neighborhood_compactness": True,
        # Lower cct_weighted_std means the neighbors agree more on the outcome - i.e. more
        # confidence, same direction as the distance metrics below, not the weight/n_eff
        # ones above.
        "cct_weighted_std": False,
        "distance_min": False,
        "distance_mean": False,
        "distance_median": False,
        "distance_spread": False,
        "distance_norm": False,
    }
    out = [_risk_coverage(_df, metric, higher) for metric, higher in metrics.items()]
    rc = pd.concat(out, ignore_index=True)
    rc.to_csv(risk_coverage_path, index=False)
    logger.info(f"Saved risk-coverage CSV to {risk_coverage_path}")
    print(rc.to_string(index=False))


if __name__ == "__main__":
    main()
