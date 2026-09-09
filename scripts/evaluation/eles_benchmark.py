"""Production-path leave-one-group-out benchmark for an eles/* dataset.

Unlike scripts/evaluation/eles_topology_candidate_eval.py (exploratory: hand-rolled its own
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
        uv run python scripts/evaluation/eles_benchmark.py

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=slovenia_only QDRANT_URL=":memory:" \\
        ELES_BENCHMARK_SAMPLE_STATES=300 uv run python scripts/evaluation/eles_benchmark.py
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

os.environ.setdefault("QDRANT_URL", ":memory:")

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scripts.evaluation.benchmark import normalize_label
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import EstimationService, _dataset_paths, build_estimation_service
from src.services.qdrant.config import get_qdrant_config
from src.services.sqlite_store import SqliteRecordStore

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root: raw/intermediate (.joblib) go to tmp/,
# CSV summaries the manuscript reports go to results/data/ (tracked).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
# Datasets whose summaries back a manuscript table or figure. Their CSVs must keep landing in
# results/data/ under the name the paper's chain already cites - the pre-submission release
# check regenerates them and requires the output to be unchanged. Every other dataset writes
# beside its own data instead, so a dataset directory carries its own evaluation history and
# a rerun after a new data drop can be diffed against the previous one.
MANUSCRIPT_DATASETS: Final[frozenset[str]] = frozenset({"eles/2026-01", "eles/2026-06", "bus39"})


def _risk_coverage_dir(dataset_name: str) -> Path:
    if dataset_name in MANUSCRIPT_DATASETS:
        return PAPER_DATA_DIR
    return PROJECT_DIR / "datasets" / dataset_name / "reports"


COVERAGES: Final[tuple[float, ...]] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
# Optional query-side subsampling (env ELES_BENCHMARK_SAMPLE_STATES), for topology-variant
# comparisons where the full leave-one-group-out pass is intractable - e.g. under a variant
# whose topology filter is a no-op, every query is scored against the full reference pool
# instead of a small same-topology subset, which is far more expensive per query (more
# distinct Crit_gen groups to aggregate), not just a bigger loop. Off by default so the
# full-population runs already used for paper numbers (e.g. lines_only) are unaffected and
# stay exactly reproducible. Same SAMPLE_SEED convention as scripts/evaluation/alpha_k_sweep.py.
# Overridable via ELES_BENCHMARK_SAMPLE_SEED for multi-seed robustness checks (e.g. repeating
# the topology with/without-filter ablation across several draws); default 42 matches the
# seed used for the paper's reported ablation numbers.
SAMPLE_SEED: Final[int] = int(os.environ.get("ELES_BENCHMARK_SAMPLE_SEED", "42"))
# Radius, in scaled feature space, within which a reference state counts as a duplicate of the
# query and is withheld alongside it (env ELES_BENCHMARK_DUPLICATE_RADIUS). Off by default so
# every existing run stays reproducible.
#
# Leave-one-state-out removes the query by *identity*, which only removes its information when
# states are distinct operating points. Where an archive repeats a point under another id, the
# answer stays in the reference library and the query effectively looks itself up: on
# interscada/fr, 331 of 1,568 states have a twin closer than 1e-6 and a quarter of all
# state-to-state nearest-neighbour distances are exactly zero. This control removes the copies
# too. It is the distance-based counterpart of the temporal exclusion used on eles, which is
# unavailable wherever states carry no timestamps.
DUPLICATE_RADIUS: Final[float] = float(os.environ.get("ELES_BENCHMARK_DUPLICATE_RADIUS", "0.0"))


def _duplicate_map(service: EstimationService, lf: pd.DataFrame, radius: float) -> dict[str, list[str]]:
    """States within `radius` of each other in the fitted scaler's own feature space.

    Distances are measured where retrieval measures them, by transforming the whole reference
    population with the service's fitted scaler and using the columns the index actually
    searches, so a state counted as a duplicate here is one retrieval would have ranked first.
    """
    scaled = cast(pd.DataFrame, service.scaler.transform(lf))
    embed_cols = [c for c in service.db.embed_cols if c in scaled.columns]
    ids = [str(i) for i in lf.index]
    finder = NearestNeighbors(radius=radius).fit(scaled[embed_cols].to_numpy(dtype=np.float32))
    neighborhoods = finder.radius_neighbors(return_distance=False)
    mapping = {ids[i]: [ids[j] for j in neighbors] for i, neighbors in enumerate(neighborhoods)}
    with_dupes = sum(1 for v in mapping.values() if v)
    total = sum(len(v) for v in mapping.values())
    logger.info(
        f"Duplicate control at radius {radius:g}: {with_dupes} of {len(ids)} states have at least "
        f"one duplicate, {total / len(ids):.2f} per state on average"
    )
    return mapping


def _assert_no_leakage(reports: dict[str, Any], excluded_norm: str) -> None:
    """Fail loudly if the query state contributed to its own estimate.

    scripts/evaluation/benchmark.py runs the equivalent check per scored row, against the
    matched generator group only; this one runs once per query and covers every group and
    every per-location report, which is both stricter and cheaper. estimate_by_generator()
    takes exclude_uids as a required positional argument, so a call site cannot forget it -
    what this guards is the layer below, where a change to how the exclusion filter is built
    could stop applying it without any call site changing.
    """
    for crit_gen, report in reports.items():
        if excluded_norm in {normalize_label(uid) for uid in report.included_state_ids}:
            raise RuntimeError(f"Data leakage: query state `{excluded_norm}` retrieved in group `{crit_gen}`")
        for location, location_report in report.per_location.items():
            if excluded_norm in {normalize_label(uid) for uid in location_report.included_state_ids}:
                raise RuntimeError(
                    f"Data leakage: query state `{excluded_norm}` retrieved at location "
                    f"`{location}` in group `{crit_gen}`"
                )


def _process_state(
    service: EstimationService,
    state_id: Any,
    state: pd.Series,
    tsa_subset: pd.DataFrame,
    duplicates: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """In-process equivalent of scripts/evaluation/benchmark.py::_process_state (leave-one-
    group-out variant: exclude the query state itself, and its duplicates when the duplicate
    control is enabled)."""
    state_id_norm = normalize_label(state_id)
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}

    excluded = [str(state_id), *duplicates]
    reports = service.estimate_by_generator(state=state_dict, exclude_uids=excluded)
    for uid in excluded:
        _assert_no_leakage(reports, normalize_label(uid))
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
        top_location = None
        top_location_score = None
        location_rank = 0
        if pred is not None and pred.location_likelihood:
            ranked_locations = [normalize_label(loc) for loc in pred.location_likelihood]
            top_location = next(iter(pred.location_likelihood))
            top_location_report = pred.per_location.get(top_location)
            top_location_score = pred.location_likelihood[top_location]
            # Rank of the recorded location in this group's likelihood ordering; 0 means the
            # recorded location drew no support in the recorded generator's group at all.
            location_rank = ranked_locations.index(location_true) + 1 if location_true in ranked_locations else 0

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
                # location_likelihood is weight_mass normalized within one generator group, so
                # a score is comparable with the other locations of the same group and with
                # nothing else - hence the rank and the winning location are stored here
                # rather than left to be reconstructed from scores across rows.
                "top_location_pred": normalize_label(top_location) if top_location is not None else None,
                "top_location_score": top_location_score,
                "location_rank": location_rank,
                "n_locations_ranked": len(pred.location_likelihood) if pred is not None else 0,
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


# Reported quantiles of the absolute-error and regret distributions. Deliberately not folded
# into _risk_coverage()'s own columns: that table reproduces reports/30's risk_coverage() and
# its CSV is what the manuscript's figure is built from, so its schema stays fixed.
# The 0th quantile is omitted: the best case is exactly zero on every dataset and every subset
# measured so far, since some record always lands on its neighbours' weighted mean, so the
# column carried no information and only widened an already twelve-column table.
ERROR_QUANTILES: Final[tuple[float, ...]] = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
SCREENING_THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)
# Top-1 and top-3 only. Larger k stops being informative here: a state holds 63 locations but
# the recorded generator's candidate set averages under six, so a top-5 rate mostly measures
# that a candidate list exists rather than that the ranking is any good.
TOPK_VALUES: Final[tuple[int, ...]] = (1, 3)


def _event_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (state, fault location), which is what a location ranking ranks.

    Most datasets already satisfy this: a record is one contingency at one location. interscada/fr
    does not, because a fault there takes a group of machines out of synchronism and its transform
    explodes that group into one row per machine, so a location recurs 2.57 times per state on
    average with the same recorded CCT (a property of the fault, not of the machine) and a
    different estimate per machine's own generator group. Ranking those rows directly would let a
    single location occupy several of the top-k slots and would make the lookup below ambiguous.

    Collapsing averages the per-machine estimates, which are estimates of the same physical
    clearing time, and takes the recorded CCT as-is since it is identical across the group.
    """
    keys = ["state", "location_true"]
    if not df.duplicated(subset=keys).any():
        return df
    return (
        df.groupby(keys, as_index=False)
        .agg(cct_true=("cct_true", "first"), cct_weighted_per_location=("cct_weighted_per_location", "mean"))
        .dropna(subset=["cct_weighted_per_location"])
    )


def _quantile_row(errors: pd.Series) -> dict[str, Any]:
    """Absolute-error summary: count, the two means, and the quantile ladder."""
    row: dict[str, Any] = {
        "n": int(len(errors)),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
    }
    for q in ERROR_QUANTILES:
        row[f"q{int(round(q * 100)):02d}"] = float(errors.quantile(q))
    return row


def _error_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Error quantiles for each estimate the benchmark produces, and for its support split.

    A mean absolute error hides the shape of the distribution, which is what decides whether
    an estimator is usable: on this task the median sits well below the mean and the maximum
    is two orders above it, so the same MAE could describe either a tight estimator with a few
    catastrophic misses or a uniformly mediocre one. The rows separate the estimate at the
    recorded fault location from the fallback at the most likely location, and split the first
    by whether the retrieved support was thin, since that is the split an operator could act on.
    """
    rows: list[dict[str, Any]] = []
    for label, column in (
        ("recorded location", "cct_weighted_per_location"),
        ("most likely location", "cct_weighted_global"),
    ):
        scored = df.dropna(subset=[column])
        if scored.empty:
            continue
        rows.append({"estimate": label, "subset": "all"} | _quantile_row((scored["cct_true"] - scored[column]).abs()))

    scored = df.dropna(subset=["cct_weighted_per_location"])
    if not scored.empty and scored["n_eff"].notna().any():
        errors = (scored["cct_true"] - scored["cct_weighted_per_location"]).abs()
        median_n_eff = scored["n_eff"].median()
        thin = scored["n_eff"] < median_n_eff
        rows.append(
            {"estimate": "recorded location", "subset": f"n_eff below median ({median_n_eff:.1f})"}
            | _quantile_row(errors[thin])
        )
        rows.append(
            {"estimate": "recorded location", "subset": f"n_eff at or above median ({median_n_eff:.1f})"}
            | _quantile_row(errors[~thin])
        )
    return pd.DataFrame(rows)


def _topk_location_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-k contingency screening: rank a state's locations by estimated severity.

    The classification analogue that fits this task is not "which class is the label" but
    "does simulating the k contingencies retrieval flags first reach the worst one". Locations
    are ordered by estimated CCT ascending, most critical first, and two things are reported.

    hit_rate asks whether a location carrying the state's true minimum CCT is inside the top k.
    It credits any location tied at that minimum, because CCT labels sit on a 10 ms grid and
    choosing either of two equally severe contingencies is not an error. regret asks the same
    question in seconds: how much clearing-time margin is left on the table by simulating only
    those k, which is zero whenever the hit lands and small when it does not.

    The second frame is a different and harder question: not "is the worst one found" but "what
    share of every unsafe location is found", for a few clearing-time thresholds. Top-k ranking
    is built for the first question; the second degrades quickly because a state can hold far
    more unsafe locations than any small k can hold.
    """
    scored = _event_level(df.dropna(subset=["cct_weighted_per_location"]))
    ranking_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for state, group in scored.groupby("state"):
        ordered = group.sort_values("cct_weighted_per_location")
        true_min = float(group["cct_true"].min())
        for k in TOPK_VALUES:
            head = ordered.head(k)
            ranking_rows.append(
                {
                    "state": state,
                    "k": k,
                    "hit": bool((head["cct_true"] == true_min).any()),
                    "regret_s": float(head["cct_true"].min() - true_min),
                }
            )
        for threshold in SCREENING_THRESHOLDS:
            unsafe = int((group["cct_true"] < threshold).sum())
            if not unsafe:
                continue
            for k in TOPK_VALUES:
                recall_rows.append(
                    {
                        "state": state,
                        "threshold_s": threshold,
                        "k": k,
                        "recall": float((ordered.head(k)["cct_true"] < threshold).sum() / unsafe),
                    }
                )

    ranking = (
        pd.DataFrame(ranking_rows)
        .groupby("k")
        .apply(
            lambda group: pd.Series(
                {
                    "n_states": group["state"].nunique(),
                    "hit_rate": group["hit"].mean(),
                    "regret_mean_s": group["regret_s"].mean(),
                }
                | {f"regret_q{int(round(q * 100)):02d}_s": group["regret_s"].quantile(q) for q in ERROR_QUANTILES}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    recall = (
        pd.DataFrame(recall_rows)
        .groupby(["threshold_s", "k"])
        .agg(n_states=("state", "nunique"), mean_recall=("recall", "mean"))
        .reset_index()
    )
    return ranking, recall


def _critical_location_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per state, does retrieval point at the most critical fault location?

    The most critical location is the one with the lowest recorded CCT, which is the location
    a screening pass should simulate first. This is a different question from the one
    location_likelihood answers: that ranks locations by retrieved weight mass, i.e. which
    location a query most resembles, not which one is most severe. cct_weighted_global is
    built from the most *likely* location for exactly that reason, so it cannot stand in here.

    Exact name matching understates performance because CCT labels sit on a 10 ms grid and
    several locations frequently tie at a state's minimum; picking either of two equally
    severe locations is not an error. attains_minimum reports the share of states whose
    chosen location actually carries the minimum recorded CCT, and regret measures how much
    clearing time is given up when it does not.
    """
    scored = _event_level(df.dropna(subset=["cct_weighted_per_location"]))
    recorded = scored.loc[scored.groupby("state")["cct_true"].idxmin(), ["state", "location_true", "cct_true"]]
    estimated = scored.loc[
        scored.groupby("state")["cct_weighted_per_location"].idxmin(),
        ["state", "location_true", "cct_weighted_per_location"],
    ]
    paired = recorded.set_index("state").join(estimated.set_index("state"), rsuffix="_pred")
    true_cct = scored.set_index(["state", "location_true"])["cct_true"]
    paired["cct_true_at_predicted"] = [
        true_cct.loc[(state, location)]
        for state, location in zip(paired.index, paired["location_true_pred"], strict=True)
    ]
    regret = paired["cct_true_at_predicted"] - paired["cct_true"]
    tie_counts = scored.groupby("state").apply(
        lambda group: int((group["cct_true"] == group["cct_true"].min()).sum()), include_groups=False
    )
    error = (paired["cct_true"] - paired["cct_weighted_per_location"]).abs()
    return pd.DataFrame(
        [
            {
                "n_states": int(len(paired)),
                "exact_location_match": float((paired["location_true"] == paired["location_true_pred"]).mean()),
                "attains_minimum": float((regret <= 1e-9).mean()),
                "regret_within_one_label_step": float((regret <= 0.0100001).mean()),
                "regret_mean_s": float(regret.mean()),
                "regret_median_s": float(regret.median()),
                "regret_q95_s": float(regret.quantile(0.95)),
                "regret_max_s": float(regret.max()),
                "mae_on_minimum_cct_s": float(error.mean()),
                "states_with_tied_minimum": int((tie_counts > 1).sum()),
            }
        ]
    )


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
        variant_tag = f"{variant_tag}-sample{sample_states}"
        if SAMPLE_SEED != 42:
            variant_tag = f"{variant_tag}-seed{SAMPLE_SEED}"
    if DUPLICATE_RADIUS > 0:
        variant_tag = f"{variant_tag}-dup{DUPLICATE_RADIUS:g}"
    report_path = TMP_DIR / f"report-{safe_dataset}-{variant_tag}.joblib"
    risk_coverage_dir = _risk_coverage_dir(config.dataset_name)
    risk_coverage_dir.mkdir(parents=True, exist_ok=True)
    risk_coverage_path = risk_coverage_dir / f"risk_coverage_{safe_dataset}_{variant_tag}.csv"

    lf = pd.read_pickle(lf_path)
    # Retrieval searches every state, so duplicates are looked for over the full population even
    # when only some states serve as queries; subsampling the query side must not hide a
    # duplicate still sitting in the reference library.
    reference_lf = lf
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
    duplicate_map: dict[str, list[str]] = (
        _duplicate_map(service, reference_lf, DUPLICATE_RADIUS) if DUPLICATE_RADIUS > 0 else {}
    )
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
        all_rows.extend(_process_state(service, state_id, state, tsa_subset, duplicate_map.get(str(state_id), ())))
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

    critical_location_path = risk_coverage_dir / f"critical_location_{safe_dataset}_{variant_tag}.csv"
    critical_location = _critical_location_stats(df)
    critical_location.to_csv(critical_location_path, index=False)
    logger.info(f"Saved most-critical-location summary to {critical_location_path}")

    ranking, recall = _topk_location_stats(df)
    ranking_path = risk_coverage_dir / f"topk_location_{safe_dataset}_{variant_tag}.csv"
    recall_path = risk_coverage_dir / f"topk_screening_recall_{safe_dataset}_{variant_tag}.csv"
    ranking.to_csv(ranking_path, index=False)
    recall.to_csv(recall_path, index=False)
    logger.info(f"Saved top-k location ranking to {ranking_path} and screening recall to {recall_path}")

    quantiles_path = risk_coverage_dir / f"error_quantiles_{safe_dataset}_{variant_tag}.csv"
    _error_quantiles(df).to_csv(quantiles_path, index=False)
    logger.info(f"Saved error quantiles to {quantiles_path}")
    print(rc.to_string(index=False))


if __name__ == "__main__":
    main()
