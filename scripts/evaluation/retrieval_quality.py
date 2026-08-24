"""Evaluate BUS39 retrieval *as retrieval*, rather than only through its aggregated CCT estimate.

The paper's founding assumption is that proximity in the engineered feature space implies similar
transient-stability outcomes. That assumption is never tested: every reported number scores the
aggregated estimate, so a good result could come from a good metric or from CCT being easy to
average regardless of who the neighbours are. Three measurements separate those.

1. Founding assumption, unrestricted. For a sampled query state and a fixed contingency slice
   (same fault location and same critical generator), correlate the query-to-state feature
   distance with the absolute CCT difference, over the full same-slice population regardless of
   topology match. A metric that carries stability information gives a positive correlation; a
   metric that carries none gives zero. Reported as Spearman rho over pairs, plus the mean |dCCT|
   for the nearest 100 states against a random 100 from the same slice, which is the effect size a
   reader can act on. NOTE: this measurement is deliberately *not* restricted to the query's
   topology group - on a dataset with small topology groups (ELES: 2-38 states) a "nearest/random
   100" comparison is not achievable within a single group at all, so this number characterizes
   the feature representation in the abstract, not the population the deployed, topology-filtered
   retrieval actually searches. See (2) for that.

2. Founding assumption, topology-restricted (added 2026-08-08, in response to peer review: (1)
   alone cannot be read as validating the deployed retrieval's assumption, since the deployed
   system never sees a same-slice pool this large - it only ever searches within the query's exact
   topology group). Repeats (1) with the same-slice population additionally restricted to states
   sharing the query's topology id, using k = fidelity_k (typically far below 100, since that's
   the largest k achievable within realistic topology-group sizes) instead of 100. Query states
   whose same-topology same-slice pool is smaller than k are skipped and counted separately, same
   convention as (3). Uses an independent RNG stream (SEED + 1) so it cannot perturb the (1)
   numbers already reported from earlier runs.

3. Index fidelity (implemented 2026-08-05, was a docstring promise with no code before that).
   Builds a real EstimationService/DatabaseQdrant and issues actual queries through it, subject
   to the same topology filter production uses. Compares the returned neighbour set against the
   exact brute-force top-K on the same scaled matrix, restricted to the same same-topology
   candidate pool, and reports mean recall@K. Query states whose same-topology pool is smaller
   than K are skipped (recall@K is not meaningful there) and counted separately. NOTE: in the
   `:memory:` configuration Qdrant may fall back to exact scan for a collection this small, in
   which case recall@K will read as ~1.0 by construction and says nothing about a production HNSW
   index's behaviour - this is a known limitation of testing against the embedded backend, not
   fixed here.

The measurement is per dataset, because the metric is per dataset: the BUS39 and ELES scaled
representations differ in dimensionality and in which feature block carries the variance, so a
result on one says nothing about the other.

Run from the repository root:
    uv run python scripts/evaluation/retrieval_quality.py [dataset] [n_query_states] [min_slice] [fidelity_k]
e.g. bus39 300 200   or   eles/2026-06 300 50 5
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("DATASET_NAME", "bus39")
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("DATA_DIR", "./datasets")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.logging import configure_logging  # noqa: E402
from src.domain.estimation.service import _make_scaler_for_dataset, build_estimation_service  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# CSV summaries the manuscript reports go to results/data/, not the repo root
# (2026-08-05 cleanup).
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
K: Final[int] = 100
SEED: Final[int] = 42


def _topology_ids(lf: pd.DataFrame, dataset: str) -> tuple[Any, dict[str, str] | None]:
    """Builds a real EstimationService (needed for its fitted significant_topology_cols) and,
    if the dataset has any significant topology columns, the per-state topology id string used
    to restrict both the index-fidelity check and the topology-restricted founding-assumption
    check to the same-topology candidate pool the deployed retrieval actually searches."""
    os.environ["DATASET_NAME"] = dataset
    service = build_estimation_service()
    topo_cols = service.db.significant_topology_cols
    if not topo_cols:
        return service, None

    topo_bits = lf[topo_cols].astype(bool).to_numpy()
    topo_id_by_state = {
        str(idx): "".join(np.where(row, "1", "0")) for idx, row in zip(lf.index, topo_bits, strict=True)
    }
    return service, topo_id_by_state


def _index_fidelity(
    lf: pd.DataFrame,
    Z: np.ndarray,
    pos: dict[str, int],
    states: list[str],
    service: Any,
    topo_id_by_state: dict[str, str] | None,
    *,
    rng: np.random.Generator,
    k: int,
    n_query: int,
) -> dict[str, Any]:
    """Measures recall@K of the vector database's own retrieval path against the exact
    brute-force top-K on the same scaled matrix, both restricted to the query's same-
    topology candidate pool (the production filter - see repository.py::query()).
    Uses the real EstimationService/DatabaseQdrant rather than assuming the founding-
    assumption check above says anything about the index itself."""
    if topo_id_by_state is None:
        return {"index_fidelity_skipped_reason": "no significant topology columns for this dataset"}

    states_by_topo: dict[str, list[str]] = defaultdict(list)
    for s in states:
        states_by_topo[topo_id_by_state[s]].append(s)

    query_states = list(rng.choice(states, size=min(n_query, len(states)), replace=False))
    recalls: list[float] = []
    n_skipped_small_pool = 0

    for qs in query_states:
        qi = pos[qs]
        pool = [s for s in states_by_topo[topo_id_by_state[qs]] if s != qs]
        if len(pool) < k:
            n_skipped_small_pool += 1
            continue

        idx = np.array([pos[s] for s in pool])
        d = np.linalg.norm(Z[idx] - Z[qi], axis=1)
        true_topk = {pool[i] for i in np.argsort(d)[:k]}

        # Matches EstimationService._query_neighbors() exactly: db.query() expects the
        # already-scaled representation (service.scaler.transform), not the raw LF row.
        sample = service.scaler.transform(lf.iloc[[qi]].astype(np.float64))
        result = service.db.query(state=sample, limit=k, exclude_source_index=qs)
        retrieved = set(result.rows.index.astype(str))
        recalls.append(len(retrieved & true_topk) / k)

    return {
        "index_fidelity_n_query_used": len(recalls),
        "index_fidelity_n_skipped_small_pool": n_skipped_small_pool,
        "index_fidelity_mean_recall_at_k": round(float(np.mean(recalls)), 4) if recalls else float("nan"),
        "index_fidelity_min_recall_at_k": round(float(np.min(recalls)), 4) if recalls else float("nan"),
    }


def _founding_assumption_topology_restricted(
    q_states: list[str],
    tsa: pd.DataFrame,
    Z: np.ndarray,
    pos: dict[str, int],
    slice_map: dict[tuple[str, str], dict[str, float]],
    topo_id_by_state: dict[str, str] | None,
    *,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """Repeats the founding-assumption check (Spearman rho of distance vs. |dCCT|, and the
    nearest-k-vs-random-k mean-|dCCT| effect size) restricted to the query's exact topology
    group, i.e. the same candidate pool the deployed, topology-filtered retrieval actually
    searches - unlike the unrestricted version above. Uses the *same* q_states sample as the
    unrestricted check for direct comparability, but its own RNG stream (seed) so it cannot
    perturb the unrestricted numbers computed earlier in main()."""
    if topo_id_by_state is None:
        return {"topo_restricted_skipped_reason": "no significant topology columns for this dataset"}

    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    near_gaps: list[float] = []
    rand_gaps: list[float] = []
    pool_sizes: list[int] = []
    n_skipped_small_pool = 0

    for qs in q_states:
        qi = pos[qs]
        rows = tsa[tsa["_state"] == qs]
        if rows.empty:
            continue
        r = rows.iloc[0]
        key = (r["_loc"], r["_gen"])
        members = slice_map.get(key, {})
        topo = topo_id_by_state[qs]
        pool = [(s, c) for s, c in members.items() if s != qs and topo_id_by_state[s] == topo]
        if len(pool) < k:
            n_skipped_small_pool += 1
            continue

        pool_sizes.append(len(pool))
        idx = np.array([pos[s] for s, _ in pool])
        ccts = np.array([c for _, c in pool], dtype=float)
        d = np.linalg.norm(Z[idx] - Z[qi], axis=1)
        gap = np.abs(ccts - float(r["CCT"]))

        dr = pd.Series(d).rank().to_numpy()
        gr = pd.Series(gap).rank().to_numpy()
        if dr.std() > 0 and gr.std() > 0:
            rhos.append(float(np.corrcoef(dr, gr)[0, 1]))

        order = np.argsort(d)
        near_gaps.append(float(gap[order[:k]].mean()))
        rand_gaps.append(float(gap[rng.choice(len(gap), size=k, replace=False)].mean()))

    if not near_gaps:
        return {
            "topo_restricted_n_query_used": 0,
            "topo_restricted_n_skipped_small_pool": n_skipped_small_pool,
        }

    near = float(np.mean(near_gaps))
    rand = float(np.mean(rand_gaps))
    rho = float(np.mean(rhos)) if rhos else float("nan")
    return {
        "topo_restricted_k": k,
        "topo_restricted_n_query_used": len(near_gaps),
        "topo_restricted_n_skipped_small_pool": n_skipped_small_pool,
        "topo_restricted_median_pool_size": float(np.median(pool_sizes)),
        "topo_restricted_spearman_distance_vs_abs_dcct_mean": round(rho, 4),
        "topo_restricted_spearman_frac_positive": round(float(np.mean([x > 0 for x in rhos])), 4)
        if rhos
        else float("nan"),
        f"topo_restricted_mean_abs_dcct_nearest_{k}_s": round(near, 5),
        f"topo_restricted_mean_abs_dcct_random_{k}_s": round(rand, 5),
        "topo_restricted_reduction_vs_random_pct": round(100.0 * (rand - near) / rand, 2) if rand else float("nan"),
    }


def main() -> None:
    configure_logging()
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bus39"
    n_query = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    min_slice = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    # Separate from K (the founding-assumption nearest-neighbor count, kept at 100): the
    # index-fidelity check needs a K achievable within a same-topology pool, which on a
    # finely-fragmented dataset (e.g. ELES lines_only, max group size ~38) can be far
    # smaller than 100 - defaults to K unchanged for datasets where that's not an issue.
    fidelity_k = int(sys.argv[4]) if len(sys.argv) > 4 else K
    base = PROJECT_DIR / "datasets" / dataset / "interim"
    out_csv = PAPER_DATA_DIR / f"retrieval_quality-{dataset.replace('/', '-')}.csv"

    lf: pd.DataFrame = pd.read_pickle(base / "lf.pkl")
    tsa: pd.DataFrame = pd.read_pickle(base / "tsa.pkl")

    service, topo_id_by_state = _topology_ids(lf, dataset)

    scaler = _make_scaler_for_dataset(dataset)
    Z = np.asarray(scaler.fit_transform(lf), dtype=np.float64)
    states = [str(i) for i in lf.index]
    pos = {s: i for i, s in enumerate(states)}
    logger.info(f"scaled matrix {Z.shape}")

    # Slice index: (location, critical generator) -> {state: CCT}. Duplicate (state, slice) pairs
    # are averaged so one state contributes one CCT per slice.
    tsa = tsa.copy()
    tsa["_state"] = tsa["state"].astype(str)
    tsa["_loc"] = tsa["Location"].astype(str).str.strip().str.lower()
    tsa["_gen"] = tsa["Crit_gen"].astype(str).str.strip().str.lower()
    grouped = tsa.groupby(["_loc", "_gen", "_state"], observed=True)["CCT"].mean()

    slice_map: dict[tuple[str, str], dict[str, float]] = {}
    for (loc, gen, st), cct in grouped.items():
        slice_map.setdefault((loc, gen), {})[st] = float(cct)
    logger.info(f"{len(slice_map)} distinct (location, generator) slices")

    rng = np.random.default_rng(SEED)
    q_states = list(rng.choice(states, size=min(n_query, len(states)), replace=False))

    rhos: list[float] = []
    near_gaps: list[float] = []
    rand_gaps: list[float] = []
    n_pairs_total = 0

    for qs in q_states:
        qi = pos[qs]
        rows = tsa[tsa["_state"] == qs]
        if rows.empty:
            continue
        # one representative slice per query, the first by (loc, gen)
        r = rows.iloc[0]
        key = (r["_loc"], r["_gen"])
        members = slice_map.get(key, {})
        others = [(s, c) for s, c in members.items() if s != qs]
        if len(others) < min_slice:
            continue

        idx = np.array([pos[s] for s, _ in others])
        ccts = np.array([c for _, c in others], dtype=float)
        d = np.linalg.norm(Z[idx] - Z[qi], axis=1)
        gap = np.abs(ccts - float(r["CCT"]))

        # Spearman via rank Pearson, stdlib-only
        dr = pd.Series(d).rank().to_numpy()
        gr = pd.Series(gap).rank().to_numpy()
        if dr.std() > 0 and gr.std() > 0:
            rhos.append(float(np.corrcoef(dr, gr)[0, 1]))
        n_pairs_total += len(others)

        order = np.argsort(d)
        near_gaps.append(float(gap[order[:K]].mean()))
        rand_gaps.append(float(gap[rng.choice(len(gap), size=K, replace=False)].mean()))

    rho = float(np.mean(rhos)) if rhos else float("nan")
    near = float(np.mean(near_gaps))
    rand = float(np.mean(rand_gaps))

    logger.info(f"Running topology-restricted founding-assumption check at k={fidelity_k}...")
    # Independent RNG stream (SEED + 1): must not perturb the unrestricted rho/near/rand
    # numbers above, which were computed from the shared `rng` and are already reported.
    topo_restricted = _founding_assumption_topology_restricted(
        q_states, tsa, Z, pos, slice_map, topo_id_by_state, k=fidelity_k, seed=SEED + 1
    )

    logger.info(f"Running index-fidelity check at K={fidelity_k} (builds a real EstimationService)...")
    fidelity = _index_fidelity(lf, Z, pos, states, service, topo_id_by_state, rng=rng, k=fidelity_k, n_query=n_query)
    fidelity["index_fidelity_k"] = fidelity_k

    rows = [
        {"quantity": "dataset", "value": dataset},
        {"quantity": "scaled_dimensionality", "value": int(Z.shape[1])},
        {"quantity": "min_slice_members", "value": min_slice},
        {"quantity": "n_query_states_used", "value": len(rhos)},
        {"quantity": "n_pairs_total", "value": n_pairs_total},
        {"quantity": "spearman_distance_vs_abs_dcct_mean", "value": round(rho, 4)},
        {"quantity": "spearman_frac_positive", "value": round(float(np.mean([r > 0 for r in rhos])), 4)},
        {"quantity": f"mean_abs_dcct_nearest_{K}_s", "value": round(near, 5)},
        {"quantity": f"mean_abs_dcct_random_{K}_s", "value": round(rand, 5)},
        {"quantity": "reduction_vs_random_pct", "value": round(100.0 * (rand - near) / rand, 2)},
    ]
    rows.extend({"quantity": key, "value": value} for key, value in topo_restricted.items())
    rows.extend({"quantity": key, "value": value} for key, value in fidelity.items())

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
