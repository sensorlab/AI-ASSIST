"""Evaluate BUS39 retrieval *as retrieval*, rather than only through its aggregated CCT estimate.

The paper's founding assumption is that proximity in the engineered feature space implies similar
transient-stability outcomes. That assumption is never tested: every reported number scores the
aggregated estimate, so a good result could come from a good metric or from CCT being easy to
average regardless of who the neighbours are. Two measurements separate those.

1. Founding assumption. For a sampled query state and a fixed contingency slice (same fault
   location and same critical generator), correlate the query-to-state feature distance with the
   absolute CCT difference. A metric that carries stability information gives a positive
   correlation; a metric that carries none gives zero. Reported as Spearman rho over pairs, plus
   the mean |dCCT| for the nearest 100 states against a random 100 from the same slice, which is
   the effect size a reader can act on.

2. Index fidelity. The prototype retrieves through the vector database's approximate index. Recall
   of the exact K nearest neighbours is measured by brute force over the same scaled matrix.
   NOTE: in the `:memory:` configuration Qdrant may fall back to exact scan for a collection this
   small, in which case the measured recall reflects that configuration and not a production HNSW
   index. The script reports which backend it used so the number is not over-read.

The measurement is per dataset, because the metric is per dataset: the BUS39 and ELES scaled
representations differ in dimensionality and in which feature block carries the variance, so a
result on one says nothing about the other.

Run from the repository root:
    uv run python scripts/service/retrieval_quality.py [dataset] [n_query_states] [min_slice]
e.g. bus39 300 200   or   eles/2026-06 300 50
"""

from __future__ import annotations

import logging
import os
import sys
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
from src.domain.estimation.service import _make_scaler_for_dataset  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
K: Final[int] = 100
SEED: Final[int] = 42


def main() -> None:
    configure_logging()
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bus39"
    n_query = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    min_slice = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    base = PROJECT_DIR / "datasets" / dataset / "interim"
    out_csv = PROJECT_DIR / f"retrieval_quality-{dataset.replace('/', '-')}.csv"

    lf: pd.DataFrame = pd.read_pickle(base / "lf.pkl")
    tsa: pd.DataFrame = pd.read_pickle(base / "tsa.pkl")

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

    out = pd.DataFrame(
        [
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
    )
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
