"""The ELES founding-assumption diagnostic under alternative representations (analysis 05, Q2, R2.11).

Pre-specified in the manuscript repository's revision/analyses/05-representations.md, committed before any number here
was computed. The published diagnostic (scripts/evaluation/retrieval_quality.py, Supplementary Note S3) samples 300
ELES states with seed 42, takes for each the first (location, recorded critical generator) slice of its records, and,
if the slice holds at least 50 other states, correlates feature distance with |delta CCT| over every state in the slice
regardless of topology (Spearman), and compares the mean |delta CCT| of the 100 nearest with 100 random ones. This
script repeats that loop verbatim with the distance swapped:

    full       the published distance: Euclidean on the full scaled mapping, switching columns included (reproduced
               first; the script stops unless it gives the published 0.1412)
    deployed   Euclidean on the retrieval embedding (switching columns excluded)
    pca90, pca99, slovenian, l1   the analysis 05 representations, on the same embedding

The random draws do not depend on the distance, so every representation sees the same queries and random draws.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/founding_assumption_representations.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _representations  # noqa: E402
from _retrieval_harness import EXACT_LINES, LINE_KEY, BruteForceDatabase, load_corpus  # noqa: E402

from src.services.qdrant.repository import RE_TOPO  # noqa: E402

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TSA_PATH: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06" / "interim" / "tsa.pkl"
PUBLISHED: Final[Path] = PROJECT_DIR / "results" / "data" / "retrieval_quality-eles-2026-06.csv"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "founding_assumption_representations.csv"
N_QUERY: Final[int] = 300
MIN_SLICE: Final[int] = 50
K: Final[int] = 100
SEED: Final[int] = 42

Pairwise = Callable[[int, np.ndarray], np.ndarray]


def diagnostic(states: list[str], tsa: pd.DataFrame, distance: Pairwise) -> dict[str, float]:
    """retrieval_quality.py's unrestricted founding-assumption loop with `distance(query_row, member_rows)`."""
    pos = {s: i for i, s in enumerate(states)}
    grouped = tsa.groupby(["_loc", "_gen", "_state"], observed=True)["CCT"].mean()
    slice_map: dict[tuple[str, str], dict[str, float]] = {}
    for (loc, gen, st), cct in grouped.items():
        slice_map.setdefault((loc, gen), {})[st] = float(cct)
    first = tsa.groupby("_state", sort=False).head(1).set_index("_state")

    rng = np.random.default_rng(SEED)
    q_states = list(rng.choice(states, size=min(N_QUERY, len(states)), replace=False))
    rhos, near, rand = [], [], []
    for qs in q_states:
        if qs not in first.index:
            continue
        r = first.loc[qs]
        others = [(s, c) for s, c in slice_map.get((r["_loc"], r["_gen"]), {}).items() if s != qs]
        if len(others) < MIN_SLICE:
            continue
        idx = np.array([pos[s] for s, _ in others])
        gap = np.abs(np.array([c for _, c in others], dtype=float) - float(r["CCT"]))
        d = distance(pos[qs], idx)
        dr, gr = pd.Series(d).rank().to_numpy(), pd.Series(gap).rank().to_numpy()
        if dr.std() > 0 and gr.std() > 0:
            rhos.append(float(np.corrcoef(dr, gr)[0, 1]))
        near.append(float(gap[np.argsort(d)[:K]].mean()))
        rand.append(float(gap[rng.choice(len(gap), size=K, replace=False)].mean()))
    n, ra = float(np.mean(near)), float(np.mean(rand))
    return {
        "n_query_states_used": len(rhos),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_frac_positive": float(np.mean([x > 0 for x in rhos])),
        "mean_abs_dcct_nearest_100_s": n,
        "mean_abs_dcct_random_100_s": ra,
        "reduction_vs_random_pct": 100.0 * (ra - n) / ra,
    }


def main() -> None:
    corpus = load_corpus()
    states = [str(i) for i in corpus.lf.index]
    tsa = pd.read_pickle(TSA_PATH).copy()
    tsa["_state"] = tsa["state"].astype(str)
    tsa["_loc"] = tsa["Location"].astype(str).str.strip().str.lower()
    tsa["_gen"] = tsa["Crit_gen"].astype(str).str.strip().str.lower()

    full = np.asarray(corpus.lf_scaled, dtype=np.float64)
    db = BruteForceDatabase(corpus.lf_scaled, corpus.key_columns, EXACT_LINES)
    if list(corpus.lf_scaled.index.astype(str)) != list(db.src_index):
        raise ValueError("corpus and database rows are not aligned")
    vectors = db.vectors

    def euclid(x: np.ndarray) -> Pairwise:
        return lambda q, idx: np.linalg.norm(x[idx] - x[q], axis=1)

    candidates: dict[str, Pairwise] = {"full": euclid(full), "deployed": euclid(vectors)}
    raw_columns = [c for c in corpus.lf.columns if not RE_TOPO.match(c)]
    facts: dict[str, dict[str, float]] = {}
    for name in _representations.NAMES:
        dist, facts[name] = _representations.build(name, vectors, db.embed_cols, raw_columns, db.bits[LINE_KEY])
        candidates[name] = lambda q, idx, dist=dist: dist(vectors[q : q + 1], vectors[idx])

    rows: list[dict[str, Any]] = []
    for name, distance in candidates.items():
        result = diagnostic(states, tsa, distance)
        if name == "full":
            published = pd.read_csv(PUBLISHED).set_index("quantity")["value"]
            if round(result["spearman_mean"], 4) != float(published["spearman_distance_vs_abs_dcct_mean"]):
                raise ValueError(f"published diagnostic not reproduced: {result['spearman_mean']:.4f}")
        dims = {"full": full.shape[1], "deployed": vectors.shape[1]}.get(name, facts.get(name, {}).get("n_dimensions"))
        rows += [{"representation": name, "quantity": k, "estimate": v} for k, v in result.items()]
        rows.append({"representation": name, "quantity": "n_dimensions", "estimate": dims})
        if name in facts:
            rows.append(
                {
                    "representation": name,
                    "quantity": "bandwidth_constant",
                    "estimate": facts[name]["bandwidth_constant"],
                }
            )
        print(name, {k: round(v, 4) for k, v in result.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
