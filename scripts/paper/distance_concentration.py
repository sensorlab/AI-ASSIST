"""Distance concentration in the retrieval feature space (R2.10a, Q1).

Pre-specified in the manuscript repository's revision/analyses/03-distance-concentration.md, committed before any
number here was computed. Descriptive, no decision rule.

Feature space: the production embedding, i.e. the scaled columns that enter the Euclidean distance (switching flags
excluded, since they filter rather than weigh), as float64 of the float32 vectors the index stores.

- Relative contrast RC = (d_max - d_min) / d_min of a query against a candidate set, query excluded. Undefined for
  fewer than two candidates and infinite when d_min = 0; both are counted, not imputed.
- RC_pool (ELES): candidates share the query's exact line-status key, the pool retrieval actually weighs.
- RC_all (ELES, BUS39): every other state.
- Null: RC_all after permuting every column independently across states (seed 42), which keeps marginals and removes
  dependence between features.
- PCA: components needed for 90, 95 and 99% of variance.
- Kernel flatness (ELES, alpha = 1): n_eff / n of the selected-generator group, groups with two or more neighbours,
  from the validated harness records.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/distance_concentration.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _retrieval_harness import LINE_KEY, load_corpus  # noqa: E402
from referral_decomposition import ELES_LOSO, VALIDATION  # noqa: E402

from src.domain.estimation.service import _make_scaler_for_dataset  # noqa: E402
from src.services.qdrant.repository import RE_TOPO  # noqa: E402

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
BUS39_LF: Final[Path] = PROJECT_DIR / "datasets" / "bus39" / "interim" / "lf.pkl"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "distance_concentration.csv"
SEED: Final[int] = 42
ROW_CHUNK: Final[int] = 1024
VARIANCE_LEVELS: Final[tuple[float, ...]] = (0.90, 0.95, 0.99)
QUANTILES: Final[tuple[float, ...]] = (0.10, 0.50, 0.90)


def embed(lf_scaled: pd.DataFrame) -> np.ndarray:
    cols = sorted(c for c in lf_scaled.columns if not RE_TOPO.match(c))
    x = lf_scaled.loc[:, cols].to_numpy(dtype=np.float32).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite features")
    return x


def distance_rows(x: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start, block of Euclidean distances from rows start.. to every row), via the Gram identity."""
    sq = np.einsum("ij,ij->i", x, x)
    for start in range(0, len(x), ROW_CHUNK):
        block = x[start : start + ROW_CHUNK]
        d2 = sq[start : start + ROW_CHUNK, None] + sq[None, :] - 2.0 * block @ x.T
        yield start, np.sqrt(np.maximum(d2, 0.0))


def relative_contrast(d: np.ndarray) -> float:
    """RC of one query's distances to its candidates (query already removed)."""
    if len(d) < 2:
        return np.nan
    lo = d.min()
    return np.inf if lo == 0 else float((d.max() - lo) / lo)


def contrasts(x: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """RC per query against every other state, or against the states sharing its group."""
    out = np.empty(len(x))
    for start, block in distance_rows(x):
        for i, row in enumerate(block):
            q = start + i
            mask = np.ones(len(x), bool) if groups is None else groups == groups[q]
            mask[q] = False
            out[q] = relative_contrast(row[mask])
    return out


def permute_columns(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permuted(x, axis=0)


def pca_components(x: np.ndarray) -> dict[float, int]:
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    share = np.cumsum(s**2) / np.sum(s**2)
    return {level: int(np.searchsorted(share, level) + 1) for level in VARIANCE_LEVELS}


def summarise(dataset: str, measure: str, values: np.ndarray) -> list[dict[str, Any]]:
    finite = values[np.isfinite(values)]
    rows = [
        {"dataset": dataset, "measure": measure, "quantity": "n_queries", "estimate": len(values)},
        {"dataset": dataset, "measure": measure, "quantity": "n_undefined", "estimate": int(np.isnan(values).sum())},
        {"dataset": dataset, "measure": measure, "quantity": "n_infinite", "estimate": int(np.isinf(values).sum())},
    ]
    for q in QUANTILES:
        rows.append(
            {
                "dataset": dataset,
                "measure": measure,
                "quantity": f"q{int(100 * q)}",
                "estimate": float(np.quantile(finite, q)),
            }
        )
    return rows


def main() -> None:
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []

    corpus = load_corpus()
    eles = embed(corpus.lf_scaled)
    key = corpus.lf_scaled.loc[:, corpus.key_columns[LINE_KEY]].to_numpy(dtype=bool)
    groups = np.unique(key, axis=0, return_inverse=True)[1].ravel()

    bus39_lf = cast(pd.DataFrame, pd.read_pickle(BUS39_LF))
    bus39 = embed(cast(pd.DataFrame, _make_scaler_for_dataset("bus39").fit_transform(bus39_lf)))

    measures: list[tuple[str, str, Callable[[], np.ndarray]]] = [
        ("ELES", "rc_pool", lambda: contrasts(eles, groups)),
        ("ELES", "rc_all", lambda: contrasts(eles)),
        ("ELES", "rc_all_null", lambda: contrasts(permute_columns(eles, rng))),
        ("BUS39", "rc_all", lambda: contrasts(bus39)),
        ("BUS39", "rc_all_null", lambda: contrasts(permute_columns(bus39, rng))),
    ]
    for dataset, measure, compute in measures:
        rows += summarise(dataset, measure, compute())
        print(f"{dataset} {measure} done", flush=True)

    for dataset, x in (("ELES", eles), ("BUS39", bus39)):
        rows.append({"dataset": dataset, "measure": "features", "quantity": "n_states", "estimate": x.shape[0]})
        rows.append({"dataset": dataset, "measure": "features", "quantity": "n_dimensions", "estimate": x.shape[1]})
        for level, n in pca_components(x).items():
            rows.append(
                {"dataset": dataset, "measure": "pca", "quantity": f"components_{int(100 * level)}pct", "estimate": n}
            )

    rec = pd.read_parquet(
        ELES_LOSO, columns=["state", "loc_true", "gen_selected", "covered", "sel_n_eff", "sel_n_neighbors"]
    )
    rec = rec[rec["covered"] & (rec["sel_n_neighbors"] >= 2)]
    # Records at the same location share one selected-generator group; count each group once.
    rec = rec.drop_duplicates(["state", "loc_true", "gen_selected"])
    rows += summarise("ELES", "neff_fraction", (rec["sel_n_eff"] / rec["sel_n_neighbors"]).to_numpy())

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
