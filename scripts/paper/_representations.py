"""Alternative retrieval representations for the brute-force harness (analysis 05, R2.10b).

A representation replaces only the distance each retrieved neighbour gets; on ELES under the exact line-status key the
neighbour set is the whole compatible pool whatever the distance (K never binds), so weights and generator ranking
change and coverage does not. Every representation works on the production embedding (scaled, switching columns
excluded), in `embed_cols` order, and is multiplied by one constant so that the median distance between distinct
states sharing an exact line-status key equals the deployed Euclidean one: the kernel exp(-alpha d) depends on the
scale of d, and the constant keeps the effective bandwidth comparable without looking at outcomes.

    pca90, pca99   Euclidean on principal-component scores (components for 90% / 99% of variance, fitted on all states)
    slovenian      Euclidean on the columns of elements listed in the Slovenian PowerFactory dictionary
    l1             Manhattan distance on all columns
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
DICTIONARY: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-01" / "raw" / "powerfactory_dictionary.xlsx"
NAMES: Final[tuple[str, ...]] = ("pca90", "pca99", "slovenian", "l1")
PCA_LEVELS: Final[dict[str, float]] = {"pca90": 0.90, "pca99": 0.99}
ELEMENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9]+_(?P<el>.+?)_\[[^\]]*\](\.\d+)?$")
TYPE_RE: Final[re.Pattern[str]] = re.compile(r"^(Bus|Lne|Lod|Gen)")
ANGLE_SUFFIXES: Final[tuple[str, ...]] = ("_cos", "_sin")

Distance = Callable[[np.ndarray, np.ndarray], np.ndarray]


def dictionary_names(path: Path = DICTIONARY) -> set[str]:
    names: set[str] = set()
    for sheet in pd.ExcelFile(path).sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        for col in frame.columns:
            names.update(str(v).strip() for v in frame[col].dropna())
    return names


def slovenian_raw_columns(raw_columns: Sequence[str], names: set[str]) -> list[str]:
    """Raw feature columns whose element (after the type prefix Bus/Lne/Lod/Gen) is a dictionary entry."""
    out = []
    for col in raw_columns:
        m = ELEMENT_RE.match(col)
        if m is None:
            continue
        element = m["el"]
        base = element[len(t.group(0)) :] if (t := TYPE_RE.match(element)) else element
        if base in names or element in names:
            out.append(col)
    return out


def embed_mask(embed_cols: Sequence[str], raw_columns: Sequence[str]) -> np.ndarray:
    """Embedding columns derived from `raw_columns`: the column itself, or its sine and cosine for an angle."""
    wanted = set(raw_columns)
    return np.array(
        [c in wanted or any(c.endswith(s) and c[: -len(s)] in wanted for s in ANGLE_SUFFIXES) for c in embed_cols]
    )


def same_key_pairs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i < j) of rows with identical key rows."""
    groups = pd.Series(range(len(keys))).groupby(pd.factorize(pd.Series(map(bytes, keys)))[0]).apply(list)
    a, b = [], []
    for members in groups:
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a.append(members[x])
                b.append(members[y])
    return np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)


def bandwidth_constant(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Constant c with median(c * candidate) == median(reference)."""
    return float(np.median(reference) / np.median(candidate))


def build(
    name: str, vectors: np.ndarray, embed_cols: Sequence[str], raw_columns: Sequence[str], keys: np.ndarray
) -> tuple[Distance, dict[str, float]]:
    """Distance function for representation `name`, fitted on the corpus vectors, plus facts to report."""
    if name in PCA_LEVELS:
        mean = vectors.mean(axis=0)
        _, s, vt = np.linalg.svd(vectors - mean, full_matrices=False)
        share = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(share, PCA_LEVELS[name]) + 1)
        basis = vt[:k].T

        def raw(q: np.ndarray, n: np.ndarray) -> np.ndarray:
            return np.linalg.norm((n - mean) @ basis - (q - mean) @ basis, axis=1)

        facts = {"n_dimensions": k}
    elif name == "slovenian":
        mask = embed_mask(embed_cols, slovenian_raw_columns(raw_columns, dictionary_names()))

        def raw(q: np.ndarray, n: np.ndarray) -> np.ndarray:
            return np.linalg.norm(n[:, mask] - q[:, mask], axis=1)

        facts = {"n_dimensions": int(mask.sum())}
    elif name == "l1":

        def raw(q: np.ndarray, n: np.ndarray) -> np.ndarray:
            return np.abs(n - q).sum(axis=1)

        facts = {"n_dimensions": vectors.shape[1]}
    else:
        raise ValueError(f"unknown representation {name!r}")

    a, b = same_key_pairs(keys)
    reference = np.linalg.norm(vectors[a] - vectors[b], axis=1)
    candidate = raw(vectors[a], vectors[b])  # every distance above is row-wise, so pairs go in one call
    c = bandwidth_constant(reference, candidate)
    facts |= {"bandwidth_constant": c, "n_same_key_pairs": len(a)}
    return (lambda q, n: c * raw(q, n)), facts
