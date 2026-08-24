"""Prototype: can two states' SSSA modes be linked by participation-vector similarity,
when they can't be linked by mode_id or by exact generator-set equality?

Investigation trigger: the SSSA Generator-Set Matching investigation
(datasets/eles/2026-06/README.md's "Conclusion") found exact generator-set equality doesn't
work for eles/2026-06 at either granularity checked - state-vs-state it's nearly as fragmented
as topology "full", and within a state the median mode shares its exact generator set with no
other mode at all (see scripts/evaluation/eles_sssa_generator_set_eval.py and
reports/xx_sssa_analysis.ipynb). Within one state, (state, mode_id) already identifies "the
same" reading; the open question is whether two DIFFERENT states' modes can be related at all,
given mode_id numbering is state-local and explicitly not comparable across states (README's
SSSA section). This checks whether a mode's participation-factor *vector* is similar enough
across states to serve as a matching key, using the mode's eigenvalue (real_part/imag_part -
damping/frequency) as an independent sanity check: a genuine cross-state counterpart should
also land close in eigenvalue, not just look similar in participation shape, since two
unrelated modes could coincidentally share a similar-looking participation vector.

Method: one vector per (state, mode_id), one entry per generator in the dataset, valued by
that generator's overall participation magnitude in the mode - max(ParMag_speed, ParMag_phi,
ParMag_Psi1d, ParMag_Psifd, ParMag_Psi1q, ParMag_Psi2q), 0 where the generator has no
participation row for that mode at all (a real absence per the README, not noise). Nearest
neighbor is by cosine similarity (robust to the very unequal per-mode generator coverage
already measured - median 7 of 166 generators), restricted to modes from a DIFFERENT state -
same-state neighbors are excluded by construction rather than filtered after the fact, since
matching within a state is the already-settled question above. The resulting neighbor pairs'
eigenvalue distance is compared against a random-pairing baseline: if cosine-similar modes also
cluster tighter in eigenvalue than random pairs do, that's evidence the participation vector
carries real cross-state signal, not just noise that happens to look similar.

First finding (pure cosine-NN, kept below as `_select_by_cosine`): median eigenvalue distance
0.064 vs a 2.22 random baseline - strong overall signal, but ~15% of matches are exact cosine
ties (distance 0.0, typically a mode dominated by one or two generators, where cosine similarity
is blind to magnitude) and ~18% of *those* still land far apart in eigenvalue - a real
false-positive tail. `_select_by_combined_rank` addresses this: instead of taking the single
top cosine-ranked cross-state candidate, it pulls a wider candidate pool (`N_CANDIDATES`) per
mode and re-ranks them by *combined* cosine-rank + eigenvalue-rank (Borda-count style - summing
raw ranks rather than raw distances sidesteps the two metrics' wildly different scales: cosine
distance here is typically ~1e-4, eigenvalue distance ~1-10). This lets a slightly-less-cosine-
similar candidate win if it's a much better eigenvalue match, which a pure top-1 cosine pick
can never do.

A natural follow-up question: cosine similarity is equivalent to L2-normalizing each vector
and then comparing them (`‖u−v‖² = 2(1 − cos_sim)` for unit vectors u, v is a monotonic
transform of cosine similarity), so normalizing-then-Euclidean/MSE would rank candidates
identically to cosine - it changes nothing. What *would* differ is skipping normalization
entirely and running Euclidean/MSE directly on the raw participation vectors
(`_fit_euclidean_pool`/`_select_by_euclidean`), so overall participation magnitude - not just
shape - drives the match. That reintroduces exactly the sensitivity cosine was chosen to avoid
(uneven per-mode generator coverage, median 7 of 166 generators, means magnitude differences
can be pure export-coverage noise rather than a genuine dynamical difference). Measured result:
worse than combined-rank on both datasets, and worse than plain cosine-only too on
`eles/2026-01` (bad-match rate 3.4% -> 0.9% cosine/raw-euclidean on `eles/2026-06`, and
11.9% -> 17.0% on `eles/2026-01`, vs combined-rank's 0.2%/7.3%) - the magnitude signal added is
mostly the coverage noise it was expected to be, not real cross-state signal.

A third normalization is also worth distinguishing, since it is *not* equivalent to either of
the above: L1 (sum-to-one) normalization (`_l1_normalize`/`_fit_l1_euclidean_pool`/
`_select_by_l1_euclidean`). Cosine similarity is invariant to rescaling either vector by any
positive scalar - L1 norm, L2 norm, or otherwise - so L1-normalizing doesn't change the cosine
ranking either. But unlike L2-normalized vectors, L1-normalized vectors don't all share the
same L2 norm afterwards - two modes with identical participation *shape* but different
*concentration* (one generator dominating vs. participation spread evenly across many) still
differ in post-L1-normalization L2 norm, and Euclidean distance on them picks that up. So
L1-then-Euclidean removes the total-magnitude confound that hurt raw-Euclidean above, while
still differing from cosine in a way cosine structurally cannot. Measured result: no better -
1.0%/16.7% bad-match rate on `eles/2026-06`/`eles/2026-01`, statistically indistinguishable from
raw-Euclidean's 0.9%/17.0% and still well behind combined-rank's 0.2%/7.3%. Participation
*concentration* turns out to be about as noisy a proxy as total magnitude was, not a source of
real cross-state signal cosine was missing.

`eles/2026-01` has a differently-shaped participation table (no per-generator-state-variable
breakdown - a plain `ParMag` column, not `ParMag_speed`/`ParMag_phi`/... - see
datasets/eles/2026-01/README.md's SSSA section), so `_build_participation_matrix()` takes the
ParMag column(s) to aggregate as a parameter rather than a hardcoded constant. It also has
~179 modes/state vs `eles/2026-06`'s ~14.6 (41.2M sssa rows vs 860k), which would make the
brute-force O(n^2) candidate-pool fit take on the order of an hour at full scale - `MAX_STATES`
subsamples states down to a size comparable to `eles/2026-06`'s run (~90k modes) purely to keep
this prototype interactively runnable; a real evaluation would need the full dataset.

Status: investigation only - no production code changed, no EstimationService/Qdrant
involved. Pure pandas/numpy/scikit-learn over the already-parsed interim/sssa.pkl.

Run from repository root:
    uv run python scripts/evaluation/eles_sssa_mode_similarity_eval.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _resolve_dataset_file

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root; SSSA is out of paper-sr's TSA-only scope
# (DIRECTION.md Decisions), so both outputs go to tmp/ rather than results/data/
# (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Per-dataset participation-column shape (see datasets/<name>/README.md's SSSA section) and a
# subsample cap so the brute-force candidate-pool fit stays interactively runnable - None means
# use every state. eles/2026-06 fits in full (62k modes, ~1min); eles/2026-01's ~179 modes/state
# would push the same fit to roughly an hour, so it's capped to a comparable mode count instead.
DATASET_CONFIGS: Final[dict[str, dict]] = {
    "eles/2026-06": {
        "parmag_cols": (
            "ParMag_speed",
            "ParMag_phi",
            "ParMag_Psi1d",
            "ParMag_Psifd",
            "ParMag_Psi1q",
            "ParMag_Psi2q",
        ),
        "max_states": None,
    },
    "eles/2026-01": {
        "parmag_cols": ("ParMag",),
        "max_states": 500,  # ~500 * 179 modes/state =~ 90k, comparable to eles/2026-06's 62k
    },
}

# Candidate pool size per mode (query itself is always rank 0). N_NEIGHBORS is the pool
# _select_by_cosine scans for the first cross-state candidate; N_CANDIDATES is the wider pool
# _select_by_combined_rank re-ranks by combined cosine+eigenvalue rank - both checked
# empirically to be enough headroom, see main().
N_NEIGHBORS: Final[int] = 10
N_CANDIDATES: Final[int] = 50
RANDOM_SEED: Final[int] = 0


def _build_participation_matrix(sssa: pd.DataFrame, parmag_cols: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per (state, mode_id), one column per generator: that generator's overall
    participation magnitude in the mode, 0 where it has no participation row for that mode.
    parmag_cols is the per-dataset ParMag column(s) to aggregate (max) into that single value -
    eles/2026-06 breaks ParMag down per generator-state-variable, eles/2026-01 doesn't."""
    parmag = sssa[["state", "mode_id", "generator", *parmag_cols]].copy()
    parmag["parmag"] = parmag[list(parmag_cols)].max(axis=1, skipna=True).fillna(0.0)

    pivot = parmag.pivot_table(index=["state", "mode_id"], columns="generator", values="parmag", fill_value=0.0)

    modes = pivot.index.to_frame(index=False).reset_index(drop=True)
    vectors = pivot.to_numpy(dtype=np.float64)
    return modes, vectors


def _fit_candidate_pool(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-NN candidate pool shared by both selection strategies below - fitting once and
    reusing the pool keeps the comparison between them apples-to-apples (same candidates,
    different ranking) and avoids paying the brute-force distance computation twice."""
    nn = NearestNeighbors(n_neighbors=N_CANDIDATES, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(vectors)
    return nn.kneighbors(vectors)


def _fit_euclidean_pool(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean-NN candidate pool on the same raw (unnormalized) vectors - a separate fit from
    the cosine pool above, since Euclidean and cosine can rank the same vectors' neighbors
    differently (Euclidean cares about magnitude, cosine deliberately doesn't)."""
    nn = NearestNeighbors(n_neighbors=N_CANDIDATES, metric="euclidean", algorithm="brute", n_jobs=-1)
    nn.fit(vectors)
    return nn.kneighbors(vectors)


def _l1_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize to sum-to-one (L1) - each entry becomes that generator's *share* of the
    mode's total participation rather than its absolute magnitude. Not equivalent to cosine
    (see module docstring): cosine similarity is invariant to per-vector rescaling by any norm,
    but Euclidean distance on L1-normalized vectors still depends on each vector's
    post-normalization L2 norm - i.e. how concentrated vs. spread-out its shares are - which
    cosine similarity discards. A handful of modes have zero total participation across every
    generator (fillna(0.0) in _build_participation_matrix with no matching row anywhere); those
    are left as all-zero rather than divided by zero."""
    sums = vectors.sum(axis=1, keepdims=True)
    return np.divide(vectors, sums, out=np.zeros_like(vectors), where=sums != 0)


def _fit_l1_euclidean_pool(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean-NN candidate pool on L1-normalized (participation-share) vectors - removes each
    mode's total participation level as a confound, unlike _fit_euclidean_pool, while still
    differing from cosine (see _l1_normalize)."""
    normalized = _l1_normalize(vectors)
    nn = NearestNeighbors(n_neighbors=N_CANDIDATES, metric="euclidean", algorithm="brute", n_jobs=-1)
    nn.fit(normalized)
    return nn.kneighbors(normalized)


def _select_nearest_cross_state(
    modes: pd.DataFrame, distances: np.ndarray, indices: np.ndarray, distance_col: str
) -> pd.DataFrame:
    """Single top-ranked candidate from a DIFFERENT state, per mode - scans only the first
    N_NEIGHBORS columns of the pool (checked empirically to be enough headroom to find a
    cross-state candidate). Shared by _select_by_cosine and _select_by_euclidean; only the
    candidate pool passed in and the resulting distance column name differ."""
    states = modes["state"].to_numpy()
    n = len(modes)
    best_idx = np.full(n, -1, dtype=np.int64)
    best_dist = np.full(n, np.nan, dtype=np.float64)
    n_exhausted = 0
    for row in range(n):
        for col in range(1, N_NEIGHBORS):  # col 0 is always the point itself (distance 0)
            candidate = indices[row, col]
            if states[candidate] != states[row]:
                best_idx[row] = candidate
                best_dist[row] = distances[row, col]
                break
        else:
            n_exhausted += 1

    if n_exhausted:
        logger.warning(
            f"{distance_col}: {n_exhausted}/{n} modes had no cross-state candidate within {N_NEIGHBORS} neighbors"
        )

    found = best_idx >= 0
    return pd.DataFrame(
        {
            "state": modes["state"].to_numpy()[found],
            "mode_id": modes["mode_id"].to_numpy()[found],
            "neighbor_state": modes["state"].to_numpy()[best_idx[found]],
            "neighbor_mode_id": modes["mode_id"].to_numpy()[best_idx[found]],
            distance_col: best_dist[found],
        }
    )


def _select_by_cosine(modes: pd.DataFrame, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    """Baseline strategy: the single top cosine-ranked candidate from a DIFFERENT state."""
    return _select_nearest_cross_state(modes, distances, indices, "cosine_distance")


def _select_by_euclidean(modes: pd.DataFrame, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    """Raw-magnitude strategy: the single nearest candidate from a DIFFERENT state by Euclidean
    distance on the *unnormalized* participation vectors - unlike cosine, this lets overall
    participation magnitude (not just shape) drive the match. See the module docstring for why
    this differs from "normalize then Euclidean" (it doesn't - that's equivalent to cosine)."""
    return _select_nearest_cross_state(modes, distances, indices, "euclidean_distance")


def _select_by_l1_euclidean(modes: pd.DataFrame, distances: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    """Participation-share strategy: the single nearest candidate from a DIFFERENT state by
    Euclidean distance on L1-normalized (sum-to-one) participation vectors - removes total
    participation level as a confound while still differing from cosine (see _l1_normalize)."""
    return _select_nearest_cross_state(modes, distances, indices, "l1_euclidean_distance")


def _rank_along_rows(values: np.ndarray) -> np.ndarray:
    """Rank each row's entries ascending (0 = smallest); ties broken by column order, which is
    irrelevant here since ties only occur among the +inf-masked (same-state) entries, which
    never win a combined-rank argmin anyway."""
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks


def _select_by_combined_rank(
    modes: pd.DataFrame, eigen: pd.DataFrame, distances: np.ndarray, indices: np.ndarray
) -> pd.DataFrame:
    """Re-ranks the same candidate pool by combined cosine-rank + eigenvalue-rank (Borda count)
    instead of taking the single top cosine match - lets a slightly-less-cosine-similar
    candidate win if it's a much closer eigenvalue match, which _select_by_cosine can never do.
    Ranks rather than raw distances are combined deliberately: cosine distance here is
    typically ~1e-4 and eigenvalue distance ~1-10, so summing raw values would just be summing
    eigenvalue distance."""
    lookup = modes[["state", "mode_id"]].merge(eigen, on=["state", "mode_id"], how="left")
    real_arr = lookup["real_part"].to_numpy()
    imag_arr = lookup["imag_part"].to_numpy()
    states = modes["state"].to_numpy()

    cand_states = states[indices]
    same_state_mask = cand_states == states[:, None]

    eig_dist = np.hypot(real_arr[indices] - real_arr[:, None], imag_arr[indices] - imag_arr[:, None])

    cos_masked = np.where(same_state_mask, np.inf, distances)
    eig_masked = np.where(same_state_mask, np.inf, eig_dist)
    combined_rank = _rank_along_rows(cos_masked) + _rank_along_rows(eig_masked)

    n = len(modes)
    best_col = np.argmin(combined_rank, axis=1)
    rows = np.arange(n)
    unmatched = np.isinf(cos_masked[rows, best_col])
    n_unmatched = int(unmatched.sum())
    if n_unmatched:
        logger.warning(f"combined: {n_unmatched}/{n} modes had no cross-state candidate within {N_CANDIDATES} pool")

    found = ~unmatched
    best_idx = indices[rows, best_col]
    return pd.DataFrame(
        {
            "state": modes["state"].to_numpy()[found],
            "mode_id": modes["mode_id"].to_numpy()[found],
            "neighbor_state": modes["state"].to_numpy()[best_idx[found]],
            "neighbor_mode_id": modes["mode_id"].to_numpy()[best_idx[found]],
            "cosine_distance": distances[rows, best_col][found],
        }
    )


def _eigenvalue_distance(matched: pd.DataFrame, eigen: pd.DataFrame) -> pd.DataFrame:
    """Attach each pair's eigenvalue gap - the independent sanity check on whether
    participation-vector similarity actually tracks physically related modes."""
    out = matched.merge(eigen, on=["state", "mode_id"], how="left")
    out = out.merge(
        eigen.rename(columns={"state": "neighbor_state", "mode_id": "neighbor_mode_id"}),
        on=["neighbor_state", "neighbor_mode_id"],
        how="left",
        suffixes=("", "_neighbor"),
    )
    out["real_part_gap"] = (out["real_part"] - out["real_part_neighbor"]).abs()
    out["imag_part_gap"] = (out["imag_part"] - out["imag_part_neighbor"]).abs()
    out["eigenvalue_distance"] = np.hypot(out["real_part_gap"], out["imag_part_gap"])
    return out


def _random_baseline(eigen: pd.DataFrame, n_pairs: int, rng: np.random.Generator) -> pd.Series:
    """Same eigenvalue-distance metric over random (state, mode_id) pairs, as the comparison
    point for whether the cosine-NN pairs above are actually tighter than chance."""
    a = eigen.sample(n=n_pairs, replace=True, random_state=rng.integers(2**32 - 1))
    b = eigen.sample(n=n_pairs, replace=True, random_state=rng.integers(2**32 - 1))
    real_gap = (a["real_part"].to_numpy() - b["real_part"].to_numpy()).flatten()
    imag_gap = (a["imag_part"].to_numpy() - b["imag_part"].to_numpy()).flatten()
    return pd.Series(np.hypot(real_gap, imag_gap), name="eigenvalue_distance")


def _run_for_dataset(
    dataset_name: str, parmag_cols: tuple[str, ...], max_states: int | None, rng: np.random.Generator
) -> None:
    app_settings = get_app_settings()
    slug = dataset_name.replace("/", "-")
    report_path = TMP_DIR / f"report-{slug}-sssa-mode-similarity.joblib"
    summary_csv_path = TMP_DIR / f"sssa_mode_similarity_summary_{slug}.csv"

    interim = app_settings.data_dir / dataset_name / "interim"
    sssa_path = _resolve_dataset_file(interim, "sssa.pkl")
    sssa = pd.read_pickle(sssa_path)
    logger.info(f"[{dataset_name}] Loaded {sssa_path} ({len(sssa)} rows)")

    if max_states is not None:
        all_states = sssa["state"].unique()
        if len(all_states) > max_states:
            sampled_states = rng.choice(all_states, size=max_states, replace=False)
            sssa = sssa[sssa["state"].isin(sampled_states)]
            logger.info(f"[{dataset_name}] Subsampled to {max_states}/{len(all_states)} states ({len(sssa)} rows)")

    t0 = time.monotonic()
    modes, vectors = _build_participation_matrix(sssa, parmag_cols)
    logger.info(
        f"[{dataset_name}] Built {vectors.shape[0]} participation vectors of dimension "
        f"{vectors.shape[1]} in {time.monotonic() - t0:.1f}s"
    )

    eigen = sssa.drop_duplicates(["state", "mode_id"])[["state", "mode_id", "real_part", "imag_part"]]

    t0 = time.monotonic()
    distances, indices = _fit_candidate_pool(vectors)
    logger.info(f"[{dataset_name}] Fit cosine candidate pool ({N_CANDIDATES} per mode) in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    euclid_distances, euclid_indices = _fit_euclidean_pool(vectors)
    logger.info(
        f"[{dataset_name}] Fit euclidean candidate pool ({N_CANDIDATES} per mode) in {time.monotonic() - t0:.1f}s"
    )

    t0 = time.monotonic()
    l1_distances, l1_indices = _fit_l1_euclidean_pool(vectors)
    logger.info(
        f"[{dataset_name}] Fit L1-euclidean candidate pool ({N_CANDIDATES} per mode) in {time.monotonic() - t0:.1f}s"
    )

    matched_cosine = _eigenvalue_distance(_select_by_cosine(modes, distances, indices), eigen)
    matched_combined = _eigenvalue_distance(_select_by_combined_rank(modes, eigen, distances, indices), eigen)
    matched_euclidean = _eigenvalue_distance(_select_by_euclidean(modes, euclid_distances, euclid_indices), eigen)
    matched_l1_euclidean = _eigenvalue_distance(_select_by_l1_euclidean(modes, l1_distances, l1_indices), eigen)
    baseline = _random_baseline(eigen, n_pairs=len(modes), rng=rng)

    summary = pd.DataFrame(
        {
            "cosine_nn": matched_cosine["eigenvalue_distance"].describe(),
            "combined_rank": matched_combined["eigenvalue_distance"].describe(),
            "raw_euclidean": matched_euclidean["eigenvalue_distance"].describe(),
            "l1_euclidean": matched_l1_euclidean["eigenvalue_distance"].describe(),
            "random_baseline": baseline.describe(),
        }
    )
    logger.info(
        f"[{dataset_name}] Eigenvalue distance, cosine-only vs combined-rank vs raw-euclidean vs "
        f"l1-euclidean vs random baseline:\n{summary}"
    )

    n_bad_cosine = (matched_cosine["eigenvalue_distance"] > 1.0).mean()
    n_bad_combined = (matched_combined["eigenvalue_distance"] > 1.0).mean()
    n_bad_euclidean = (matched_euclidean["eigenvalue_distance"] > 1.0).mean()
    n_bad_l1_euclidean = (matched_l1_euclidean["eigenvalue_distance"] > 1.0).mean()
    logger.info(
        f"[{dataset_name}] Fraction of matches with eigenvalue_distance > 1.0: "
        f"cosine-only {n_bad_cosine:.1%}, combined-rank {n_bad_combined:.1%}, "
        f"raw-euclidean {n_bad_euclidean:.1%}, l1-euclidean {n_bad_l1_euclidean:.1%}"
    )

    joblib.dump(
        {
            "matched_cosine": matched_cosine,
            "matched_combined": matched_combined,
            "matched_euclidean": matched_euclidean,
            "matched_l1_euclidean": matched_l1_euclidean,
            "baseline": baseline,
        },
        report_path,
    )
    logger.info(f"[{dataset_name}] Saved matched pairs + baseline to {report_path}")

    summary.to_csv(summary_csv_path)
    logger.info(f"[{dataset_name}] Saved summary CSV to {summary_csv_path}")

    print(f"=== {dataset_name} ===")
    print(summary.to_string())
    print()
    print(
        f"eigenvalue_distance > 1.0: cosine-only {n_bad_cosine:.1%}, combined-rank {n_bad_combined:.1%}, "
        f"raw-euclidean {n_bad_euclidean:.1%}, l1-euclidean {n_bad_l1_euclidean:.1%}"
    )
    print()
    print("Example combined-rank matches (closest cosine distance):")
    print(matched_combined.sort_values("cosine_distance").head(10).to_string(index=False))
    print()


def main() -> None:
    configure_logging()
    rng = np.random.default_rng(RANDOM_SEED)

    for dataset_name, config in DATASET_CONFIGS.items():
        _run_for_dataset(dataset_name, config["parmag_cols"], config["max_states"], rng)


if __name__ == "__main__":
    main()
