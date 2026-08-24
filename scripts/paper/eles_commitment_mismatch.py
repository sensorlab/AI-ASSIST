"""Quantifies generator-commitment mismatch among line-status-matched ELES states.

The deployed topology key matches on line status only, so two states in the same group may
differ in which machines are committed. Methods notes that an uncommitted machine is
represented as zero injection, which is exact for the power flow and lossy dynamically, since
it cannot separate a disconnected machine from one synchronized at zero output. Those differ
in inertia and voltage support, which is what a clearing time depends on. An external review
asked for the size of that effect: how often line-matched neighbors differ in commitment, and
whether the difference shows up in CCT.

Retrieval only ever returns states from the query's own topology group, so commitment mismatch
between line-matched neighbors is exactly the within-group variation of the generator-status
flags. That is a structural property of the archive and needs no retrieval run.

For the second half we pair states within a group that share a (location, critical generator)
contingency, so the two CCTs describe the same fault on the same machine, and ask whether the
absolute CCT difference grows with the number of generator flags on which the pair disagrees.
This is an association across pairs, not a causal estimate: states differing in commitment
also differ in dispatch, and the analysis cannot separate the two.

Run from the repository root:
    uv run python scripts/paper/eles_commitment_mismatch.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06" / "interim"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_commitment_mismatch.csv"
MAX_PAIRS_PER_GROUP: Final[int] = 400
SEED: Final[int] = 42


def main() -> None:
    configure_logging()
    lf = pd.read_pickle(DATASET_DIR / "lf.pkl")
    lines = list(joblib.load(DATASET_DIR / "topology_cols_lines_only.joblib.z"))
    full = list(joblib.load(DATASET_DIR / "topology_cols_full.joblib.z"))
    gen_flags = [c for c in full if c not in set(lines) and lf[c].nunique() > 1]
    logger.info(f"{len(gen_flags)} generator-status flags vary across the record")

    group = lf.groupby(lines, sort=False).ngroup()
    flags = lf[gen_flags].astype(bool).to_numpy()
    states = np.array([str(i) for i in lf.index])

    tsa = pd.read_pickle(DATASET_DIR / "tsa.pkl")
    tsa["state"] = tsa["state"].astype(str)
    outcome = {
        (s, loc, gen): cct
        for s, loc, gen, cct in zip(tsa["state"], tsa["Location"], tsa["Crit_gen"], tsa["CCT"], strict=True)
    }
    by_state: dict[str, set[tuple[str, str]]] = {}
    for s, loc, gen in zip(tsa["state"], tsa["Location"], tsa["Crit_gen"], strict=True):
        by_state.setdefault(s, set()).add((loc, gen))

    rng = np.random.default_rng(SEED)
    n_uncapped = 0  # pairs before the per-group cap, so the sampled total can be read against it
    pair_mismatch: list[int] = []
    contingency_mismatch: list[int] = []
    contingency_dcct: list[float] = []

    # group is indexed by state id; align on position so the groupby yields row offsets.
    positions = pd.Series(np.arange(len(lf)))
    for _gid, idx in positions.groupby(group.to_numpy()).indices.items():
        if len(idx) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1 :]]
        n_uncapped += len(pairs)
        if len(pairs) > MAX_PAIRS_PER_GROUP:
            pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS_PER_GROUP, replace=False)]
        for a, b in pairs:
            differing = int(np.count_nonzero(flags[a] ^ flags[b]))
            pair_mismatch.append(differing)
            shared = by_state.get(states[a], set()) & by_state.get(states[b], set())
            for loc, gen in shared:
                contingency_mismatch.append(differing)
                contingency_dcct.append(abs(outcome[(states[a], loc, gen)] - outcome[(states[b], loc, gen)]))

    pm = np.array(pair_mismatch)
    cm, cd = np.array(contingency_mismatch), np.array(contingency_dcct)
    logger.info(f"within-group state pairs: {len(pm):,}; matched contingencies across pairs: {len(cm):,}")
    logger.info(
        f"pairs differing in commitment: {100 * (pm > 0).mean():.2f}%  "
        f"(median flags differing {np.median(pm):.0f}, mean {pm.mean():.2f}, max {pm.max()})"
    )

    same, diff = cd[cm == 0], cd[cm > 0]
    rho, p_rho = stats.spearmanr(cm, cd)
    logger.info(
        f"|dCCT| on matched contingencies: commitment identical {same.mean():.4f} s (n={len(same):,}), "
        f"differing {diff.mean():.4f} s (n={len(diff):,})"
    )
    logger.info(f"Spearman(flags differing, |dCCT|) = {rho:.4f} (p={p_rho:.3e})")

    rows = [
        {"quantity": "n_generator_flags_varying", "value": len(gen_flags)},
        {"quantity": "n_within_group_state_pairs", "value": len(pm)},
        {"quantity": "n_within_group_state_pairs_uncapped", "value": n_uncapped},
        {"quantity": "frac_pairs_differing_in_commitment", "value": float((pm > 0).mean())},
        {"quantity": "median_flags_differing", "value": float(np.median(pm))},
        {"quantity": "mean_flags_differing", "value": float(pm.mean())},
        {"quantity": "max_flags_differing", "value": int(pm.max())},
        {"quantity": "n_matched_contingencies", "value": len(cm)},
        {"quantity": "mean_abs_dcct_commitment_identical", "value": float(same.mean()) if len(same) else float("nan")},
        {"quantity": "n_commitment_identical", "value": len(same)},
        {"quantity": "mean_abs_dcct_commitment_differing", "value": float(diff.mean()) if len(diff) else float("nan")},
        {"quantity": "n_commitment_differing", "value": len(diff)},
        {"quantity": "spearman_flags_vs_abs_dcct", "value": float(rho)},
        {"quantity": "spearman_p_value", "value": float(p_rho)},
        {"quantity": "max_pairs_per_group", "value": MAX_PAIRS_PER_GROUP},
        {"quantity": "seed", "value": SEED},
    ]
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
