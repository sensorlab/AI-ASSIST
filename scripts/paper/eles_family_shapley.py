"""Exact Shapley attribution of the ELES distance over feature families (exploratory analysis 12, R2.10).

Recorded in the manuscript repository's revision/analyses/12-family-shapley.md before any new coalition was computed.
Players are the five families of _representations.FAMILIES. A coalition's retrieval uses the Euclidean distance on its
families' columns, scaled as in analyses 05 and 11; the empty coalition is uniform weights (analysis 03) and the full
one the deployed distance. Value v(S) = MAE(empty) - MAE(S) on the paired covered records; Shapley values over all 32
coalitions sum to v(full). Intervals: cluster bootstrap over state, topology group and calendar day (widest reported)
and calendar week, recomputing every coalition's MAE on each resample. Coalition runs are cached in tmp/ and reused
from analyses 03, 05 and 11.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_family_shapley.py [n_jobs]
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Hashable, Sequence
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _representations import FAMILIES  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "eles_family_shapley.csv"
PLAYERS: Final[tuple[str, ...]] = tuple(FAMILIES)
KEY: Final[list[str]] = ["state", "record_ordinal"]


def shapley(value: Callable[[frozenset], float], players: Sequence[Hashable]) -> dict[Hashable, float]:
    """Exact Shapley values of the game `value` (a function of a frozenset of players)."""
    n = len(players)
    v = {frozenset(c): value(frozenset(c)) for k in range(n + 1) for c in combinations(players, k)}
    out = {}
    for i in players:
        others = [p for p in players if p != i]
        out[i] = sum(
            factorial(k) * factorial(n - k - 1) / factorial(n) * (v[frozenset(c) | {i}] - v[frozenset(c)])
            for k in range(n)
            for c in combinations(others, k)
        )
    return out


def run_name(coalition: frozenset) -> str:
    """Name under which a coalition's records are cached: the existing runs where they exist."""
    members = [p for p in PLAYERS if p in coalition]
    if not members:
        return "uniform"
    if len(members) == len(PLAYERS):
        return "exact_lines"
    if len(members) == 1:
        return f"only_{members[0]}"
    if len(members) == len(PLAYERS) - 1:
        return f"without_{next(p for p in PLAYERS if p not in coalition)}"
    return "families_" + "+".join(members)


COALITIONS: Final[list[frozenset]] = [frozenset(c) for k in range(len(PLAYERS) + 1) for c in combinations(PLAYERS, k)]


def _records(coalition: frozenset, tag: str, exclusions, n_jobs: int) -> pd.DataFrame:
    name = run_name(coalition)
    path = TMP_DIR / f"harness_records_{name}_{tag}.parquet"
    if not path.exists():
        from _retrieval_harness import EXACT_LINES, Rule, run_rule

        if not name.startswith("families_"):
            raise FileNotFoundError(f"{path.name} should exist from analyses 03, 05 or 11")
        rule = Rule(f"exact_lines_{name}", EXACT_LINES.key, EXACT_LINES.max_hamming, representation=name)
        records, _, _ = run_rule(rule, exclusions, n_jobs=n_jobs)
        records.to_parquet(path, index=False)
        logger.info(f"ran {name} {tag}")
    return pd.read_parquet(path, columns=[*KEY, "covered", "pred_selection", "cct_true"])


def state_errors(frames: dict[frozenset, pd.DataFrame]) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Per state: covered-record count and, per coalition (column order COALITIONS), summed absolute error."""
    base = frames[COALITIONS[0]].set_index(KEY)
    sums = []
    for c in COALITIONS:
        f = frames[c].set_index(KEY).reindex(base.index)
        if not f["covered"].equals(base["covered"]) or not f["cct_true"].equals(base["cct_true"]):
            raise ValueError(f"{run_name(c)}: coverage or labels differ from uniform weights")
        cov = f[f["covered"]]
        sums.append((cov["pred_selection"] - cov["cct_true"]).abs().groupby(level="state").sum())
    states = pd.Index(sorted(base.index.get_level_values("state").unique()))
    counts = base[base["covered"]].groupby(level="state").size().reindex(states, fill_value=0).to_numpy(float)
    errors = np.column_stack([s.reindex(states, fill_value=0.0).to_numpy() for s in sums])
    return states, counts, errors


def statistics(counts: np.ndarray, errors: np.ndarray, w: np.ndarray) -> dict[str, float]:
    mae = (w @ errors) / (w @ counts)
    v = dict(zip(COALITIONS, mae[0] - mae, strict=True))
    phi = shapley(v.__getitem__, PLAYERS)
    full = v[COALITIONS[-1]]
    out = {f"mae_{run_name(c)}": m for c, m in zip(COALITIONS, mae, strict=True)}
    out["value_full"] = full
    out |= {f"shapley_{p}": phi[p] for p in PLAYERS}
    out |= {f"share_{p}": phi[p] / full for p in PLAYERS}
    return out


def main() -> None:
    from _common import load_eles_state_timestamps
    from _retrieval_harness import causal_exclusions
    from referral_decomposition import (
        DECISION_UNITS,
        ELES_DIR,
        N_BOOTSTRAP,
        RAW_ZIP,
        SEED,
        VALIDATION,
        _eles_clusters,
    )

    from src.config.logging import configure_logging

    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    state_ids = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl").index.astype(str)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(state_ids)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")

    rows = []
    for availability, tag, exclusions in (
        ("retrospective", "loso", None),
        ("causal_lag0", "causal0h", causal_exclusions(timestamps, 0.0)),
    ):
        frames = {c: _records(c, tag, exclusions, n_jobs) for c in COALITIONS}
        states, counts, errors = state_errors(frames)
        est = statistics(counts, errors, np.ones(len(states)))
        for unit, clusters in _eles_clusters(states.astype(str)).items():
            codes = np.arange(len(states)) if clusters is None else clusters
            n_units = int(codes.max()) + 1
            rng = np.random.default_rng(SEED)
            boot = []
            for _ in range(N_BOOTSTRAP):
                draws = np.bincount(rng.integers(0, n_units, size=n_units), minlength=n_units)
                boot.append(statistics(counts, errors, draws[codes].astype(np.float64)))
            boot = pd.DataFrame(boot)
            for q, e in est.items():
                rows.append(
                    {
                        "availability": availability,
                        "quantity": q,
                        "estimate": e,
                        "resampling_unit": unit,
                        "n_units": n_units,
                        "decides": unit in DECISION_UNITS,
                        "ci_low": float(np.percentile(boot[q], 2.5)),
                        "ci_high": float(np.percentile(boot[q], 97.5)),
                    }
                )
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
        out = pd.DataFrame(rows)
        out = out[(out["availability"] == availability) & out["decides"]]
        widest = out.assign(width=out["ci_high"] - out["ci_low"]).sort_values("width").groupby("quantity").tail(1)
        for r in widest[widest["quantity"].str.startswith(("shapley_", "value_"))].itertuples():
            logger.info(
                f"{availability} {r.quantity}: {1000 * r.estimate:+.2f} ms "
                f"[{1000 * r.ci_low:+.2f}, {1000 * r.ci_high:+.2f}] under {r.resampling_unit}"
            )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
