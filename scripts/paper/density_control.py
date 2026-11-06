"""Archive size and fragmentation as the source of the BUS39-ELES difference (analysis 07, R2.2a).

Pre-specified in the manuscript repository's revision/analyses/07-density-control.md, committed before any number here
was computed. On BUS39, with network, features, contingencies, labels, scaler and the deployed setting held fixed:

    A0  all 21,783 states, one group (the validated harness records)
    A1  4,393 states drawn at random, one group
    A2  A1's states in random groups with exactly ELES's 1,785 group sizes
    A3  all states in random groups whose sizes are drawn with replacement from ELES's sizes, the last one truncated

A1 to A3 are repeated over seeds 0 to 19 (A1 and A2 share one generator per seed: subsample, then partition; A3 has its
own generator per seed). Each arm runs leave one state out through the validated harness, a query retrieving only
from its own archive and group, imposed through per-query exclusions. Reported per measure: mean and 2.5th to 97.5th
percentiles over draws, and the reproduced share of the gap (arm - A0) / (ELES - A0), read as "largely" (>= 0.75),
"partly" (0.25 to 0.75) or "little" (< 0.25); gaps under 0.010 s (MAE) or one percentage point (shares) are not read.
ELES and A0 are measured with the same definitions from the validated harness records.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/density_control.py [n_jobs]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _retrieval_harness import LINE_KEY, Rule, run_rule  # noqa: E402
from alpha_k_sensitivity import describe  # noqa: E402
from referral_decomposition import ELES_DIR, ELES_LOSO, _eles_clusters  # noqa: E402
from retrieval_harness_validation_bus39 import HARNESS_RECORDS as BUS39_LOSO  # noqa: E402
from retrieval_harness_validation_bus39 import OUTPUT_PATH as VALIDATION_BUS39  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
BUS39_LF: Final[Path] = PROJECT_DIR / "datasets" / "bus39" / "interim" / "lf.pkl"
RUN_DIR: Final[Path] = PROJECT_DIR / "tmp" / "density_control"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "density_control.csv"
SEEDS: Final[range] = range(20)
RULE: Final[Rule] = Rule("bus39_density", LINE_KEY, 0, dataset="bus39")
COLUMNS: Final[list[str]] = [
    "state",
    "record_ordinal",
    "covered",
    "cct_true",
    "gen_true",
    "gen_selected",
    "pred_selection",
    "pred_screening_min",
]
# Rates only: counts (records, missed unsafe states) scale with archive size and have no comparable gap.
SHARE_MEASURES: Final[tuple[str, ...]] = (
    "record_coverage",
    "complete_state_coverage",
    "any_estimate_state_coverage",
    "mae",
    "top1",
    "referral_load_150ms",
    "recall_150ms",
    "referral_load_200ms",
    "recall_200ms",
)
MIN_GAP: Final[dict[str, float]] = {"mae": 0.010}  # one label step; every other share measure: one percentage point
PRIMARY: Final[tuple[str, ...]] = ("complete_state_coverage", "referral_load_200ms")


def eles_group_sizes() -> np.ndarray:
    """The 1,785 ELES topology-group sizes under the line-status key (counts only)."""
    states = pd.Index(pd.read_pickle(ELES_DIR / "interim" / "lf.pkl").index.astype(str))
    return pd.Series(_eles_clusters(states)["topology_group"]).value_counts().to_numpy()


def partition_exact(states: np.ndarray, sizes: np.ndarray, rng: np.random.Generator) -> dict[str, int]:
    """Random partition of `states` into groups of exactly `sizes`."""
    if int(sizes.sum()) != len(states):
        raise ValueError(f"sizes sum to {int(sizes.sum())}, not {len(states)} states")
    labels = np.repeat(np.arange(len(sizes)), sizes)
    return dict(zip(rng.permutation(states).tolist(), labels.tolist(), strict=True))


def partition_sampled(states: np.ndarray, size_pool: np.ndarray, rng: np.random.Generator) -> dict[str, int]:
    """Random partition with sizes drawn with replacement from `size_pool` until every state is placed."""
    sizes: list[int] = []
    while sum(sizes) < len(states):
        sizes.append(int(rng.choice(size_pool)))
    sizes[-1] -= sum(sizes) - len(states)
    return partition_exact(states, np.array(sizes), rng)


def exclusions(all_states: np.ndarray, group_of: dict[str, int]) -> dict[str, list[str]]:
    """Per query in `group_of`: every state outside its group (and so outside the archive), plus itself."""
    labels = np.array([group_of.get(s, -1) for s in all_states])
    out: dict[str, list[str]] = {}
    for g in np.unique(labels[labels >= 0]):
        outside = all_states[labels != g].tolist()
        for s in all_states[labels == g]:
            out[str(s)] = [*outside, str(s)]
    return out


def reproduced_share(arm: float, a0: float, eles: float) -> float:
    return (arm - a0) / (eles - a0)


def reading(share: float) -> str:
    if share >= 0.75:
        return "largely" if share <= 1 else "overshoots"
    return "partly" if share >= 0.25 else "little"


def measures(records: pd.DataFrame, states: pd.DataFrame | None = None) -> dict[str, float]:
    out = describe(records, states)
    out["any_estimate_state_coverage"] = float(records.groupby("state")["covered"].any().mean())
    return out


def _run(name: str, group_of: dict[str, int], all_states: np.ndarray, n_jobs: int) -> pd.DataFrame:
    path = RUN_DIR / f"{name}.parquet"
    if not path.exists():
        logger.info(f"running {name} ({len(group_of):,} queries)")
        records, _, _ = run_rule(RULE, exclusions(all_states, group_of), n_jobs=n_jobs)
        records[COLUMNS].to_parquet(path, index=False)
    return pd.read_parquet(path)


def summarise(per_draw: pd.DataFrame, a0: dict[str, float], eles: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (arm, quantity), g in per_draw.groupby(["arm", "quantity"]):
        v = g["estimate"]
        rows.append(
            {
                "arm": arm,
                "quantity": quantity,
                "statistic": "mean_over_draws",
                "estimate": v.mean(),
                "low": v.quantile(0.025),
                "high": v.quantile(0.975),
            }
        )
        if quantity not in SHARE_MEASURES:
            continue
        gap = eles[quantity] - a0[quantity]
        if abs(gap) < MIN_GAP.get(quantity, 0.01):
            rows.append({"arm": arm, "quantity": quantity, "statistic": "gap_not_read", "estimate": gap})
            continue
        share = reproduced_share(v, a0[quantity], eles[quantity])  # vectorised over draws
        rows.append(
            {
                "arm": arm,
                "quantity": quantity,
                "statistic": "reproduced_share",
                "estimate": share.mean(),
                "low": share.quantile(0.025),
                "high": share.quantile(0.975),
                "reading": reading(share.mean()),
                "primary": quantity in PRIMARY,
            }
        )
    return rows


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION_BUS39)["passed"].all():
        raise RuntimeError(f"{VALIDATION_BUS39.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    all_states = pd.read_pickle(BUS39_LF).index.astype(str).to_numpy()
    sizes = eles_group_sizes()
    n_eles = int(sizes.sum())

    rows: list[dict[str, Any]] = []
    a0 = measures(pd.read_parquet(BUS39_LOSO))
    eles = measures(
        pd.read_parquet(ELES_LOSO), pd.read_parquet(ELES_LOSO.with_name(ELES_LOSO.stem + "_states.parquet"))
    )
    rows += [{"arm": "A0", "quantity": q, "statistic": "point", "estimate": v} for q, v in a0.items()]
    rows += [{"arm": "ELES", "quantity": q, "statistic": "point", "estimate": v} for q, v in eles.items()]

    per_draw: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        archive = rng.choice(all_states, size=n_eles, replace=False)
        arms = {
            "A1": dict.fromkeys(archive.tolist(), 0),
            "A2": partition_exact(archive, sizes, rng),
            "A3": partition_sampled(all_states, sizes, np.random.default_rng(seed)),
        }
        for arm, group_of in arms.items():
            m = measures(_run(f"{arm}_seed{seed}", group_of, all_states, n_jobs))
            per_draw += [
                {"arm": arm, "draw": seed, "quantity": q, "statistic": "draw", "estimate": v} for q, v in m.items()
            ]
            logger.info(
                f"{arm} seed {seed}: complete coverage {m['complete_state_coverage']:.4f}, "
                f"referral 200 ms {m['referral_load_200ms']:.4f}, MAE {m['mae']:.4f}"
            )
        frame = pd.DataFrame(per_draw)
        pd.concat([pd.DataFrame(rows), frame, pd.DataFrame(summarise(frame, a0, eles))], ignore_index=True).to_csv(
            OUTPUT_PATH, index=False
        )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
