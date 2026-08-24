"""Recomputes n_neighbors / n_unique_states nAURC on the selected population behind Table 3.

Table 3 reports five diagnostics. Two more were computed and not shown: n_neighbors (the raw
retrieved count) and n_unique_states. On the *oracle* population in
per_generator_diagnostic_naurc_*.csv both sit below 1.0 on all three available datasets, which
would sit awkwardly beside the manuscript's statement that the results "do not support using
one neighborhood statistic as a portable confidence score" - two of the five reported
diagnostics invert on BUS39, and these two do not. That comparison is not admissible as-is:
Table 3 uses the highest-support-selected population, not the oracle one. This script settles
it on the population the manuscript actually reports.

Two populations are computed per dataset, because they answer different questions:

- "table3_common": the exact five-metric common-support mask Table 3 uses, so the new numbers
  are directly comparable to the published ones. The script reproduces the five published
  nAURC values on this mask first and aborts if any disagrees, so a silent divergence in the
  population or the tie policy cannot be mistaken for a finding.
- "own_support": every selected-covered record where the metric itself is finite. n_neighbors
  carries no n >= 2 requirement, unlike weighted dispersion and compactness, so this shows what
  the common-support restriction costs in coverage.

Tie policy is "fractional" and bootstrap CIs use 1,000 state-level resamples with seed 42,
matching the committed bootstrap_ci_selected_*.csv these numbers are read against.

Run from the repository root:
    uv run python scripts/paper/n_neighbors_selected_naurc.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# The paper's ELES results are the deployed line-status key. bootstrap_risk_coverage_selected
# resolves the ELES parquet through os.environ["TOPOLOGY_VARIANT"], defaulting to lines_only.
# QdrantConfig declares no env_file, so a repository .env does NOT reach os.environ and cannot
# redirect this by itself - but docker-compose passes .env into the container as real
# environment variables, and a developer who exports it gets the same effect. A .env carrying
# slovenia_only would then point this analysis at the no-filter control, which collapses ELES
# to one topology group. Pin it rather than inherit it.
os.environ["TOPOLOGY_VARIANT"] = "lines_only"

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "evaluation"))

from bootstrap_risk_coverage_selected import (  # noqa: E402
    METRIC_SETS,
    _common_support_mask,
    _load_frame,
)

from src.benchmarking import naurc, risk_coverage_point  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "n_neighbors_selected_naurc.csv"

N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5
TIE_POLICY: Final[str] = "fractional"

CANDIDATES: Final[tuple[tuple[str, bool], ...]] = (("n_neighbors", True), ("n_unique_states", True))

DATASETS: Final[tuple[tuple[str, str], ...]] = (
    ("bus39", "bootstrap_ci_selected_bus39.csv"),
    ("eles/2026-06", "bootstrap_ci_selected_eles-2026-06.csv"),
)


def _coverage_maes(metric: np.ndarray, err: np.ndarray, *, higher_is_better: bool) -> dict[float, float]:
    return {
        cov: mae
        for cov, (mae, _rmse) in risk_coverage_point(
            metric, err, higher_is_better=higher_is_better, tie_policy=TIE_POLICY
        ).items()
    }


def _naurc_of(metric: np.ndarray, err: np.ndarray, *, higher_is_better: bool) -> float:
    return naurc(_coverage_maes(metric, err, higher_is_better=higher_is_better))


def _bootstrap_ci(
    metric: np.ndarray, err: np.ndarray, states: np.ndarray, *, higher_is_better: bool
) -> tuple[float, float]:
    """State-clustered percentile CI on nAURC, matching the committed artifacts' convention."""
    by_state = pd.Series(range(len(states))).groupby(states).apply(lambda s: s.to_numpy())
    index_groups = [np.asarray(g) for g in by_state]
    rng = np.random.default_rng(SEED)
    n_groups = len(index_groups)
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        picked = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([index_groups[i] for i in picked])
        draws[b] = _naurc_of(metric[idx], err[idx], higher_is_better=higher_is_better)
    return float(np.percentile(draws, CI_LOW)), float(np.percentile(draws, CI_HIGH))


def _verify_table3(df: pd.DataFrame, published_path: Path) -> None:
    """Reproduce the five published nAURC values before trusting anything new on this mask."""
    published_df = pd.read_csv(published_path)
    published_df = published_df[published_df["quantity"] == "nAURC(MAE)"]
    policies = set(published_df["tie_policy"].unique())
    if len(policies) != 1:
        raise ValueError(f"{published_path.name}: expected one tie policy, found {sorted(policies)}")
    published_tie_policy = policies.pop()
    published = published_df.set_index("metric")["point_estimate"]
    err = df["err"].to_numpy(dtype=np.float64)
    for name, higher_is_better in METRIC_SETS["main"]:
        # Verify under the tie policy each published table was actually computed with, read from
        # the file rather than assumed. TIE_POLICY here is "fractional" because the candidates
        # below are integer counts with heavy ties, but the published files do not agree with
        # each other: BUS39 was written under "hard" and ELES under "fractional". Recomputing
        # both one way compares two different estimators, and passed only while they happened to
        # agree to 1e-9.
        coverage_maes = {
            cov: mae
            for cov, (mae, _rmse) in risk_coverage_point(
                df[name].to_numpy(dtype=np.float64),
                err,
                higher_is_better=higher_is_better,
                tie_policy=published_tie_policy,
            ).items()
        }
        got = naurc(coverage_maes)
        want = float(published.loc[name])
        if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
            raise AssertionError(f"{name}: recomputed {got:.10f} != published {want:.10f}")
    logger.info("  verified: all five published Table 3 nAURC values reproduce exactly on this mask")


def main() -> None:
    configure_logging()
    rows: list[dict[str, object]] = []

    for dataset, published_csv in DATASETS:
        logger.info(f"=== {dataset} ===")
        covered = _load_frame(dataset)
        mask = _common_support_mask(covered, METRIC_SETS["main"])
        table3 = covered[mask]
        logger.info(f"  Table 3 common population: {len(table3):,} records, {table3['state'].nunique():,} states")
        _verify_table3(table3, PAPER_DATA_DIR / published_csv)

        for metric_name, higher_is_better in CANDIDATES:
            for population, frame in (("table3_common", table3), ("own_support", covered)):
                sub = frame[np.isfinite(frame[metric_name].to_numpy(dtype=np.float64))]
                metric = sub[metric_name].to_numpy(dtype=np.float64)
                err = sub["err"].to_numpy(dtype=np.float64)
                point = _naurc_of(metric, err, higher_is_better=higher_is_better)
                lo, hi = _bootstrap_ci(
                    metric, err, sub["state"].astype(str).to_numpy(), higher_is_better=higher_is_better
                )
                logger.info(
                    f"  {metric_name:<17} {population:<14} nAURC={point:.4f} ({lo:.4f}, {hi:.4f})  "
                    f"n={len(sub):,} ({len(sub) / len(covered):.2%} of selected-covered)"
                )
                shared = {
                    "dataset": dataset,
                    "metric": metric_name,
                    "population": population,
                    "n_records": len(sub),
                    "n_states": sub["state"].nunique(),
                    "n_selected_covered": len(covered),
                    "availability_rate": len(sub) / len(covered),
                    "n_bootstrap": N_BOOTSTRAP,
                    "tie_policy": TIE_POLICY,
                    "seed": SEED,
                }
                rows.append(
                    {**shared, "coverage": np.nan, "quantity": "nAURC(MAE)", "point_estimate": point}
                    | {"ci_low": lo, "ci_high": hi}
                )
                # Per-coverage MAEs in the same (metric, coverage, quantity) shape as
                # bootstrap_ci_selected_*.csv, so make_risk_coverage_figure.py can read the
                # five published diagnostics and these two from one common schema.
                for cov, mae in _coverage_maes(metric, err, higher_is_better=higher_is_better).items():
                    rows.append(
                        {**shared, "coverage": cov, "quantity": "mae", "point_estimate": mae}
                        | {"ci_low": np.nan, "ci_high": np.nan}
                    )

    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
