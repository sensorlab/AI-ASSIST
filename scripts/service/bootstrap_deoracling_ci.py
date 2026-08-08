"""Bootstrap confidence intervals for the de-oracling MAE table (paper.tex Table 1).

Table 1 currently reports point-estimate MAE only, at five (level, dataset) combinations,
while Table 2's diagnostic nAURC values already carry bootstrap 95% CIs (state-level
resampling, 200 resamples, matching Statistical evaluation in Methods). This script closes
that gap using the same convention, without re-running the expensive EstimationService
queries that produced the underlying predictions: full_deoracled_bound.py,
generator_deoracled_bound.py, and eles_deoracled_bound.py already persist per-record
predictions (with a "state" column) to tmp/*.parquet, so this only needs to resample those.

Levels, per dataset:
    oracle                        - true (location, generator) known
    generator_only_selection      - true generator known, top-ranked location
    generator_only_screening_min  - true generator known, worst-case (min) across locations
    fully_deoracled_selection     - neither known, top-ranked (generator, location)
    fully_deoracled_screening_min - neither known, worst-case (min) across all pairs

Run from the repository root:
    uv run python scripts/service/bootstrap_deoracling_ci.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "bootstrap_ci_deoracling.csv"

# Matches bootstrap_risk_coverage.py's convention exactly (same N, seed, percentiles), so the
# two tables' CIs are produced the same way even though this script reads different source data.
N_BOOTSTRAP: Final[int] = int(os.environ.get("BOOTSTRAP_DEORACLING_N", "200"))
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5


def _bootstrap_mae(
    state: np.ndarray, cct_true: np.ndarray, pred: np.ndarray, *, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) for MAE, resampling unique states with
    replacement (not individual records), matching every other bootstrap in this paper.

    Uses DataFrameGroupBy.indices for the state -> row-indices lookup: an O(N) group-position
    map, not an O(n_states x N) boolean-mask-per-state loop (which is what a first pass at
    this used, and which is intractable on BUS39's ~1M records x ~21.8k states - the same
    trap bootstrap_risk_coverage.py's own comments already warn about)."""
    err = np.abs(cct_true - pred)
    point = float(err.mean())

    state_row_indices = pd.Series(np.arange(len(state))).groupby(pd.Series(state).astype(str).to_numpy()).indices
    unique_states = np.array(list(state_row_indices.keys()))
    n_states = len(unique_states)

    samples = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled_states = rng.choice(unique_states, size=n_states, replace=True)
        idx = np.concatenate([state_row_indices[s] for s in sampled_states])
        samples[b] = err[idx].mean()

    return point, float(np.percentile(samples, CI_LOW)), float(np.percentile(samples, CI_HIGH))


def _bus39_levels(rng: np.random.Generator) -> list[dict[str, object]]:
    full = pd.read_parquet(TMP_DIR / "full_deoracled_records.parquet")
    gen = pd.read_parquet(TMP_DIR / "generator_deoracled_records.parquet")

    rows: list[dict[str, object]] = []
    specs = (
        ("oracle", gen, "pred_oracle_gen"),
        ("generator_only_selection", gen, "pred_selection"),
        ("generator_only_screening_min", gen, "pred_screening_min"),
        ("fully_deoracled_selection", full, "pred_selection"),
        ("fully_deoracled_screening_min", full, "pred_screening_min"),
    )
    for level, df, col in specs:
        sub = df[df["covered"] & df[col].notna()]
        point, lo, hi = _bootstrap_mae(
            sub["state"].to_numpy(),
            sub["cct_true"].to_numpy(dtype=np.float64),
            sub[col].to_numpy(dtype=np.float64),
            rng=rng,
        )
        rows.append(
            {"dataset": "bus39", "level": level, "mae_point": point, "mae_ci_low": lo, "mae_ci_high": hi, "n": len(sub)}
        )
        logger.info(f"bus39/{level}: MAE {point:.4f} (95% CI {lo:.4f}-{hi:.4f}, n={len(sub):,})")
    return rows


def _eles_levels(rng: np.random.Generator) -> list[dict[str, object]]:
    df = pd.read_parquet(TMP_DIR / "eles_deoracled_records_eles-2026-06_lines_only.parquet")

    rows: list[dict[str, object]] = []
    specs = (
        ("oracle", "full_deoracled_covered", "pred_oracle_loc_and_gen"),
        ("generator_only_selection", "gen_deoracled_covered", "pred_gen_deoracled_selection"),
        ("generator_only_screening_min", "gen_deoracled_covered", "pred_gen_deoracled_screening_min"),
        ("fully_deoracled_selection", "full_deoracled_covered", "pred_full_deoracled_selection"),
        ("fully_deoracled_screening_min", "full_deoracled_covered", "pred_full_deoracled_screening_min"),
    )
    for level, covered_col, col in specs:
        sub = df[df[covered_col] & df[col].notna()]
        point, lo, hi = _bootstrap_mae(
            sub["state"].to_numpy(),
            sub["cct_true"].to_numpy(dtype=np.float64),
            sub[col].to_numpy(dtype=np.float64),
            rng=rng,
        )
        rows.append(
            {
                "dataset": "eles/2026-06",
                "level": level,
                "mae_point": point,
                "mae_ci_low": lo,
                "mae_ci_high": hi,
                "n": len(sub),
            }
        )
        logger.info(f"eles/2026-06/{level}: MAE {point:.4f} (95% CI {lo:.4f}-{hi:.4f}, n={len(sub):,})")
    return rows


def main() -> None:
    configure_logging()
    rng = np.random.default_rng(SEED)

    t0 = time.monotonic()
    rows = _bus39_levels(rng) + _eles_levels(rng)
    elapsed = time.monotonic() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    out = pd.DataFrame(rows)
    out["n_bootstrap"] = N_BOOTSTRAP
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
