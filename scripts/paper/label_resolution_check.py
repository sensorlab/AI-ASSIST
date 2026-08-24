"""Tests whether the ELES retrieval-versus-gradient-boosting gap survives the label resolution.

CCT is not measured, it is found by bisection until the bracket narrows to 0.01 s, on a solver
whose base step is also 0.01 s. Every label in both datasets therefore lies exactly on a 0.01 s
grid: 182 distinct values on ELES, 192 on BUS39. The gap the paper reports between retrieval and
gradient boosting on matched ELES records is 0.00475 s, under half of one grid step, so it is
smaller than the quantity the labels can express. A difference that small can be statistically
significant and still describe nothing a simulation could have resolved.

Three things are measured here, and they answer different questions:

- the paired difference with a state-clustered bootstrap interval, since records from one state
  are not independent and record-level MAE weights states with many contingencies more heavily;
- the same comparison after rounding both predictions to the label grid, which asks whether the
  ordering survives being expressed in the units the ground truth actually has;
- how often the two methods differ by at least one grid step on the same record, which asks how
  much of the aggregate gap comes from differences the labels could represent at all.

Run from the repository root:
    uv run python scripts/paper/label_resolution_check.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "label_resolution_check.csv"
RETRIEVAL: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_group_kfold.parquet"
SUPERVISED: Final[Path] = TMP_DIR / "ml_benchmark_predictions-eles-2026-06.parquet"
MODEL: Final[str] = "hist_gradient_boosting"
# The bisection tolerance and the solver base step, both 0.01 s (Methods). Not a chosen
# parameter: it is the width of the interval the search stops at, so no label can distinguish
# two clearing times closer than this.
GRID: Final[float] = 0.01
N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42


def _clustered_ci(diffs_by_state: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI on the mean paired difference, resampling whole states."""
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(diffs_by_state), size=len(diffs_by_state))
        draws[b] = np.concatenate([diffs_by_state[i] for i in picked]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    configure_logging()
    r = pd.read_parquet(RETRIEVAL)
    if "record_ordinal" not in r.columns:
        r["record_ordinal"] = r.groupby("state", sort=False).cumcount()
    r = r[r["covered"] & r["cct_weighted_per_location"].notna()].copy()
    r["state"] = r["state"].astype(str)

    m = pd.read_parquet(SUPERVISED)
    m = m[m["model"] == MODEL].copy()
    m["state"] = m["state"].astype(str)
    if "record_ordinal" not in m.columns:
        m["record_ordinal"] = m.groupby("state", sort=False).cumcount()

    j = r.merge(
        m[["state", "record_ordinal", "cct_true", "cct_pred"]], on=["state", "record_ordinal"], suffixes=("", "_ml")
    )
    if not np.allclose(j["cct_true"], j["cct_true_ml"], atol=1e-12):
        raise ValueError("cct_true disagrees between arms after the record-key join")
    logger.info(f"{len(j):,} matched records across {j['state'].nunique():,} states")

    truth = j["cct_true"].to_numpy(float)
    if not np.allclose(truth / GRID, np.round(truth / GRID), atol=1e-9):
        raise ValueError(f"labels are not on the {GRID} s grid this analysis assumes")

    ret, gb = j["cct_weighted_per_location"].to_numpy(float), j["cct_pred"].to_numpy(float)
    err_ret, err_gb = np.abs(truth - ret), np.abs(truth - gb)
    # Rounding to the grid asks the comparison in the units the labels have. Retrieval's estimate
    # is a kernel-weighted mean of gridded values and the model's is a fitted real number, so
    # neither lands on the grid by construction; both are rounded the same way.
    q_ret, q_gb = np.round(ret / GRID) * GRID, np.round(gb / GRID) * GRID
    qerr_ret, qerr_gb = np.abs(truth - q_ret), np.abs(truth - q_gb)

    rng = np.random.default_rng(SEED)
    # Factorize first: the ids are strings, and np.diff cannot subtract them.
    codes = pd.factorize(j["state"], sort=False)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1

    rows = []
    for label, a, b in (("raw", err_ret, err_gb), ("rounded to label grid", qerr_ret, qerr_gb)):
        d = a - b
        ci_lo, ci_hi = _clustered_ci(np.split(d[order], bounds), rng)
        rows.append(
            {
                "comparison": label,
                "mae_retrieval": float(a.mean()),
                "mae_gradient_boosting": float(b.mean()),
                "mean_paired_diff": float(d.mean()),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "diff_in_grid_steps": float(d.mean() / GRID),
                "retrieval_lower": bool(a.mean() < b.mean()),
            }
        )
        logger.info(
            f"{label}: retrieval {a.mean():.5f} vs gradient boosting {b.mean():.5f}; "
            f"paired diff {d.mean():+.5f} s ({d.mean() / GRID:+.2f} grid steps, CI {ci_lo:+.5f} {ci_hi:+.5f})"
        )

    within = float((np.abs(err_ret - err_gb) < GRID).mean())
    same_cell = float((q_ret == q_gb).mean())
    logger.info(
        f"per-record absolute errors differ by less than one grid step on {within:.1%} of records; "
        f"the two predictions round to the same label on {same_cell:.1%}"
    )
    rows += [
        {"comparison": "records where methods differ by < 1 grid step", "mean_paired_diff": within},
        {"comparison": "records where predictions round to the same label", "mean_paired_diff": same_cell},
        {"comparison": "grid step (s)", "mean_paired_diff": GRID},
        {"comparison": "n_records", "mean_paired_diff": float(len(j))},
        {"comparison": "n_states", "mean_paired_diff": float(j["state"].nunique())},
    ]
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
