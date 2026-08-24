"""Auditable replacement for an earlier throwaway sensitivity check (ai2ai.md, 2026-08-09,
Codex review): quantifies how much the nAURC(MAE) values in journal.tex Table 2 depend on
how ties among a diagnostic's values are broken, without touching the committed
bootstrap_ci_selected_*.csv files those numbers actually come from.

For each of the five main-table diagnostics on both datasets, computes the point-estimate
nAURC(MAE) under three tie policies from src.benchmarking.risk_coverage_point:
  - "hard": the original, order-dependent tie-break (argsort's default order) - what every
    currently committed number uses.
  - "randomized": N_RANDOM_SEEDS independent random permutations *within* each exact-value
    group (no numerical jitter - ties are found by exact equality, then permuted with an
    index-level shuffle), reporting the min/max nAURC observed across seeds. This is the
    same statistic "hard" produces, just under different arbitrary tie orders, so its spread
    across seeds measures how much the arbitrariness of "hard" can move the reported number.
  - "fractional": the tie-safe estimator proposed as a fix - deterministic, not order-
    dependent, provably reproduces "hard" whenever no tie straddles a coverage cutoff.

A diagnostic whose "randomized" spread is small relative to its reported bootstrap CI width
is not materially affected by tie order; a large spread means the tie-breaking choice alone
can move the number by a practically significant amount, independent of any bootstrap
resampling.

Run from the repository root:
    uv run python scripts/paper/tie_sensitivity_check.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmarking import naurc, risk_coverage_point  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "tie_sensitivity_check.csv"

N_RANDOM_SEEDS: Final[int] = 20
SEED: Final[int] = 42

# Matches bootstrap_risk_coverage_selected.py's MAIN_METRIC_NAMES/METRIC_SETS["main"] exactly -
# the five diagnostics displayed in Table 2.
MAIN_METRICS: Final[tuple[tuple[str, bool], ...]] = (
    ("cct_weighted_std", False),
    ("distance_min", False),
    ("location_weight_mass", True),
    ("n_eff", True),
    ("neighborhood_compactness", True),
)

DATASETS: Final[tuple[tuple[str, Path], ...]] = (
    ("bus39", TMP_DIR / "generator_diagnostics_selected_bus39.parquet"),
    ("eles/2026-06", TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"),
)


def _load_common_support_frame(path: Path) -> pd.DataFrame:
    # Mirrors bootstrap_risk_coverage_selected.py's _load_frame + _common_support_mask
    # exactly: selected-covered records with a defined location-covered estimate, then
    # masked to the population where every one of the five main diagnostics is finite.
    df = pd.read_parquet(path)
    df = df[df["covered"]].dropna(subset=["cct_weighted_per_location"]).copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    mask = np.isfinite(df["err"].to_numpy(dtype=np.float64))
    for name, _ in MAIN_METRICS:
        mask &= np.isfinite(df[name].to_numpy(dtype=np.float64))
    return df[mask]


def main() -> None:
    configure_logging()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for dataset_name, path in DATASETS:
        if not path.exists():
            logger.warning(f"Skipping {dataset_name}: {path} not found (run generator_diagnostics_selected first)")
            continue
        df = _load_common_support_frame(path)
        err = df["err"].to_numpy(dtype=np.float64)
        logger.info(f"[{dataset_name}] common-support population: {len(df):,} records")

        for name, higher_is_better in MAIN_METRICS:
            metric = df[name].to_numpy(dtype=np.float64)

            hard_cov = risk_coverage_point(metric, err, higher_is_better=higher_is_better, tie_policy="hard")
            hard_naurc = naurc({c: mae for c, (mae, _rmse) in hard_cov.items()})

            frac_cov = risk_coverage_point(metric, err, higher_is_better=higher_is_better, tie_policy="fractional")
            frac_naurc = naurc({c: mae for c, (mae, _rmse) in frac_cov.items()})

            random_naurcs = []
            for _ in range(N_RANDOM_SEEDS):
                cov = risk_coverage_point(
                    metric, err, higher_is_better=higher_is_better, tie_policy="randomized", rng=rng
                )
                random_naurcs.append(naurc({c: mae for c, (mae, _rmse) in cov.items()}))
            random_naurcs = np.array(random_naurcs)

            row = {
                "dataset": dataset_name,
                "metric": name,
                "n_records": len(df),
                "hard_naurc": hard_naurc,
                "fractional_naurc": frac_naurc,
                "randomized_naurc_min": float(random_naurcs.min()),
                "randomized_naurc_max": float(random_naurcs.max()),
                "randomized_naurc_spread": float(random_naurcs.max() - random_naurcs.min()),
                "n_random_seeds": N_RANDOM_SEEDS,
            }
            rows.append(row)
            logger.info(
                f"[{dataset_name}/{name}] hard={hard_naurc:.5f} fractional={frac_naurc:.5f} "
                f"randomized=[{random_naurcs.min():.5f}, {random_naurcs.max():.5f}] "
                f"spread={row['randomized_naurc_spread']:.5f}"
            )

    out = pd.DataFrame(rows)
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
