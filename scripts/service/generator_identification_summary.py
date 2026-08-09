"""Persists the generator-identification top-1/top-3 rates cited in paper.tex (Results,
"Critical-generator identification conditional on fault location"), for both datasets.

generator_deoracled_bound.py and eles_deoracled_bound.py already compute these numbers (as
`gen_top1`/`gen_top3` in their in-memory `stats` dict) but only ever print them via
`json.dumps(stats, ...)` - never write them to a CSV, unlike every other number this paper
cites. That gap is exactly how a stale, pre-scaling-fix figure (81.2%, from
paper-sr/EXPERIMENTS.md's "old scaler" column) ended up hand-copied into the manuscript
instead of the current pipeline's actual value: there was no persisted, current-data
artifact to check it against.

Reuses each script's exact denominator convention: `gen_true_rank > 0` (excludes records
where the true critical generator wasn't among the retrieved candidates at all - a coverage
gap, not a mis-identification; see the two scripts' own `stats` computation for the
authoritative definition this mirrors).

Run from the repository root:
    uv run python scripts/service/generator_identification_summary.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "generator_identification_summary.csv"


def _stats(*, dataset: str, df: pd.DataFrame, covered_col: str) -> dict[str, object]:
    cov = df[df[covered_col]]
    ranked = cov[cov["gen_true_rank"] > 0]
    gen_top1 = float((ranked["gen_true_rank"] == 1).mean())
    n_total = int(len(df))
    n_covered = int(len(cov))
    n_ranked = int(len(ranked))
    location_coverage_rate = n_covered / n_total
    candidate_set_recall = n_ranked / n_covered
    joint_rate_within_covered = candidate_set_recall * gen_top1
    success_rate_all_records = (n_ranked * gen_top1) / n_total
    return {
        "dataset": dataset,
        "n_total": n_total,
        "n_covered": n_covered,
        "n_ranked": n_ranked,
        "n_true_gen_not_a_candidate": n_covered - n_ranked,
        "gen_top1": gen_top1,
        "gen_top3": float((ranked["gen_true_rank"] <= 3).mean()),
        "mean_candidate_gens": float(cov["n_candidate_gens"].mean()),
        # Full denominator chain behind the Results sentence on critical-generator
        # identification (paper-sr issue #1, 2026-08-09): location_coverage_rate is
        # n_covered/n_total (some retrieval-visible candidate exists at the recorded
        # location at all); candidate_set_recall is n_ranked/n_covered (conditional on
        # location coverage, the recorded true generator is among those candidates);
        # joint_rate_within_covered is candidate_set_recall * gen_top1 (candidate presence
        # and top-1 selection combined, within the location-covered population);
        # success_rate_all_records divides by n_total instead of n_covered, so it also
        # counts location-uncovered records as failures. None of these condition away the
        # true fault location itself, which every row here still supplies as an input.
        "location_coverage_rate": location_coverage_rate,
        "candidate_set_recall": candidate_set_recall,
        "joint_rate_within_covered": joint_rate_within_covered,
        "success_rate_all_records": success_rate_all_records,
    }


def _bus39_stats() -> dict[str, object]:
    df = pd.read_parquet(TMP_DIR / "generator_deoracled_records.parquet")
    return _stats(dataset="bus39", df=df, covered_col="covered")


def _eles_stats() -> dict[str, object]:
    df = pd.read_parquet(TMP_DIR / "eles_deoracled_records_eles-2026-06_lines_only.parquet")
    return _stats(dataset="eles/2026-06", df=df, covered_col="gen_deoracled_covered")


def main() -> None:
    configure_logging()
    rows = [_bus39_stats(), _eles_stats()]
    for row in rows:
        logger.info(
            f"{row['dataset']}: gen_top1={row['gen_top1']:.4f}, gen_top3={row['gen_top3']:.4f}, "
            f"n_ranked={row['n_ranked']:,} (excluded {row['n_true_gen_not_a_candidate']:,} "
            f"records where the true generator wasn't a candidate at all)"
        )
        logger.info(
            f"{row['dataset']}: location_coverage_rate={row['location_coverage_rate']:.4%}, "
            f"candidate_set_recall={row['candidate_set_recall']:.4%}, "
            f"joint_rate_within_covered={row['joint_rate_within_covered']:.4%}, "
            f"success_rate_all_records={row['success_rate_all_records']:.4%}"
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
