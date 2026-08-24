"""Per-generator breakdown of retrieval-support diagnostic value (nAURC), stratified by
crit_gen_true - extends bootstrap_risk_coverage.py's aggregate analysis.

Motivation: the aggregate nAURC (bootstrap_risk_coverage.py, Contribution 3 in
paper-sr/DIRECTION.md) pools every covered record together regardless of which critical
generator's group it belongs to. That can mask heterogeneity - a diagnostic (e.g.
neighborhood_compactness, "cluster density") might be informative for some generators and
uninformative or counterproductive for others, invisible inside one pooled number.

Reuses the exact same point-estimate risk_coverage()/nAURC computation as
bootstrap_risk_coverage.py (_risk_coverage_point, _naurc, same METRICS/COVERAGES), applied
separately within each crit_gen_true group instead of over the whole dataset at once. No new
benchmark run needed beyond whatever already produced the report-service-group-kfold[-*].joblib
this reads - dataset-parameterized (2026-08-06) via --dataset, same convention as
bootstrap_risk_coverage.py.

Run from repository root:
    uv run python scripts/service/per_generator_diagnostic_value.py --dataset bus39
    uv run python scripts/service/per_generator_diagnostic_value.py --dataset eles/2026-06
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import pandas as pd

from scripts.service.bootstrap_risk_coverage import (
    COVERAGES,
    METRICS,
    _dataset_safe_name,
    _load_frame,
    _naurc,
    _risk_coverage_point,
)
from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "paper-sr" / "data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

MIN_GROUP_SIZE: Final[int] = 500  # below this, per-generator nAURC is too noisy to report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bus39")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    output_path = PAPER_DATA_DIR / f"per_generator_diagnostic_naurc_{_dataset_safe_name(args.dataset)}.csv"
    df = _load_frame(args.dataset)
    logger.info(f"Loaded {len(df):,} covered records")

    gen_counts = df["crit_gen_true"].value_counts()
    logger.info(f"Generators: {dict(gen_counts)}")

    rows: list[dict[str, object]] = []

    # Row 0 per (generator, metric): the existing aggregate number, repeated for easy
    # side-by-side comparison against each generator's own figure.
    for name, higher_is_better in METRICS:
        agg_cov = _risk_coverage_point(
            df[name].to_numpy(dtype=float), df["err"].to_numpy(dtype=float), higher_is_better=higher_is_better
        )
        agg_naurc = _naurc({c: mae for c, (mae, _rmse) in agg_cov.items()})
        rows.append({"crit_gen": "ALL (aggregate)", "n": len(df), "metric": name, "naurc": agg_naurc})

    for gen, n in gen_counts.items():
        sub = df[df["crit_gen_true"] == gen]
        if n < MIN_GROUP_SIZE:
            logger.info(f"{gen}: n={n} < {MIN_GROUP_SIZE}, skipping (too small to report a stable nAURC)")
            for name, _ in METRICS:
                rows.append({"crit_gen": gen, "n": int(n), "metric": name, "naurc": float("nan")})
            continue

        for name, higher_is_better in METRICS:
            metric_values = sub[name].to_numpy(dtype=float)
            err_values = sub["err"].to_numpy(dtype=float)
            cov_result = _risk_coverage_point(metric_values, err_values, higher_is_better=higher_is_better)
            naurc = _naurc({c: mae for c, (mae, _rmse) in cov_result.items()})
            rows.append({"crit_gen": gen, "n": int(n), "metric": name, "naurc": naurc})

    out = pd.DataFrame(rows)
    pivot = out.pivot(index="metric", columns="crit_gen", values="naurc")
    # Column order: aggregate first, then generators sorted by descending group size.
    gen_order = ["ALL (aggregate)"] + list(gen_counts.index)
    pivot = pivot[[c for c in gen_order if c in pivot.columns]]

    out.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
