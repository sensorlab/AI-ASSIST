"""Absolute-error quantiles of the conditional critical-generator evaluation (Supplementary Table S1).

Reads the leave-one-state-out record tables written by scripts/evaluation/generator_deoracled_bound.py (BUS39) and
scripts/evaluation/eles_deoracled_bound.py (ELES) and writes results/data/deoracling_quantiles.csv: per dataset and
level (recorded critical generator, highest-support candidate), the number of records and the 0th to 100th
percentiles of absolute CCT error, each row on its own covered population as in main-text Table 2.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/deoracling_quantiles.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "deoracling_quantiles.csv"
QUANTILES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
SOURCES: Final[dict[str, tuple[str, str, str]]] = {  # dataset -> (records file, oracle column, selected column)
    "BUS39": ("generator_deoracled_records_loso", "pred_oracle_gen", "pred_selection"),
    "ELES": (
        "eles_deoracled_records_eles-2026-06_lines_only",
        "pred_oracle_gen_and_loc",
        "pred_gen_deoracled_selection",
    ),
}


def main() -> None:
    rows = []
    for dataset, (name, oracle, selected) in SOURCES.items():
        records = pd.read_parquet(TMP_DIR / f"{name}.parquet")
        for level, col in (("recorded_generator", oracle), ("highest_support_generator", selected)):
            ae = (records[col] - records["cct_true"]).abs().dropna()
            rows.append(
                {"dataset": dataset, "level": level, "n": len(ae)}
                | {f"q{round(100 * q)}": v for q, v in zip(QUANTILES, np.quantile(ae, QUANTILES), strict=True)}
            )
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
