"""Validation gate for the harness's BUS39 path: reproduce production before analysis 06 uses it.

Pre-specified in the manuscript repository's revision/analyses/06-alpha-k-sensitivity.md. With the deployed setting
(alpha = 1, K = 100, leave one state out) the harness must reproduce, record for record, the BUS39 records of
scripts/evaluation/generator_deoracled_bound.py run with BUS39_BENCHMARK_SPLIT=leave-one-state-out
(tmp/generator_deoracled_records_loso.parquet): the covered set and candidate counts exactly, the selected,
screening-minimum and oracle estimates within 1e-9 s. As in the ELES gate, the selected estimate and the true
generator's rank are compared where kernel masses are not tied up to rounding; production's file does not store the
selected generator, so tied records are counted, not checked.

The harness records are kept in tmp/ as analysis 06's deployed BUS39 point.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/retrieval_harness_validation_bus39.py [n_jobs]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from _retrieval_harness import EXACT_LINES, Rule, run_rule  # noqa: E402
from retrieval_harness_validation import _check_close, _check_equal, _untied, _with_ordinal  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PROD_RECORDS: Final[Path] = TMP_DIR / "generator_deoracled_records_loso.parquet"
HARNESS_RECORDS: Final[Path] = TMP_DIR / "harness_records_bus39_loso.parquet"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "retrieval_harness_validation_bus39.csv"
BUS39: Final[Rule] = Rule("bus39", EXACT_LINES.key, EXACT_LINES.max_hamming, dataset="bus39")


def checks(harness: pd.DataFrame, production: pd.DataFrame) -> list[dict[str, Any]]:
    h, p = _with_ordinal(harness), _with_ordinal(production)
    out = [
        _check_equal("bus39 loso: true CCT", h["cct_true"], p["cct_true"]),
        _check_equal("bus39 loso: covered set", h["covered"].astype(bool), p["covered"].astype(bool)),
    ]
    hc, pc = h[h["covered"]], p[p["covered"].astype(bool)]
    top, anywhere = _untied(hc), _untied(hc, any_tie=True)
    out += [
        _check_equal("bus39 loso: candidate generators", hc["n_candidate_gens"].astype(int), pc["n_candidate_gens"]),
        _check_close(
            "bus39 loso: selected estimate, untied",
            hc["pred_selection"].reindex(top),
            pc["pred_selection"].reindex(top),
        ),
        _check_close("bus39 loso: screening minimum", hc["pred_screening_min"], pc["pred_screening_min"]),
        _check_close("bus39 loso: oracle estimate", hc["pred_oracle"], pc["pred_oracle_gen"]),
        _check_equal(
            "bus39 loso: true generator rank, no tie",
            hc["gen_true_rank"].reindex(anywhere).astype(int),
            pc["gen_true_rank"].reindex(anywhere).astype(int),
        ),
    ]
    out.append(
        {
            "check": "bus39 loso: tied top generator (counted, not checked)",
            "n_compared": int(len(hc) - len(top)),
            "n_mismatch": 0,
            "max_abs_diff": np.nan,
            "tolerance": np.nan,
            "passed": True,
        }
    )
    return out


def main() -> None:
    configure_logging()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    records, _, _ = run_rule(BUS39, None, n_jobs=n_jobs)
    records.to_parquet(HARNESS_RECORDS, index=False)
    logger.info(f"BUS39 LOSO: {len(records):,} records, {int(records['covered'].sum()):,} covered")
    out = pd.DataFrame(checks(records, pd.read_parquet(PROD_RECORDS)))
    out.to_csv(OUTPUT_PATH, index=False)
    for row in out.itertuples():
        logger.info(f"{'PASS' if row.passed else 'FAIL'} {row.check}: {row.n_mismatch} of {row.n_compared}")
    logger.info(f"{int(out['passed'].sum())} of {len(out)} checks passed; saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
