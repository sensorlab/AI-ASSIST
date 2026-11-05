"""Validation gate for the exact retrieval harness (_retrieval_harness.py): reproduce production before any new use.

Pre-specified in the manuscript repository's revision/analyses/01-referral-decomposition-and-frontier.md. With the
deployed rule (exact line status, K=100, alpha=1) the harness must reproduce, record for record:

1. leave-one-state-out: the covered set and the selected, screening-minimum and oracle estimates of
   eles_deoracled_bound.py (tmp/eles_deoracled_records_eles-2026-06_lines_only.parquet), and the selected-generator
   diagnostics of eles_generator_diagnostics_selected.py (tmp/eles_generator_diagnostics_selected_*_lines_only.parquet);
2. causal availability at lag 0: the covered set and selected estimate of the causal run
   (tmp/eles_generator_diagnostics_selected_*_lines_only_causal0h.parquet);
3. the ELES screening rows of results/data/clearing_time_threshold_crossing.csv (Table 1).

Estimates must agree within 1e-9 s and every set and count exactly. The two harness runs are kept in tmp/ because they
also carry what the causal run never stored, the causal screening minimum, which the referral decomposition reads.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/retrieval_harness_validation.py [n_jobs]
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

from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import EXACT_LINES, causal_exclusions, run_rule  # noqa: E402
from clearing_time_threshold_crossing import _state_screening_labels, _state_screening_metrics  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
RAW_ZIP: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06" / "raw" / "Podatki_DSA.zip"
LF_PATH: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06" / "interim" / "lf.pkl"

PROD_DEORACLED: Final[Path] = TMP_DIR / "eles_deoracled_records_eles-2026-06_lines_only.parquet"
PROD_SELECTED: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"
PROD_CAUSAL0: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_causal0h.parquet"
PROD_SCREENING: Final[Path] = PAPER_DATA_DIR / "clearing_time_threshold_crossing.csv"

# Harness outputs reused by referral_decomposition.py.
HARNESS_LOSO: Final[Path] = TMP_DIR / "harness_records_exact_lines_loso.parquet"
HARNESS_CAUSAL0: Final[Path] = TMP_DIR / "harness_records_exact_lines_causal0h.parquet"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "retrieval_harness_validation.csv"

TOLERANCE_S: Final[float] = 1e-9
SCREENING_METRICS: Final[tuple[str, ...]] = (
    "n_states",
    "n_unsafe_states",
    "screening_recall",
    "screening_false_alarm_rate",
    "screening_referral_load",
    "referral_threshold_only_load",
    "referral_coverage_only_load",
    "referral_both_load",
    "referral_neither_load",
    "n_states_any_estimate",
    "n_states_complete_coverage",
)


def _with_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "record_ordinal" not in df:
        df["record_ordinal"] = df.groupby("state", sort=False).cumcount()
    return df.set_index(["state", "record_ordinal"])


def _check_equal(name: str, got: pd.Series, expected: pd.Series) -> dict[str, Any]:
    joined = pd.concat([got.rename("got"), expected.rename("expected")], axis=1, join="outer")
    mismatch = int((joined["got"].astype(object) != joined["expected"].astype(object)).sum())
    return {
        "check": name,
        "n_compared": len(joined),
        "n_mismatch": mismatch,
        "max_abs_diff": np.nan,
        "tolerance": 0.0,
        "passed": mismatch == 0,
    }


def _check_close(name: str, got: pd.Series, expected: pd.Series) -> dict[str, Any]:
    joined = pd.concat([got.rename("got"), expected.rename("expected")], axis=1, join="outer")
    both = joined.dropna()
    one_missing = int(joined["got"].isna().ne(joined["expected"].isna()).sum())
    diff = (both["got"].astype(float) - both["expected"].astype(float)).abs()
    max_diff = float(diff.max()) if len(diff) else 0.0
    mismatch = one_missing + int((diff > TOLERANCE_S).sum())
    return {
        "check": name,
        "n_compared": len(joined),
        "n_mismatch": mismatch,
        "max_abs_diff": max_diff,
        "tolerance": TOLERANCE_S,
        "passed": mismatch == 0,
    }


def _screening_rows(records: pd.DataFrame) -> pd.DataFrame:
    df = records.rename(columns={"pred_screening_min": "cct_weighted_per_location"})
    df.loc[~df["covered"], "cct_weighted_per_location"] = np.nan
    rows = []
    for tau in (0.10, 0.15, 0.20):
        metrics = _state_screening_metrics(_state_screening_labels(df, tau))
        rows.append({"tau_s": tau, **{m: metrics[m] for m in SCREENING_METRICS}})
    return pd.DataFrame(rows).set_index("tau_s")


def main() -> None:
    configure_logging()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    loso, loso_states, _ = run_rule(EXACT_LINES, None, n_jobs=n_jobs)
    loso.to_parquet(HARNESS_LOSO, index=False)
    loso_states.to_parquet(HARNESS_LOSO.with_name(HARNESS_LOSO.stem + "_states.parquet"), index=False)
    logger.info(f"LOSO: {len(loso):,} records, {int(loso['covered'].sum()):,} covered")

    state_ids = pd.read_pickle(LF_PATH).index.astype(str)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(state_ids)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")
    causal, causal_states, _ = run_rule(EXACT_LINES, causal_exclusions(timestamps, 0.0), n_jobs=n_jobs)
    causal.to_parquet(HARNESS_CAUSAL0, index=False)
    causal_states.to_parquet(HARNESS_CAUSAL0.with_name(HARNESS_CAUSAL0.stem + "_states.parquet"), index=False)
    logger.info(f"Causal lag 0: {len(causal):,} records, {int(causal['covered'].sum()):,} covered")

    checks: list[dict[str, Any]] = []
    h = _with_ordinal(loso)
    p = _with_ordinal(pd.read_parquet(PROD_DEORACLED))
    checks.append(_check_equal("loso: covered set (deoracled)", h["covered"], p["gen_deoracled_covered"].astype(bool)))
    hc, pc = h[h["covered"]], p[p["gen_deoracled_covered"].astype(bool)]
    checks.append(_check_close("loso: selected estimate", hc["pred_selection"], pc["pred_gen_deoracled_selection"]))
    checks.append(
        _check_close("loso: screening minimum", hc["pred_screening_min"], pc["pred_gen_deoracled_screening_min"])
    )
    checks.append(_check_close("loso: oracle estimate", hc["pred_oracle"], pc["pred_oracle_gen_and_loc"]))
    checks.append(
        _check_equal("loso: true generator rank", hc["gen_true_rank"].astype(int), pc["gen_true_rank"].astype(int))
    )
    checks.append(_check_equal("loso: true CCT", h["cct_true"], p["cct_true"]))

    s = _with_ordinal(pd.read_parquet(PROD_SELECTED))
    sc = s[s["covered"]]
    checks.append(_check_equal("loso: covered set (selected diagnostics)", h["covered"], s["covered"]))
    checks.append(_check_equal("loso: selected generator", hc["gen_selected"], sc["gen_selected"]))
    checks.append(
        _check_close("loso: selected estimate (diagnostics)", hc["pred_selection"], sc["cct_weighted_per_location"])
    )
    checks.append(
        _check_equal("loso: selected neighbour count", hc["sel_n_neighbors"].astype(int), sc["n_neighbors"].astype(int))
    )
    checks.append(_check_close("loso: selected minimum distance", hc["sel_distance_min"], sc["distance_min"]))

    c = _with_ordinal(causal)
    pc0 = _with_ordinal(pd.read_parquet(PROD_CAUSAL0))
    cc, pcc = c[c["covered"]], pc0[pc0["covered"]]
    checks.append(_check_equal("causal0h: covered set", c["covered"], pc0["covered"]))
    checks.append(_check_equal("causal0h: selected generator", cc["gen_selected"], pcc["gen_selected"]))
    checks.append(_check_close("causal0h: selected estimate", cc["pred_selection"], pcc["cct_weighted_per_location"]))

    published = pd.read_csv(PROD_SCREENING)
    published = published[published["dataset"] == "ELES"].set_index("tau_s")
    reproduced = _screening_rows(loso)
    for tau in reproduced.index:
        for metric in SCREENING_METRICS:
            got, expected = float(reproduced.loc[tau, metric]), float(published.loc[tau, metric])
            # Undefined on both sides (recall with no unsafe state, ELES at 100 ms) counts as agreement.
            diff = 0.0 if np.isnan(got) and np.isnan(expected) else abs(got - expected)
            checks.append(
                {
                    "check": f"table 1 ELES tau={tau:g}: {metric}",
                    "n_compared": 1,
                    "n_mismatch": int(diff > 1e-12),
                    "max_abs_diff": diff,
                    "tolerance": 1e-12,
                    "passed": diff <= 1e-12,
                }
            )

    for label, states in (("loso", loso_states), ("causal0h", causal_states)):
        n_err = int(states["query_error"].notna().sum())
        n_fb = int(states["n_fallback_groups"].sum())
        checks.append(
            {
                "check": f"{label}: queries that raise on weight underflow",
                "n_compared": len(states),
                "n_mismatch": n_err,
                "max_abs_diff": np.nan,
                "tolerance": 0.0,
                "passed": n_err == 0,
            }
        )
        checks.append(
            {
                "check": f"{label}: location groups with zero weight mass",
                "n_compared": len(states),
                "n_mismatch": n_fb,
                "max_abs_diff": np.nan,
                "tolerance": 0.0,
                "passed": n_fb == 0,
            }
        )

    out = pd.DataFrame(checks)
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))
    failed = out[~out["passed"]]
    if len(failed):
        logger.error(f"{len(failed)} of {len(out)} checks failed; the harness must not be used until they pass")
        sys.exit(1)
    logger.info(f"All {len(out)} checks passed; saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
