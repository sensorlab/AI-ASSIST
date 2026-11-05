"""Compatibility frontier for ELES: coverage, accuracy and screening under alternative compatibility rules (R2.9).

Pre-specified in the manuscript repository's revision/analyses/01-referral-decomposition-and-frontier.md. Every rule uses
the production estimator through the validated exact harness (_retrieval_harness.py; retrieval_harness_validation.py
must have passed), K=100, alpha=1, both leave-one-state-out and causal at lag 0:

    no filter, domestic only (254 Slovenian line flags), Hamming h in {1, 2, 4, 8, 16, 32} over the 907 line flags,
    exact line status (deployed, h = 0), all switching flags (907 line plus 165 generator flags).

No rule is selected; the frontier measures the trade-off. Hamming h counts every differing flag equally. That is
defensible here only because none of the 254 Slovenian flags varies in this window (checked below), so every flag
that can differ belongs to a neighbouring system.

Outputs: results/data/eles_compatibility_frontier.csv (one row per rule and availability) and
results/data/eles_compatibility_mismatch_systems.csv (how often the line that differs between query and retrieved
neighbour belongs to each neighbouring system). ELES data may be published only in aggregate, without naming lines, so
the per-line table stays in tmp/eles_compatibility_mismatch_flags.parquet. Per-run records go to tmp/frontier_<rule>_<availability>.parquet; a run whose parquet exists is reused, so
an interrupted job resumes.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_compatibility_frontier.py [n_jobs]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import EXACT_LINES, Rule, causal_exclusions, run_rule  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
ELES_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
RAW_ZIP: Final[Path] = ELES_DIR / "raw" / "Podatki_DSA.zip"
VALIDATION: Final[Path] = PAPER_DATA_DIR / "retrieval_harness_validation.csv"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_compatibility_frontier.csv"
SYSTEMS_PATH: Final[Path] = PAPER_DATA_DIR / "eles_compatibility_mismatch_systems.csv"
FLAGS_PATH: Final[Path] = TMP_DIR / "eles_compatibility_mismatch_flags.parquet"  # names lines: never publish

RULES: Final[tuple[Rule, ...]] = (
    Rule("no_filter", None),
    Rule("domestic_only", "slovenia_only", 0),
    *(Rule(f"hamming_{h}", "lines_only", h) for h in (32, 16, 8, 4, 2, 1)),
    EXACT_LINES,
    Rule("all_switching", "full", 0),
)
THRESHOLDS: Final[tuple[float, ...]] = (0.15, 0.20)

# Node-name prefix of the neighbouring system in the observability-area model. Not in the ELES data dictionaries:
# I, O and H confirmed by the authors, the rest read from substation names and confirmed by them (2026-11-05).
SYSTEM_PREFIX: Final[dict[str, str]] = {
    "O": "Austria",
    "I": "Italy",
    "H": "Croatia",
    "M": "Hungary",
    "W": "Bosnia and Herzegovina",
    "S": "Switzerland",
    "Q": "Slovakia",
    "R": "Romania",
    "J": "Serbia",
    "D": "Germany",
}


def by_system(flags: pd.DataFrame) -> pd.DataFrame:
    """Per-line mismatch counts aggregated to neighbouring systems, the only form in which they may be published.

    `share_of_mismatches` is the system's part of all query-neighbour line differences under a rule; a neighbour that
    differs in several lines counts once per line. `max_line_share_of_neighbours` is the largest share of retrieved
    neighbours that any single line of the system accounts for, reported without the line's name."""
    keys = ["rule", "availability"]
    g = flags.groupby([*keys, "system"], sort=False)
    out = g.agg(
        n_lines_differing=("flag", "size"),
        n_lines_end_nodes_disagree=("end_nodes_disagree", "sum"),
        n_retrieved_mismatch=("n_retrieved_mismatch", "sum"),
        max_line_share_of_neighbours=("share_of_retrieved_neighbours", "max"),
    ).reset_index()
    total = out.groupby(keys)["n_retrieved_mismatch"].transform("sum")
    out.insert(
        out.columns.get_loc("n_retrieved_mismatch") + 1, "share_of_mismatches", out["n_retrieved_mismatch"] / total
    )
    return out.sort_values([*keys, "n_retrieved_mismatch"], ascending=[True, True, False], ignore_index=True)


def line_system(flag: str) -> tuple[str, bool]:
    """Neighbouring system of a line flag, and whether its two end nodes carry different known prefixes.

    Some first nodes use local names whose first letter is not a system code, while
    the second node is coded, so the second node decides when its prefix is known, then the first, else unassigned."""
    first, _, second = flag.removeprefix("oserv_Lne").partition("-")
    known = [SYSTEM_PREFIX.get(node[:1]) for node in (second, first) if node]
    known = [k for k in known if k]
    return (known[0] if known else "unassigned"), len(set(known)) > 1


def _flags(lf: pd.DataFrame, key: str) -> np.ndarray:
    cols = json.loads((ELES_DIR / "processed" / f"topology_cols_{key}.json").read_text())
    return lf[cols].ne(0).to_numpy()


def _pairwise_hamming(bits: np.ndarray) -> np.ndarray:
    a = bits.astype(np.float32)
    return np.rint(a @ (1 - a).T + (1 - a) @ a.T).astype(np.int32)


def _pool_stats(lf: pd.DataFrame, rule: Rule, earlier: np.ndarray) -> dict[str, Any]:
    """Admissible candidate pools (before K) per query: distinct pools, and mean size retrospective and causal."""
    n = len(lf)
    if rule.key is None:
        admissible = np.ones((n, n), dtype=bool)
    else:
        admissible = _pairwise_hamming(_flags(lf, rule.key)) <= rule.max_hamming
    distinct = len(np.unique(np.packbits(admissible, axis=1), axis=0))
    np.fill_diagonal(admissible, False)
    return {
        "n_distinct_candidate_pools": distinct,
        "mean_candidate_pool_retrospective": float(admissible.sum(axis=1).mean()),
        "mean_candidate_pool_causal": float((admissible & earlier).sum(axis=1).mean()),
    }


def _screening(records: pd.DataFrame, tau: float) -> dict[str, float]:
    est = records["pred_screening_min"].where(records["covered"])
    g = records.assign(_est=est).groupby("state", sort=False)
    unsafe = g["cct_true"].min() < tau
    referred = ~g["covered"].all() | (g["_est"].min() < tau).fillna(False)
    ms = f"{round(tau * 1000)}ms"
    return {
        f"referral_load_{ms}": float(referred.mean()),
        f"recall_{ms}": float((unsafe & referred).sum() / unsafe.sum()) if unsafe.any() else float("nan"),
        f"missed_unsafe_{ms}": int((unsafe & ~referred).sum()),
    }


def _summary(records: pd.DataFrame, states: pd.DataFrame, exact: pd.DataFrame) -> dict[str, Any]:
    cov = records[records["covered"]]
    err = (cov["cct_true"] - cov["pred_selection"]).abs()
    by_state = records.groupby("state", sort=False)["covered"]
    key = ["state", "record_ordinal"]
    common = cov.merge(exact.loc[exact["covered"], [*key, "pred_selection"]], on=key, suffixes=("", "_exact"))
    paired = (common["cct_true"] - common["pred_selection"]).abs() - (
        common["cct_true"] - common["pred_selection_exact"]
    ).abs()
    row: dict[str, Any] = {
        "n_records": len(records),
        "record_coverage": float(records["covered"].mean()),
        "state_coverage_partial": float(by_state.any().mean()),
        "state_coverage_complete": float(by_state.all().mean()),
        "mae": float(err.mean()),
        "abs_error_q50": float(err.quantile(0.50)),
        "abs_error_q95": float(err.quantile(0.95)),
        "abs_error_q99": float(err.quantile(0.99)),
        "abs_error_max": float(err.max()),
        "n_common_with_exact": len(common),
        "paired_mae_minus_exact": float(paired.mean()) if len(common) else float("nan"),
        "generator_top1": float((cov["gen_true_rank"] == 1).mean()),
        "share_neighbours_line_mismatch": float(
            states["n_retrieved_line_mismatch"].sum() / max(states["n_retrieved_states"].sum(), 1)
        ),
        "mean_line_hamming_of_neighbours": float(
            (states["mean_line_hamming"].fillna(0) * states["n_retrieved_states"]).sum()
            / max(states["n_retrieved_states"].sum(), 1)
        ),
        "n_query_errors": int(states["query_error"].notna().sum()),
        "n_zero_mass_groups": int(states["n_fallback_groups"].sum()),
    }
    for tau in THRESHOLDS:
        row.update(_screening(records, tau))
    return row


def main() -> None:
    configure_logging()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness must not be used")

    lf = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl")
    lf.index = lf.index.astype(str)
    n_varying_domestic = int((_flags(lf, "slovenia_only") != _flags(lf, "slovenia_only")[0]).any(axis=0).sum())
    logger.info(f"Slovenian line flags that vary in this window: {n_varying_domestic} of 254")

    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(lf.index)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")
    causal = causal_exclusions(timestamps, 0.0)
    t = timestamps.to_numpy()
    earlier = t[None, :] <= t[:, None]  # usable at lag 0: recorded at or before the query (self removed later)

    runs: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    mismatch_rows: list[pd.DataFrame] = []
    for rule in RULES:
        for availability, exclusions in (("retrospective", None), ("causal_lag0", causal)):
            path = TMP_DIR / f"frontier_{rule.name}_{availability}.parquet"
            states_path = path.with_name(path.stem + "_states.parquet")
            flags_path = path.with_name(path.stem + "_flags.parquet")
            if path.exists() and states_path.exists() and flags_path.exists():
                logger.info(f"{rule.name} {availability}: reusing {path.name}")
                records, states, mismatch = (
                    pd.read_parquet(path),
                    pd.read_parquet(states_path),
                    pd.read_parquet(flags_path)["n_retrieved_mismatch"],
                )
            else:
                logger.info(f"{rule.name} {availability}: running")
                records, states, mismatch = run_rule(rule, exclusions, n_jobs=n_jobs)
                records.to_parquet(path, index=False)
                states.to_parquet(states_path, index=False)
                mismatch.to_frame().to_parquet(flags_path)
            runs[(rule.name, availability)] = (records, states)
            n_neighbours = int(states["n_retrieved_states"].sum())
            mismatch_rows.append(
                pd.DataFrame(
                    {
                        "rule": rule.name,
                        "availability": availability,
                        "flag": mismatch.index,
                        "system": [line_system(f)[0] for f in mismatch.index],
                        "end_nodes_disagree": [line_system(f)[1] for f in mismatch.index],
                        "n_retrieved_mismatch": mismatch.to_numpy(),
                        "n_retrieved_neighbours": n_neighbours,
                        "share_of_retrieved_neighbours": mismatch.to_numpy() / max(n_neighbours, 1),
                    }
                )
            )

    rows = []
    for rule in RULES:
        pools = _pool_stats(lf, rule, earlier)
        for availability in ("retrospective", "causal_lag0"):
            records, states = runs[(rule.name, availability)]
            exact = runs[(EXACT_LINES.name, availability)][0]
            rows.append(
                {
                    "rule": rule.name,
                    "key": rule.key or "none",
                    "max_hamming": rule.max_hamming,
                    "availability": availability,
                    **pools,
                    **_summary(records, states, exact),
                }
            )
    out = pd.DataFrame(rows)

    for availability in ("retrospective", "causal_lag0"):
        a, b = runs[("no_filter", availability)][0], runs[("domestic_only", availability)][0]
        same = a["covered"].equals(b["covered"]) and np.allclose(
            a["pred_selection"].to_numpy(float), b["pred_selection"].to_numpy(float), equal_nan=True, rtol=0, atol=1e-12
        )
        logger.info(f"domestic only equals no filter ({availability}): {same}")
        out.loc[out["availability"] == availability, "domestic_only_equals_no_filter"] = same
    out["n_varying_domestic_flags"] = n_varying_domestic

    out.to_csv(OUTPUT_PATH, index=False)
    flags = pd.concat(mismatch_rows, ignore_index=True)
    flags = flags[flags["n_retrieved_mismatch"] > 0]
    flags.to_parquet(FLAGS_PATH, index=False)
    by_system(flags).to_csv(SYSTEMS_PATH, index=False)
    print(out.to_string(index=False))
    logger.info(f"Saved {OUTPUT_PATH} and {SYSTEMS_PATH}")


if __name__ == "__main__":
    main()
