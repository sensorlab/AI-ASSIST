"""Check whether SSSA's per-state generator coverage is stable enough to use as a
query-time matching key, analogous to the topology bitstring exact-match filter
(see scripts/evaluation/eles_topology_candidate_eval.py and datasets/eles/2026-06/README.md's
"Topology Variants" section).

Investigation trigger: EstimationService.estimate_sssa_by_generator() groups retrieved SSSA
rows by generator (mode_id is a per-state local identifier, never comparable across states -
see datasets/eles/2026-01/README.md and datasets/eles/2026-06/README.md's SSSA sections), but
it's not yet exposed via the API, and the domain partners haven't specified what a live SSSA
query should actually match on. Before designing that beta endpoint, we need to know: does
every state's SSSA data cover the same set of generators, or does that set vary from state to
state? If it varies a lot, grouping by generator at query time may produce very uneven
coverage across retrieved neighbors.

This computes, per state, the exact set of generators with at least one SSSA participation
row, then groups states by that set - strict equality only. {"G1","G2","G3"} is a different
group from {"G1","G2","G3","G4"} even though they share 3 generators; no subset/overlap credit
is given here (that would be a materially different, more permissive matching definition).
Mirrors the topology-candidate table's shape: unique groups, % of states with >=1 same-set
"neighbor", max group size.

Pure pandas over already-parsed interim artifacts - no Qdrant, no scaler, no EstimationService
involved (unlike the topology-candidate script, which needed those to test retrieval).

Run from repository root:
    uv run python scripts/evaluation/eles_sssa_generator_set_eval.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _resolve_dataset_file

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
# Evaluation artifacts don't belong at the repo root; SSSA is out of paper-sr's TSA-only scope
# (DIRECTION.md Decisions), so both outputs go to tmp/ rather than paper-sr/data/
# (2026-08-05 cleanup).
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH: Final[Path] = TMP_DIR / "report-eles-sssa-generator-sets.joblib"
SUMMARY_CSV_PATH: Final[Path] = TMP_DIR / "sssa_generator_set_groups.csv"

DATASETS: Final[tuple[str, ...]] = ("eles/2026-01", "eles/2026-06")


def _analyze_dataset(data_dir: Path, dataset_name: str) -> tuple[dict[str, object], pd.Series]:
    interim = data_dir / dataset_name / "interim"

    lf_path = _resolve_dataset_file(interim, "lf.pkl")
    sssa_path = _resolve_dataset_file(interim, "sssa.pkl")

    lf = pd.read_pickle(lf_path)
    sssa = pd.read_pickle(sssa_path)

    n_states_total = len(lf)

    per_state_generators = sssa.groupby("state")["generator"].agg(lambda s: frozenset(s.unique()))

    # Strict-equality group key - a sorted-tuple signature of the full generator set, mirroring
    # _get_topology_id()'s bitstring-as-group-key idea. Two states only share a group if this
    # signature is identical outright; no partial-overlap credit.
    generator_set_id = per_state_generators.map(lambda s: tuple(sorted(s)))
    group_sizes = generator_set_id.value_counts()

    n_states_with_sssa = len(per_state_generators)
    set_sizes = per_state_generators.map(len)
    states_with_match = group_sizes[group_sizes > 1].sum()

    summary = {
        "dataset": dataset_name,
        "n_states_total": n_states_total,
        "n_states_with_sssa": n_states_with_sssa,
        "n_distinct_generator_sets": len(group_sizes),
        "max_group_size": int(group_sizes.max()),
        "pct_states_with_exact_match": states_with_match / n_states_with_sssa * 100,
        "mean_generator_set_size": set_sizes.mean(),
        "median_generator_set_size": set_sizes.median(),
    }
    return summary, per_state_generators


def main() -> None:
    configure_logging()
    app_settings = get_app_settings()

    summaries: list[dict[str, object]] = []
    per_state_by_dataset: dict[str, pd.Series] = {}

    for dataset_name in DATASETS:
        logger.info(f"Analyzing {dataset_name}...")
        t0 = time.monotonic()
        summary, per_state_generators = _analyze_dataset(app_settings.data_dir, dataset_name)
        logger.info(f"{dataset_name} done in {time.monotonic() - t0:.1f}s: {summary}")
        summaries.append(summary)
        per_state_by_dataset[dataset_name] = per_state_generators

    joblib.dump(per_state_by_dataset, REPORT_PATH)
    logger.info(f"Saved raw per-state generator sets to {REPORT_PATH}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    logger.info(f"Saved summary CSV to {SUMMARY_CSV_PATH}")

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
