"""Auditable chronology/support audit for ELES 2026-06's exact-topology filter (ai2ai.md,
2026-08-09/10, Codex review) - design support for the still-open chronological-vs-retrospective
ELES evaluation decision, NOT the chronological evaluation itself. Does not touch any Table 2 or
bootstrap artifact and does not run the retrieval pipeline.

Reads state timestamps from `datasets/eles/2026-06/raw/Podatki_DSA.zip`'s `Dates/` folder into a
TemporaryDirectory (never committed to the repo - the raw ZIP is the only persistent location
for this operational-data timestamp, and this dataset's README documents it as anonymized
operational data, not further-restricted, but there is no reason to leave an extracted copy
lying around either). Joins `Date_main_N.csv` (row -> `<YYYYMMDD>_<HHMM>` timestamp) onto the
`{batch}_{row}` state id via the exact same construction `transform.py::_load_sssa_state_mapping()`
uses, so this reuses a verified mapping convention rather than inventing a new one.

Topology groups are recomputed from each variant's `topology_cols_{variant}.joblib.z` directly
(cross-checked byte-identical against `processed/topology_cols_{variant}.json`, the file the
live service actually reads) via `groupby(all boolean columns).ngroup()`. Covers all three
variants Gregor described choosing between (ai2ai.md, 2026-08-09/10): "lines_only" (the deployed
key: line-status flags only, 907 columns, 1,785 groups), "full" (every switching flag including
generator status, 1,072 columns - fragments almost completely, 4,281 groups), and
"slovenia_only" (a smaller dictionary-matched Slovenian-specific key, 254 columns - collapses to
1 group on this data window, a vacuous non-filter rather than a well-matched coarser key). All
three counts were independently reconciled against Codex's own read of the same files.

For a PRE-SPECIFIED cutoff grid (deciles of elapsed calendar time, 10%-90% - fixed by a rule
independent of the resulting support numbers, not chosen after inspecting them; see Codex's
review, ai2ai.md, on cutoff-selection bias), reports two support definitions per Codex's
distinction - conflating them was the bug in the first version of this analysis:

- "frozen": candidate neighbors are restricted to states strictly before the cutoff (a single,
  static reference library - the deployment model where the reference database does not ingest
  new outcomes after some point).
- "expanding": candidate neighbors may be any state with a strictly earlier timestamp than the
  specific query, regardless of the cutoff (an online/adaptive reference library that keeps
  ingesting outcomes as they arrive). Support is dramatically higher here specifically because
  ELES's largest topology groups are temporally bursty (occur within a 1-3 day window, then do
  not recur in this 7-month dataset) - later states within the same burst can use earlier states
  from that same burst as neighbors, which a frozen pre-cutoff library cannot.

These are two explicit availability SCENARIOS, not a universal lower/upper bound on every
possible deployment (Codex review, ai2ai.md, 2026-08-10): "frozen" models a static pre-cutoff
library, "expanding" models zero-lag continuous ingestion (an optimistic scenario, not
necessarily an upper bound on every batch-refresh policy).

A third scenario, "expanding_20min_lag", is an IDEALIZED PER-CONTINGENCY IMMEDIATE-INSERTION
SCENARIO WITH SUFFICIENT PARALLEL CAPACITY - not a general state-availability model (Gregor +
Codex, ai2ai.md, 2026-08-10, correcting this scenario's original framing). Gregor confirmed the
~20 minutes is per-contingency compute time (a state has up to 172 contingencies, simulated in
parallel but each still taking ~20 minutes), and that reference-library update policy (immediate
insertion vs. periodic batches) is not implemented or decided - there is no single observed
operational cadence to model yet. This scenario assumes every contingency starts at its state's
timestamp and enough parallel workers exist that none queues, so the whole state's records become
available together 20 minutes after that timestamp - the best case, not a demonstrated one. It
also does not model contingency-record-level availability at all: this script only checks
whether an earlier same-topology STATE exists, which is necessary but not sufficient for a given
retrieval to succeed, since the specific (location, critical generator) outcome required must
also exist for that earlier state. A real prospective evaluation needs eligibility checked per
contingency record, not per state/topology-group, and should compare at least three explicitly
hypothetical regimes (idealized immediate insertion, one or more assumed batch cadences, and the
static frozen-library scenario above) - none labeled the deployed or observed workflow. See
ai2ai.md for Codex's full recommended design, not yet run or approved.

Independently verified (not assumed): ELES's states are sampled hourly and no same-topology
predecessor pair (in any of the three variants) is less than 60 minutes apart, so under the
idealization above this scenario happens to produce the same STATE-level support numbers as
zero-lag - this says nothing about contingency-record-level availability, which is not computed
here. The script reports this as an explicit equality check per row, not a runtime assertion - a
future dataset with sub-20-minute sampling would show a genuine, reportable divergence here, not a
program error, and the audit must still write that divergence rather than abort before it can be
seen.

Run from the repository root:
    uv run python scripts/paper/eles_chronological_topology_support.py
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
RAW_ZIP: Final[Path] = DATASET_DIR / "raw" / "Podatki_DSA.zip"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_chronological_topology_support.csv"

# Deciles of elapsed calendar time, fixed before looking at any support number.
CUTOFF_FRACTIONS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# lines_only is the deployed key; full and slovenia_only are fragmentation/no-filter bounds,
# not competing tuned alternatives (ai2ai.md, Codex review, 2026-08-09/10).
VARIANTS: Final[tuple[str, ...]] = ("lines_only", "full", "slovenia_only")
# Author-reported per-contingency compute time (Gregor, ai2ai.md 2026-08-10): ~20 minutes, with
# up to 172 contingencies per state, simulated in parallel. Used here only for an idealized
# immediate-insertion-with-sufficient-parallelism scenario - see module docstring, provisional,
# not the reference-library update policy (not implemented/decided) or an observed cadence.
AUTHOR_REPORTED_LAG: Final[pd.Timedelta] = pd.Timedelta(minutes=20)

_DATE_FILE_RE = re.compile(r"Date_main_(?P<idx>\d+)\.csv")


def _load_state_timestamps(zip_path: Path) -> pd.Series:
    """Same {batch}_{row} state-id construction as transform.py::_load_sssa_state_mapping(),
    inverted (state -> timestamp instead of timestamp -> state). Extracts only the Dates/
    member paths into a TemporaryDirectory, never the whole archive, and the directory is
    removed automatically on exit regardless of success or failure."""
    rows: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if "/Dates/" in m or m.startswith("Dates/")]
            zf.extractall(tmp_dir, members=members)
        for path in sorted(tmp_dir.glob("**/Date_main_*.csv")):
            m = _DATE_FILE_RE.match(path.name)
            if not m:
                continue
            batch = int(m["idx"])
            dates = pd.read_csv(path, sep=";", index_col=0)
            for row, timestamp in dates["DateTime"].items():
                rows.append((f"{batch}_{row}", timestamp))
    state_ts = pd.DataFrame(rows, columns=["state", "timestamp"]).set_index("state")
    return pd.to_datetime(state_ts["timestamp"], format="%Y%m%d_%H%M")


def _topology_group_ids(lf: pd.DataFrame, variant: str) -> pd.Series:
    interim_cols = list(joblib.load(DATASET_DIR / "interim" / f"topology_cols_{variant}.joblib.z"))
    processed_path = DATASET_DIR / "processed" / f"topology_cols_{variant}.json"
    if processed_path.exists():
        import json

        with open(processed_path) as f:
            processed_cols = set(json.load(f))
        if processed_cols != set(interim_cols):
            raise AssertionError(
                f"interim/processed topology_cols_{variant} mismatch: "
                f"{len(interim_cols)} vs {len(processed_cols)} columns - "
                f"the live service and this script would disagree on topology groups."
            )
    bits = lf[interim_cols].astype(bool)
    return bits.groupby(list(bits.columns)).ngroup()


def _has_earlier_same_topology(df: pd.DataFrame, lag: pd.Timedelta | None = None) -> pd.Series:
    """For each row, is there a same-topology-group candidate whose timestamp plus `lag` has
    passed by this row's timestamp - i.e. candidate_dt + lag <= row_dt, availability inclusive
    at exact completion? Two conditions, both against each group's minimum timestamp:
    (1) row_dt > group_min_dt - a strictly-earlier candidate exists at all (raw ordering,
    independent of lag). Not idxmin()-based first-row detection - idxmin() names exactly one row
    per group regardless of ties, so any other row sharing that minimum would be wrongly marked
    "has an earlier state" when it is merely tied, not later (Codex review, ai2ai.md,
    2026-08-10). This condition is also correct for the row that IS the group minimum: comparing
    a row against itself is always False, the right answer (no predecessor exists at all).
    (2) row_dt >= group_min_dt + lag - that candidate's completion-plus-lag has passed
    (inclusive: a candidate completing exactly `lag` before this row is available, not excluded -
    Codex review, ai2ai.md, 2026-08-10, on the earlier strict-only version's off-by-boundary bug).
    Condition (1) alone is what's needed when lag is zero (condition (2) is implied by it)."""
    lag = lag if lag is not None else pd.Timedelta(0)
    group_min_dt = df.groupby("group")["dt"].transform("min")
    return (df["dt"] > group_min_dt) & (df["dt"] >= group_min_dt + lag)


def _variant_rows(lf: pd.DataFrame, dt: pd.Series, variant: str) -> list[dict[str, object]]:
    group_id = _topology_group_ids(lf, variant)
    group_id.index = group_id.index.astype(str)
    n_groups = int(group_id.nunique())
    n_states = len(group_id)
    logger.info(f"[{variant}] Reconciled topology-group count: {n_groups} groups across {n_states} states")

    df = pd.DataFrame({"group": group_id, "dt": dt.reindex(group_id.index)})
    assert df["dt"].notna().all(), "every LF state must have a matched timestamp - check the ZIP/Dates join"

    df["has_earlier_same_topology"] = _has_earlier_same_topology(df)
    df["has_earlier_same_topology_20min_lag"] = _has_earlier_same_topology(df, lag=AUTHOR_REPORTED_LAG)

    # Report divergence, don't assert it away (Codex review, ai2ai.md, 2026-08-10): on a future
    # dataset with sub-lag sampling, the lagged and zero-lag columns legitimately differ, and
    # that is exactly the sensitivity this audit exists to surface - not a condition to abort on.
    lag_equals_zero_lag = bool(df["has_earlier_same_topology"].equals(df["has_earlier_same_topology_20min_lag"]))
    if not lag_equals_zero_lag:
        logger.warning(
            f"[{variant}] 20-minute-lag support diverges from zero-lag support on this data - "
            f"report both numbers, do not assume they match."
        )

    global_expanding_support = float(df["has_earlier_same_topology"].mean())
    logger.info(
        f"[{variant}] Global expanding-library support (fraction of states with >=1 "
        f"strictly-earlier same-topology state, any lag): {global_expanding_support:.4f}"
    )

    start, end = df["dt"].min(), df["dt"].max()
    span = end - start
    rows: list[dict[str, object]] = [
        {"variant": variant, "quantity": "n_topology_groups", "value": n_groups},
        {"variant": variant, "quantity": "n_states", "value": n_states},
        {"variant": variant, "quantity": "global_expanding_support_fraction", "value": global_expanding_support},
        {
            "variant": variant,
            "quantity": "global_expanding_support_fraction_20min_lag_equals_zero_lag",
            "value": lag_equals_zero_lag,
        },
        {"variant": variant, "quantity": "date_range_start", "value": str(start)},
        {"variant": variant, "quantity": "date_range_end", "value": str(end)},
    ]

    for frac in CUTOFF_FRACTIONS:
        cutoff = start + span * frac
        post = df[df["dt"] >= cutoff]
        pre_groups = set(df[df["dt"] < cutoff]["group"])
        frozen_support = float(post["group"].isin(pre_groups).mean()) if len(post) else float("nan")
        expanding_support = float(post["has_earlier_same_topology"].mean()) if len(post) else float("nan")
        expanding_support_lagged = (
            float(post["has_earlier_same_topology_20min_lag"].mean()) if len(post) else float("nan")
        )
        rows.append(
            {
                "variant": variant,
                "quantity": "cutoff_support",
                "cutoff_fraction": frac,
                "cutoff_date": str(cutoff),
                "n_post_cutoff_queries": len(post),
                "frozen_library_support_fraction": frozen_support,
                "expanding_library_support_fraction": expanding_support,
                "expanding_library_support_fraction_20min_lag": expanding_support_lagged,
            }
        )
        logger.info(
            f"[{variant}] cutoff_fraction={frac:.1f} ({cutoff.date()}) n_post={len(post):,} "
            f"frozen={frozen_support:.4f} expanding={expanding_support:.4f} "
            f"expanding_20min_lag={expanding_support_lagged:.4f}"
        )
    return rows


def main() -> None:
    configure_logging()
    dt = _load_state_timestamps(RAW_ZIP)
    lf = pd.read_pickle(DATASET_DIR / "interim" / "lf.pkl")

    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows.extend(_variant_rows(lf, dt, variant))

    out = pd.DataFrame(rows)
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
