"""Measures how ELES topology groups are distributed in time.

The manuscript states that ELES's largest topology groups are temporally bursty: they occur
within a one- to three-day window and do not recur over the seven-month record. That claim
carries the frozen-versus-continuous support gap, the framing of the exclusion sweep, and the
conclusion's "met that demand only inside short bursts", but until now it had no measurement
behind it - it lived only in the docstring of the chronological audit.

Two quantities per topology group, using the deployed `lines_only` key:

- calendar span, the time between a group's first and last state. A group whose states all fall
  inside a few days cannot supply neighbors to a query months later.
- burst count, the number of clusters left after splitting the group's sorted timestamps wherever
  consecutive states are more than `GAP_HOURS` apart. A group that recurs has more than one.

`GAP_HOURS` is a reporting choice, not a fitted parameter, so the sensitivity across a grid of
thresholds is written to the CSV alongside the headline figure. States are sampled hourly, so
any threshold above one hour separates genuinely distinct occasions rather than sampling jitter.

Groups are weighted two ways because they answer different questions. Per group answers "how
does a typical configuration behave"; per state answers "what does a typical query face", which
is what retrieval coverage actually depends on, since large groups hold most of the states.

Timestamps come from the raw ZIP through the same mapping the chronological audit uses. Reads
no retrieval artifact and runs no query.

Run from the repository root:
    uv run python scripts/paper/eles_topology_burstiness.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "paper"))

from eles_chronological_topology_support import _load_state_timestamps  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_topology_burstiness.csv"
GROUPS_PATH: Final[Path] = PAPER_DATA_DIR / "eles_topology_burstiness_groups.csv"
VARIANT: Final[str] = "lines_only"
GAP_HOURS: Final[float] = 24.0
GAP_GRID: Final[tuple[float, ...]] = (6.0, 12.0, 24.0, 48.0, 72.0)
# "Largest" in the manuscript's sentence needs a definition. Groups holding at least this many
# states are the ones that can supply a meaningful neighbor set at all; K=100 never binds on
# ELES, so the constraint is always the group, not the neighbor cap.
LARGE_GROUP_MIN_STATES: Final[int] = 10


def _burst_count(times: np.ndarray, gap_hours: float) -> int:
    """Clusters remaining after splitting sorted timestamps at gaps wider than gap_hours."""
    if len(times) < 2:
        return 1
    gaps_h = np.diff(np.sort(times)).astype("timedelta64[m]").astype(float) / 60.0
    return int(1 + (gaps_h > gap_hours).sum())


def main() -> None:
    configure_logging()
    lf = pd.read_pickle(DATASET_DIR / "interim" / "lf.pkl")
    cols = list(joblib.load(DATASET_DIR / "interim" / f"topology_cols_{VARIANT}.joblib.z"))
    group = lf[cols].astype(bool).groupby(cols, sort=False).ngroup()

    ts = _load_state_timestamps(DATASET_DIR / "raw" / "Podatki_DSA.zip")
    df = pd.DataFrame({"group": group.to_numpy(), "state": [str(i) for i in lf.index]})
    df["dt"] = df["state"].map(ts)
    missing = int(df["dt"].isna().sum())
    if missing:
        raise ValueError(f"{missing} states have no timestamp; the state-id mapping is wrong")

    record_days = (df["dt"].max() - df["dt"].min()).total_seconds() / 86400.0
    logger.info(f"{len(df):,} states, {df['group'].nunique():,} {VARIANT} groups, {record_days:.0f}-day record")

    rows = []
    for gid, g in df.groupby("group"):
        t = g["dt"].to_numpy()
        rows.append(
            {
                "group": int(gid),
                "n_states": len(g),
                "span_days": (t.max() - t.min()).astype("timedelta64[m]").astype(float) / 1440.0,
                **{f"bursts_{int(h)}h": _burst_count(t, h) for h in GAP_GRID},
            }
        )
    groups = pd.DataFrame(rows).sort_values("n_states", ascending=False)
    multi = groups[groups["n_states"] >= 2]
    large = groups[groups["n_states"] >= LARGE_GROUP_MIN_STATES]
    b = f"bursts_{int(GAP_HOURS)}h"

    def _share_states(frame: pd.DataFrame, mask: pd.Series) -> float:
        return float(frame.loc[mask, "n_states"].sum() / frame["n_states"].sum())

    logger.info(
        f"multi-state groups: {len(multi):,} holding {multi['n_states'].sum():,} states; "
        f"median span {multi['span_days'].median():.2f} d, "
        f"{100 * (multi['span_days'] <= 3).mean():.1f}% span 3 d or less"
    )
    logger.info(
        f"largest groups (>= {LARGE_GROUP_MIN_STATES} states, n={len(large)}): "
        f"median span {large['span_days'].median():.2f} d, "
        f"single burst at {GAP_HOURS:g} h in {100 * (large[b] == 1).mean():.1f}% of them"
    )

    summary = [
        {"quantity": "n_states", "value": len(df)},
        {"quantity": "n_groups", "value": int(groups["group"].nunique())},
        {"quantity": "record_span_days", "value": record_days},
        {"quantity": "n_singleton_groups", "value": int((groups["n_states"] == 1).sum())},
        {"quantity": "n_multi_state_groups", "value": len(multi)},
        {"quantity": "states_in_multi_state_groups", "value": int(multi["n_states"].sum())},
        {"quantity": "median_span_days_multi", "value": float(multi["span_days"].median())},
        {"quantity": "p90_span_days_multi", "value": float(multi["span_days"].quantile(0.90))},
        {"quantity": "max_span_days_multi", "value": float(multi["span_days"].max())},
        {"quantity": "frac_multi_groups_span_le_1d", "value": float((multi["span_days"] <= 1).mean())},
        {"quantity": "frac_multi_groups_span_le_3d", "value": float((multi["span_days"] <= 3).mean())},
        {"quantity": "frac_multi_states_span_le_3d", "value": _share_states(multi, multi["span_days"] <= 3)},
        {"quantity": "large_group_min_states", "value": LARGE_GROUP_MIN_STATES},
        {"quantity": "n_large_groups", "value": len(large)},
        {"quantity": "states_in_large_groups", "value": int(large["n_states"].sum())},
        {"quantity": "median_span_days_large", "value": float(large["span_days"].median())},
        {"quantity": "max_span_days_large", "value": float(large["span_days"].max())},
        {"quantity": "frac_large_groups_span_le_3d", "value": float((large["span_days"] <= 3).mean())},
        {"quantity": "gap_hours_headline", "value": GAP_HOURS},
    ]
    for h in GAP_GRID:
        col = f"bursts_{int(h)}h"
        summary += [
            {"quantity": f"frac_multi_groups_single_burst_{int(h)}h", "value": float((multi[col] == 1).mean())},
            {"quantity": f"frac_multi_states_single_burst_{int(h)}h", "value": _share_states(multi, multi[col] == 1)},
            {"quantity": f"frac_large_groups_single_burst_{int(h)}h", "value": float((large[col] == 1).mean())},
        ]

    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUTPUT_PATH, index=False)
    groups.to_csv(GROUPS_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH} and {GROUPS_PATH}")


if __name__ == "__main__":
    main()
