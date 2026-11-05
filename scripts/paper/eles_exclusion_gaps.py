"""Do the excluded ELES states fall at topology changes? (R2.3a, exploratory, without the campaign failure log)

The analysed states occupy 4,393 of the 5,136 hourly slots between the first and last analysed timestamp, and the 743
empty slots equal the number of excluded states, so the gaps in the hourly series are where the excluded states fell.
If failures cluster at topology changes, the line-status key should differ across a gap more often than across an
equally long stretch with no gap: the comparison is matched on span, because two states further apart differ more
often whatever the cause.

- Gap pair: two consecutive analysed states with at least one empty slot between them (span s >= 2 hours).
- Control pair: two analysed states s hours apart with every slot between them analysed.
- Expected gap rate: control rates per span, weighted by the number of gap pairs at that span. Spans with fewer than
  MIN_CONTROL control pairs are left out of the comparison and counted.

Reported: share of pairs whose line-status key differs and mean number of differing line flags, for gap pairs against
their matched expectation, with a calendar-day cluster bootstrap (by the day of the pair's first state) on the
difference. Output is aggregate only: no timestamps, no lines.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_exclusion_gaps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from referral_decomposition import ELES_DIR, RAW_ZIP  # noqa: E402

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "eles_exclusion_gaps.csv"
N_ANALYSED: Final[int] = 4393
N_EXCLUDED: Final[int] = 743
MIN_CONTROL: Final[int] = 20
N_BOOTSTRAP: Final[int] = 2000
SEED: Final[int] = 42


def pairs(hours: np.ndarray, flags: np.ndarray) -> pd.DataFrame:
    """Gap pairs and span-matched control pairs. hours: sorted integer slot index per state; flags: bool matrix."""
    present = np.zeros(hours[-1] + 1, bool)
    present[hours] = True
    pos = np.full(len(present), -1)
    pos[hours] = np.arange(len(hours))
    # Length of the run of present slots ending at each slot, to test "every slot between is present".
    run = np.zeros(len(present), int)
    for i, p in enumerate(present):
        run[i] = run[i - 1] + 1 if p and i else int(p)
    rows = []
    gap_spans = np.diff(hours)
    for i, s in enumerate(gap_spans):
        if s >= 2:
            rows.append((i, i + 1, int(s), "gap"))
    for s in np.unique(gap_spans[gap_spans >= 2]):
        ends = np.flatnonzero(run >= s + 1)  # slots t + s whose run covers t .. t + s
        for end in ends:
            rows.append((pos[end - s], pos[end], int(s), "control"))
    df = pd.DataFrame(rows, columns=["a", "b", "span", "kind"])
    diff = flags[df["a"].to_numpy()] != flags[df["b"].to_numpy()]
    df["n_differing"] = diff.sum(axis=1)
    df["key_differs"] = df["n_differing"] > 0
    return df


def matched(df: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    """Gap rates against span-matched control expectations, optionally reweighting pairs."""
    w = np.ones(len(df)) if weights is None else weights
    d = df.assign(w=w)
    out: dict[str, float] = {}
    for metric in ("key_differs", "n_differing"):
        g = d.assign(x=d["w"] * d[metric]).groupby(["kind", "span"])[["x", "w"]].sum()
        gap, ctrl = g.loc["gap"], g.loc["control"]
        gap = gap[gap["w"] > 0]
        rate_ctrl = (ctrl["x"] / ctrl["w"]).reindex(gap.index)
        observed = gap["x"].sum() / gap["w"].sum()
        # A resample can lose every control pair at a span; that resample has no matched expectation.
        expected = np.nan if rate_ctrl.isna().any() else float((rate_ctrl * gap["w"]).sum() / gap["w"].sum())
        out[f"{metric}_gap"] = float(observed)
        out[f"{metric}_matched_control"] = expected
        out[f"{metric}_difference"] = float(observed - expected)
    return out


def main() -> None:
    lf = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl")
    lf.index = lf.index.astype(str)
    cols = json.loads((ELES_DIR / "processed" / "topology_cols_lines_only.json").read_text())
    ts = load_eles_state_timestamps(RAW_ZIP)
    states = pd.read_parquet(PROJECT_DIR / "tmp" / "harness_records_exact_lines_loso_states.parquet")["state"]
    ts = ts.reindex(states).sort_values()
    if ts.isna().any() or len(ts) != N_ANALYSED:
        raise ValueError("analysed states and timestamps do not match")
    hours_f = (ts - ts.iloc[0]) / pd.Timedelta(hours=1)
    if not np.allclose(hours_f, hours_f.round()):
        raise ValueError("analysed states are not on an hourly grid")
    hours = hours_f.round().astype(int).to_numpy()
    if len(np.unique(hours)) != len(hours) or hours[-1] + 1 - len(hours) != N_EXCLUDED:
        raise ValueError("empty hourly slots do not equal the number of excluded states")
    flags = lf.loc[ts.index, cols].ne(0).to_numpy()

    df = pairs(hours, flags)
    n_ctrl = df[df["kind"] == "control"].groupby("span").size()
    keep = n_ctrl[n_ctrl >= MIN_CONTROL].index
    gaps = df[df["kind"] == "gap"]
    dropped = gaps[~gaps["span"].isin(keep)]
    df = df[df["span"].isin(keep)].reset_index(drop=True)
    point = matched(df)

    day = pd.factorize(ts.dt.floor("D").to_numpy()[df["a"].to_numpy()])[0]
    rng = np.random.default_rng(SEED)
    n_days = day.max() + 1
    boot = []
    for _ in range(N_BOOTSTRAP):
        w = np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)[day].astype(float)
        boot.append(matched(df, w))
    boot_df = pd.DataFrame(boot)

    spans = np.diff(hours)
    rows = [
        {"quantity": "n_hourly_slots", "estimate": hours[-1] + 1},
        {"quantity": "n_empty_slots", "estimate": N_EXCLUDED},
        {"quantity": "n_gaps", "estimate": int((spans >= 2).sum())},
        {"quantity": "n_single_slot_gaps", "estimate": int((spans == 2).sum())},
        {"quantity": "longest_gap_slots", "estimate": int(spans.max() - 1)},
        {"quantity": "n_gap_pairs_compared", "estimate": int((df["kind"] == "gap").sum())},
        {"quantity": "n_gap_pairs_unmatched", "estimate": len(dropped)},
        {"quantity": "n_empty_slots_in_unmatched_gaps", "estimate": int((dropped["span"] - 1).sum())},
        {"quantity": "n_control_pairs", "estimate": int((df["kind"] == "control").sum())},
        {
            "quantity": "key_differs_consecutive_hours",
            "estimate": float((flags[1:] != flags[:-1]).any(axis=1)[spans == 1].mean()),
        },
    ]
    for name, value in point.items():
        lo, hi = np.percentile(boot_df[name].dropna(), [2.5, 97.5])
        rows.append(
            {"quantity": name, "estimate": value, "ci_low": lo, "ci_high": hi, "resampling_unit": "calendar_day"}
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
