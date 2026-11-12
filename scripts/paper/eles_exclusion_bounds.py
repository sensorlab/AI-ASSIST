"""Bounds on ELES coverage and abstention over every attempted state, not only the analysed ones (R2.3b).

The source campaign attempted 5,136 operating states; 4,393 are analysed, and the other 743 could not be assessed
because the initial load flow did not converge (Methods). The excluded states have no features, so retrieval
cannot be run for them, and their effect can only be bounded:

- all_uncovered: every excluded state lacks compatible support for every contingency, so it abstains and is referred.
  This is a strict lower bound on coverage and an upper bound on abstention and referral load: adding states to the
  archive can only add candidates for the analysed states, never remove them.
- observed_rate: the excluded states are covered and referred at the analysed rates. A reference point, not a bound.

Reported per availability (retrospective, causal lag 0) at the paper's thresholds.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_exclusion_bounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from referral_decomposition import ELES_CAUSAL0, ELES_LOSO, VALIDATION, _state_frame  # noqa: E402

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "eles_exclusion_bounds.csv"

# Campaign totals as reported in Methods (ELES data description); the excluded states are not in any delivered table.
N_ATTEMPTED: Final[int] = 5136
N_ANALYSED: Final[int] = 4393
THRESHOLDS: Final[tuple[float, ...]] = (0.15, 0.20)


def bounds(states: pd.DataFrame, n_attempted: int) -> list[dict[str, object]]:
    """states: one row per analysed state with boolean columns partial, complete and referred_<tau>."""
    n = len(states)
    n_excluded = n_attempted - n
    if n_excluded < 0:
        raise ValueError("more analysed states than attempted")
    rows = []
    for col in states.columns:
        observed = float(states[col].mean())
        # An excluded state is uncovered, so it is neither partially nor completely covered, and it is referred.
        worst = float((states[col].sum() + (0 if col in ("partial", "complete") else n_excluded)) / n_attempted)
        rows.append({"quantity": col, "observed_rate": observed, "all_uncovered": worst})
    return rows


def main() -> None:
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    out = []
    for availability, path in (("retrospective", ELES_LOSO), ("causal_lag0", ELES_CAUSAL0)):
        records = pd.read_parquet(path)
        g = records.groupby("state", sort=False)["covered"]
        states = pd.DataFrame({"partial": g.any(), "complete": g.all()})
        for tau in THRESHOLDS:
            states[f"referred_{round(1000 * tau)}ms"] = _state_frame(records, "pred_screening_min", "covered", tau)[
                "referred"
            ]
        states["abstention"] = ~states["complete"]
        if len(states) != N_ANALYSED:
            raise ValueError(f"{len(states)} analysed states, expected {N_ANALYSED}")
        for row in bounds(states, N_ATTEMPTED):
            out.append({"availability": availability, "n_analysed": N_ANALYSED, "n_attempted": N_ATTEMPTED, **row})
    pd.DataFrame(out).to_csv(OUTPUT_PATH, index=False)
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
