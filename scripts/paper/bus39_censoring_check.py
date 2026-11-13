"""How are BUS39 fault records structured, and does their number follow stability? (exploratory analysis 13, R2.14)

Fault locations are not a fixed list: for each state, faults are simulated in sequence from the generator terminals to
nearby elements, each bus and then the lines leaving it, until the CCT exceeds a predefined threshold, so the number of
faults differs between states. The raw
TSA table is as wide as the largest state (59 faults). A fault stable at the 2.00 s search limit ("too large" in the
data dictionary) or whose simulation did not converge is not recorded, and the output cannot tell either from a
location the search did not reach. If the search omits faults stable at the limit, lightly loaded states record fewer
faults. Aggregate output only.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/bus39_censoring_check.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_DIR / "datasets" / "bus39" / "interim"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "bus39_censoring_check.csv"
N_SLOTS: Final[int] = 59  # CCT_0 .. CCT_58: width of the raw wide table, the largest number of faults in any state


def main() -> None:
    tsa = pd.read_pickle(DATA_DIR / "tsa.pkl")
    lf = pd.read_pickle(DATA_DIR / "lf.pkl")
    records = tsa.groupby("state").size().reindex(lf.index, fill_value=0)
    assert tsa["experiment"].between(0, N_SLOTS - 1).all()
    contiguous = tsa.groupby("state")["experiment"].max().add(1).reindex(lf.index).eq(records)

    # Same proxy as contingency_coverage_audit.py: pre-fault loading, untouched by any simulation outcome.
    loading = lf[[c for c in lf.columns if str(c).startswith("P")]].clip(lower=0).sum(axis=1)
    rho, p = spearmanr(loading, records)
    by_slot = tsa.groupby("experiment")["Location"].nunique()
    rows = [
        {"quantity": "n_states", "estimate": len(lf)},
        {"quantity": "n_records", "estimate": len(tsa)},
        {"quantity": "n_states_slots_filled_from_zero", "estimate": int(contiguous.sum())},
        {"quantity": "n_states_with_max_records", "estimate": int((records == N_SLOTS).sum())},
        {"quantity": "n_states_without_records", "estimate": int((records == 0).sum())},
        {"quantity": "records_per_state_min", "estimate": int(records.min())},
        {"quantity": "n_elements", "estimate": int(tsa["Location"].nunique())},
        {"quantity": "n_element_terminal_pairs", "estimate": int(tsa.groupby(["Location", "Terminal"]).ngroups)},
        {"quantity": "elements_per_slot_min", "estimate": int(by_slot.min())},
        {"quantity": "elements_per_slot_max", "estimate": int(by_slot.max())},
        {
            "quantity": "n_states_repeating_an_element",
            "estimate": int(tsa.duplicated(["state", "Location"]).groupby(tsa["state"]).any().sum()),
        },
        {"quantity": "spearman_loading_vs_records", "estimate": float(rho), "p_value": float(p), "n": len(lf)},
        {"quantity": "max_recorded_cct_s", "estimate": float(tsa["CCT"].max())},
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).assign(dataset="BUS39")
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
