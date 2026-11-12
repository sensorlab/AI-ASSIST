"""How are BUS39 fault records missing, and does the shortfall follow stability? (exploratory analysis 13, R2.14)

The raw TSA table has 59 fault slots per state. Its data dictionary says a CCT that is "too large" (stable at the
2.00 s search limit) is not recorded, and simulations that did not converge were also dropped. Fault locations are
drawn per state and the slots are filled from 0 without gaps, so an absent fault leaves no record of its location:
only the number absent per state (59 minus the records) is known. If absent faults are mostly stable at the limit,
lightly loaded states lose more of them; if they are mostly non-converged runs unrelated to stability, loading does
not predict the count. Aggregate output only.

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
N_SLOTS: Final[int] = 59  # CCT_0 .. CCT_58 in the raw wide table


def main() -> None:
    tsa = pd.read_pickle(DATA_DIR / "tsa.pkl")
    lf = pd.read_pickle(DATA_DIR / "lf.pkl")
    records = tsa.groupby("state").size().reindex(lf.index, fill_value=0)
    assert tsa["experiment"].between(0, N_SLOTS - 1).all()
    contiguous = tsa.groupby("state")["experiment"].max().add(1).reindex(lf.index).eq(records)
    absent = N_SLOTS - records

    # Same proxy as contingency_coverage_audit.py: pre-fault loading, untouched by any simulation outcome.
    loading = lf[[c for c in lf.columns if str(c).startswith("P")]].clip(lower=0).sum(axis=1)
    rho, p = spearmanr(loading, absent)
    by_slot = tsa.groupby("experiment")["Location"].nunique()
    rows = [
        {"quantity": "n_states", "estimate": len(lf)},
        {"quantity": "n_slots_per_state", "estimate": N_SLOTS},
        {"quantity": "n_fault_slots", "estimate": len(lf) * N_SLOTS},
        {"quantity": "n_records", "estimate": len(tsa)},
        {"quantity": "n_absent", "estimate": int(absent.sum())},
        {"quantity": "share_absent", "estimate": float(absent.sum() / (len(lf) * N_SLOTS))},
        {"quantity": "n_states_slots_filled_from_zero", "estimate": int(contiguous.sum())},
        {"quantity": "n_states_all_slots_filled", "estimate": int((absent == 0).sum())},
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
        {"quantity": "spearman_loading_vs_absent", "estimate": float(rho), "p_value": float(p), "n": len(lf)},
        {"quantity": "max_recorded_cct_s", "estimate": float(tsa["CCT"].max())},
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).assign(dataset="BUS39")
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
