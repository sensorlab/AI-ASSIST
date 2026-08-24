"""Why each state carries fewer contingencies than the campaign's maximum (ISSUES.md SIM-8).

BUS39 records 47.96 of "up to 59" contingencies per state and ELES 73.44 of "up to 172".
The concern this audit answers is specific: if the absent runs are simulation failures
concentrated on severe cases, every reported error metric is optimistic, because the hard
cases would be missing from both the archive and the evaluation.

The obvious test - compare observed CCT between states that keep a contingency and states
that drop it - is circular, since the observed CCT of a state is computed from exactly the
records whose selection is in question. This audit therefore ranks state severity by a
pre-fault covariate that no simulation outcome can influence: total positive active power
across the load-flow columns, i.e. how heavily the system is loaded before any fault.

Two Spearman correlations settle it. The first validates the proxy (loading against observed
mean CCT: heavier loading must mean lower CCT). The second is the answer (loading against the
number of contingencies recorded). A positive second correlation means heavily loaded, severe
states carry more records rather than fewer, so the shortfall sits with mild states and the
feared loss of hard cases did not occur.

Run from the repository root:

    nice -n 19 uv run python scripts/paper/contingency_coverage_audit.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.logging import configure_logging

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "contingency_coverage_audit.csv"

DATASETS: Final[tuple[str, ...]] = ("bus39", "eles/2026-06")
CONTINGENCY_KEY: Final[list[str]] = ["Location", "Terminal", "Type"]


def _audit(dataset: str) -> dict[str, float | int | str]:
    tsa = pd.read_pickle(PROJECT_DIR / "datasets" / dataset / "interim" / "tsa.pkl")
    lf = pd.read_pickle(PROJECT_DIR / "datasets" / dataset / "interim" / "lf.pkl")
    lf.index = lf.index.astype(str)

    tsa["cid"] = tsa[CONTINGENCY_KEY].astype(str).agg("|".join, axis=1)
    # Records per state, which is what the manuscript's "47.96 of up to 59" and "73.49 of up
    # to 172" count. On BUS39 that equals the distinct-contingency count; on ELES a handful of
    # states repeat a (Location, Terminal, Type) triple, so the two differ slightly and both
    # are reported below.
    n_contingencies = tsa.groupby("state").size()
    n_distinct_per_state = tsa.groupby("state")["cid"].nunique()
    n_distinct_per_state.index = n_distinct_per_state.index.astype(str)
    observed_cct = tsa.groupby("state")["CCT"].mean()
    n_contingencies.index = n_contingencies.index.astype(str)
    observed_cct.index = observed_cct.index.astype(str)

    states = lf.index.intersection(n_contingencies.index)
    # Pre-fault loading: settled before any fault is applied, so no simulation outcome,
    # failure or truncation can move it. Negative entries are generation, hence the clip.
    active_power = lf.loc[states, [c for c in lf.columns if str(c).startswith("P")]]
    loading = active_power.clip(lower=0).sum(axis=1)

    proxy_rho, proxy_p = spearmanr(loading, observed_cct[states])
    answer_rho, answer_p = spearmanr(loading, n_contingencies[states])
    grid = tsa["state"].nunique() * tsa["cid"].nunique()

    return {
        "dataset": dataset,
        "n_states": int(tsa["state"].nunique()),
        "n_distinct_contingencies": int(tsa["cid"].nunique()),
        "n_records": int(len(tsa)),
        "contingencies_per_state_mean": float(n_contingencies.mean()),
        "contingencies_per_state_min": int(n_contingencies.min()),
        "contingencies_per_state_max": int(n_contingencies.max()),
        "distinct_contingencies_per_state_mean": float(n_distinct_per_state.mean()),
        "distinct_contingencies_per_state_max": int(n_distinct_per_state.max()),
        "grid_fill_rate": float(len(tsa) / grid),
        "rho_loading_vs_observed_cct": float(proxy_rho),
        "p_loading_vs_observed_cct": float(proxy_p),
        "rho_loading_vs_n_contingencies": float(answer_rho),
        "p_loading_vs_n_contingencies": float(answer_p),
        "severe_states_carry_more_records": bool(answer_rho > 0),
    }


def main() -> None:
    configure_logging()
    rows = []
    for dataset in DATASETS:
        row = _audit(dataset)
        rows.append(row)
        logger.info(
            f"{dataset}: {row['contingencies_per_state_mean']:.2f} contingencies/state, "
            f"rho(loading, n_contingencies)={row['rho_loading_vs_n_contingencies']:+.4f}"
        )
    summary = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)
    logger.info(f"Saved {OUT_PATH}")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
