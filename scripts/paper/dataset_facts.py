"""Archive facts quoted in the manuscript's Methods (Datasets) and Supplementary Notes S1 and S4.

Counts only, no element names, in keeping with the ELES data agreement: feature columns per family, switching flags
by kind, how often generator status changes, how many states share a configuration, how many continuous columns the
key-locality control uses, and the share of training records the three-hour temporal block removes per fold.

Inputs: datasets/{bus39,eles/2026-06}/ (prepared), the ELES raw archive for timestamps, the Slovenian PowerFactory
dictionary, and tmp/ml_benchmark_predictions-eles-2026-06.parquet for the five state-group folds.
Output: results/data/dataset_facts.csv.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/dataset_facts.py
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from _representations import TYPE_RE, dictionary_names  # noqa: E402
from ml_benchmark import _temporally_blocked_train_idx  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
ELES_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
RAW_ZIP: Final[Path] = ELES_DIR / "raw" / "Podatki_DSA.zip"
FOLDS: Final[Path] = PROJECT_DIR / "tmp" / "ml_benchmark_predictions-eles-2026-06.parquet"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "dataset_facts.csv"
FAMILY_RE: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z0-9]+?)_")
BLOCK_WINDOW: Final[pd.Timedelta] = pd.Timedelta(hours=3)


def flag_changes(bits: np.ndarray) -> int:
    """Number of flag flips between consecutive rows (states in time order)."""
    return int((bits[1:] != bits[:-1]).sum())


def share_sharing_a_key(keys: pd.Series) -> float:
    """Share of states whose key value occurs in at least one other state."""
    return float(keys.duplicated(keep=False).mean())


def _rows(dataset: str, lf: pd.DataFrame, n_records: int) -> list[dict]:
    rows = [
        {"dataset": dataset, "quantity": "n_states", "value": len(lf)},
        {"dataset": dataset, "quantity": "n_records", "value": n_records},
        {"dataset": dataset, "quantity": "records_per_state_mean", "value": n_records / len(lf)},
    ]
    families = pd.Series([m.group(1) if (m := FAMILY_RE.match(c)) else c for c in lf.columns]).value_counts()
    rows += [{"dataset": dataset, "quantity": f"n_columns_{f}", "value": int(n)} for f, n in families.items()]
    return rows


def _n_records(dataset_dir: Path) -> int:
    with sqlite3.connect(dataset_dir / "processed" / "tsa.db") as conn:
        return int(conn.execute("SELECT COUNT(*) FROM tsa").fetchone()[0])


def main() -> None:
    configure_logging()
    bus = pd.read_pickle(PROJECT_DIR / "datasets" / "bus39" / "interim" / "lf.pkl")
    rows = _rows("BUS39", bus, _n_records(PROJECT_DIR / "datasets" / "bus39"))

    lf = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl")
    rows += _rows("ELES", lf, _n_records(ELES_DIR))

    def cols(key: str) -> list[str]:
        return [c for c in json.loads((ELES_DIR / "processed" / f"topology_cols_{key}.json").read_text()) if c in lf]

    lines, full, si = cols("lines_only"), cols("full"), set(cols("slovenia_only"))
    gens = [c for c in full if c not in set(lines)]
    varying = lambda cs: [c for c in cs if lf[c].nunique(dropna=False) > 1]  # noqa: E731
    names = dictionary_names()
    si_gens = [c for c in gens if (el := c.removeprefix("oserv_")) in names or TYPE_RE.sub("", el) in names]

    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(lf.index.astype(str))
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} ELES states have no timestamp")
    ordered = lf.iloc[np.argsort(timestamps.to_numpy(), kind="stable")]
    gen_bits = ordered[gens].ne(0).to_numpy()

    key = lambda cs: lf[cs].ne(0).astype(np.int8).astype(str).agg("".join, axis=1)  # noqa: E731
    cont = lf.drop(columns=[c for c in lf.columns if c.startswith("oserv_")]).select_dtypes(include=[np.number])
    cont = cont.loc[:, cont.notna().all() & (cont.std(numeric_only=True) > 0)]  # as eles_topology_key_locality.py

    rows += [
        {"dataset": "ELES", "quantity": q, "value": v}
        for q, v in [
            ("n_switching_flags", len(full)),
            ("n_line_flags", len(lines)),
            ("n_line_flags_varying", len(varying(lines))),
            ("n_slovenian_line_flags", len(si)),
            ("n_slovenian_line_flags_varying", len(varying([c for c in lines if c in si]))),
            ("n_generator_flags", len(gens)),
            ("n_generator_flags_varying", len(varying(gens))),
            ("n_slovenian_generator_flags", len(si_gens)),
            ("n_slovenian_generator_flags_varying", len(varying(si_gens))),
            ("n_time_order_transitions", len(ordered) - 1),
            ("n_generator_flag_changes", flag_changes(gen_bits)),
            ("n_generator_status_configurations", key(gens).nunique()),
            ("n_line_status_groups", key(lines).nunique()),
            ("n_all_switching_groups", key(full).nunique()),
            ("share_states_sharing_all_switching_group", share_sharing_a_key(key(full))),
            ("n_complete_continuous_columns", cont.shape[1]),
        ]
    ]

    folds = pd.read_parquet(FOLDS, columns=["state", "record_ordinal", "fold"]).drop_duplicates(
        ["state", "record_ordinal"]
    )
    groups = folds["state"].astype(str).reset_index(drop=True)
    times = timestamps.copy()
    times.index = times.index.astype(str)
    for f in sorted(folds["fold"].unique()):
        test_idx = np.flatnonzero(folds["fold"].to_numpy() == f)
        train_idx = np.flatnonzero(folds["fold"].to_numpy() != f)
        kept = _temporally_blocked_train_idx(train_idx, test_idx, groups, times, BLOCK_WINDOW)
        rows.append(
            {
                "dataset": "ELES",
                "quantity": f"temporal_block_3h_train_share_removed_fold{int(f)}",
                "value": 1 - len(kept) / len(train_idx),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    for r in out.itertuples():
        logger.info(
            f"{r.dataset} {r.quantity}: {r.value:,.4f}"
            if isinstance(r.value, float)
            else f"{r.dataset} {r.quantity}: {r.value:,}"
        )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
