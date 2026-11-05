"""Does any alternative representation estimate better than the deployed one on ELES? (analysis 05, Q1, R2.10b)

Pre-specified in the manuscript repository's revision/analyses/05-representations.md, committed before any number here
was computed. Each representation (_representations.NAMES) is run through the validated harness under the exact
line-status key, leave one state out and causal lag 0; the neighbour pool, and so coverage, is the deployed one, and
the comparison against the deployed records is the paired comparison of analysis 03 (eles_uniform_weights.compare),
relabelled. Rule: better or worse only with a difference of at least 0.010 s whose interval, under the widest of state,
topology-group and calendar-day resampling, excludes zero; negligible if it excludes zero below 0.010 s; otherwise no
material difference. No representation replaces the deployed one in the paper.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/eles_representations.py [n_jobs]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _representations  # noqa: E402
from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import (  # noqa: E402
    EXACT_LINES,
    LINE_KEY,
    BruteForceDatabase,
    Rule,
    causal_exclusions,
    load_corpus,
    run_rule,
)
from eles_uniform_weights import compare  # noqa: E402
from referral_decomposition import ELES_CAUSAL0, ELES_DIR, ELES_LOSO, RAW_ZIP, VALIDATION, _eles_clusters  # noqa: E402

from src.config.logging import configure_logging  # noqa: E402
from src.services.qdrant.repository import RE_TOPO  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "eles_representations.csv"
VERDICTS: Final[dict[str, str]] = {
    "decision_kernel_improves": "decision_representation_worse",
    "decision_uniform_improves": "decision_representation_better",
    "decision_negligible": "decision_negligible",
    "decision_no_difference": "decision_no_material_difference",
}


def relabel(out: pd.DataFrame, name: str) -> pd.DataFrame:
    q = out["quantity"].map(lambda s: VERDICTS.get(s, s.replace("uniform", name).replace("kernel", "deployed")))
    return out.assign(quantity=q, representation=name)


def facts() -> list[dict[str, Any]]:
    corpus = load_corpus()
    db = BruteForceDatabase(corpus.lf_scaled, corpus.key_columns, EXACT_LINES)
    raw_columns = [c for c in corpus.lf.columns if not RE_TOPO.match(c)]
    rows = []
    for name in _representations.NAMES:
        _, f = _representations.build(name, db.vectors, db.embed_cols, raw_columns, db.bits[LINE_KEY])
        rows += [{"representation": name, "quantity": k, "estimate": v} for k, v in f.items()]
    return rows


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    state_ids = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl").index.astype(str)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(state_ids)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")

    results = [pd.DataFrame(facts())]
    deployed = {"retrospective": pd.read_parquet(ELES_LOSO), "causal_lag0": pd.read_parquet(ELES_CAUSAL0)}
    units = _eles_clusters(pd.Index(deployed["retrospective"]["state"].drop_duplicates()))
    for name in _representations.NAMES:
        rule = Rule(f"exact_lines_{name}", EXACT_LINES.key, EXACT_LINES.max_hamming, representation=name)
        for availability, exclusions in (("retrospective", None), ("causal_lag0", causal_exclusions(timestamps, 0.0))):
            records, _, _ = run_rule(rule, exclusions, n_jobs=n_jobs)
            tag = "loso" if exclusions is None else "causal0h"
            records.to_parquet(TMP_DIR / f"harness_records_{name}_{tag}.parquet", index=False)
            results.append(relabel(compare(deployed[availability], records, units, availability), name))
            pd.concat(results, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
            d = results[-1][results[-1]["quantity"].str.startswith("decision_")].iloc[0]
            logger.info(
                f"{name} {availability}: MAE {name} - deployed {d['estimate']:+.5f} s "
                f"[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}] under {d['resampling_unit']} -> {d['quantity']}"
            )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
