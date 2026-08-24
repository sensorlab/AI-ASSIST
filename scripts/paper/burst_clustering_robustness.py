"""Re-tests the inferential claims whose margins sit closest to zero under a coarser cluster.

Every interval in the manuscript resamples pre-fault states. The manuscript also shows that
ELES states sharing a switching configuration arrive in short bursts - 1,785 topology groups,
1,049 of them singletons, median span three hours, 99.3% forming a single cluster within 24
hours - and states in a burst retrieve each other and share nearly all of their reference
support. Their errors are therefore not independent, and resampling them as though they were
makes every interval narrower than it should be.

The correct unit is the topology group, which is defined by the compatibility key rather than
by anything about the outcomes, so it cannot be tuned to preserve or destroy significance. This
script recomputes only the three claims whose interpretation depends on significance and whose
published intervals sit close to zero:

  A  retrieval against gradient boosting on matched records (the 9.4% ELES lead)
  B  the causal-availability penalty on causally covered records
  C  the topology-filter ablation, per seed

Everything else in the paper is either a count, a coverage share, or a margin far from zero, and
is left alone deliberately: rerunning unrelated intervals would invite reading a robustness check
as a new result.

Run from the repository root:
    uv run python scripts/paper/burst_clustering_robustness.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
DATASET_DIR: Final[Path] = PROJECT_DIR / "datasets" / "eles" / "2026-06"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "burst_clustering_robustness.csv"
RETRIEVAL: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_group_kfold.parquet"
BASELINE: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"
CAUSAL: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only_causal0h.parquet"
SUPERVISED: Final[Path] = TMP_DIR / "ml_benchmark_predictions-eles-2026-06.parquet"
ABLATION_SEEDS: Final[tuple[str, ...]] = ("", "-seed123", "-seed7")
N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42


def _group_map() -> dict[str, int]:
    """State id -> topology group under the deployed line-status key."""
    lf = pd.read_pickle(DATASET_DIR / "interim/lf.pkl")
    lines = [c for c in json.loads((DATASET_DIR / "processed/topology_cols_lines_only.json").read_text()) if c in lf]
    codes = pd.factorize(lf[lines].astype(str).agg("|".join, axis=1))[0]
    return dict(zip((str(i) for i in lf.index), codes, strict=True))


def _ci(diffs: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI on the mean paired difference, resampling whole clusters."""
    codes = pd.factorize(clusters, sort=False)[0]
    order = np.argsort(codes, kind="stable")
    parts = np.split(diffs[order], np.flatnonzero(np.diff(codes[order])) + 1)
    draws = np.array(
        [np.concatenate([parts[j] for j in rng.integers(0, len(parts), len(parts))]).mean() for _ in range(N_BOOTSTRAP)]
    )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _report(claim: str, diffs: np.ndarray, states: np.ndarray, gmap: dict[str, int], rng: np.random.Generator) -> dict:
    groups = np.array([gmap.get(s, -1) for s in states])
    lo_s, hi_s = _ci(diffs, states, rng)
    lo_g, hi_g = _ci(diffs, groups, rng)
    crosses = lo_g <= 0.0 <= hi_g
    verdict = "no longer supported inferentially" if crosses else "unchanged"
    logger.info(
        f"{claim}\n"
        f"    mean paired diff {diffs.mean():+.5f} s over {len(diffs):,} records, "
        f"{len(np.unique(states)):,} states, {len(np.unique(groups)):,} topology groups\n"
        f"    by state  CI ({lo_s:+.5f}, {hi_s:+.5f})  width {hi_s - lo_s:.5f}\n"
        f"    by group  CI ({lo_g:+.5f}, {hi_g:+.5f})  width {hi_g - lo_g:.5f}  "
        f"[{(hi_g - lo_g) / (hi_s - lo_s):.2f}x wider]  -> {verdict.upper()}"
    )
    return {
        "claim": claim,
        "mean_paired_diff": float(diffs.mean()),
        "n_records": len(diffs),
        "n_states": int(len(np.unique(states))),
        "n_topology_groups": int(len(np.unique(groups))),
        "ci_low_state": lo_s,
        "ci_high_state": hi_s,
        "ci_low_group": lo_g,
        "ci_high_group": hi_g,
        "width_ratio_group_over_state": (hi_g - lo_g) / (hi_s - lo_s),
        "group_ci_crosses_zero": bool(crosses),
        "verdict": verdict,
    }


def _keyed(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["state"] = df["state"].astype(str)
    if "record_ordinal" not in df.columns:
        df["record_ordinal"] = df.groupby("state", sort=False).cumcount()
    return df


def main() -> None:
    configure_logging()
    gmap = _group_map()
    rng = np.random.default_rng(SEED)
    rows = []

    # A - retrieval against gradient boosting on matched records.
    r = _keyed(RETRIEVAL)
    r = r[r["covered"] & r["cct_weighted_per_location"].notna()]
    m = _keyed(SUPERVISED)
    m = m[m["model"] == "hist_gradient_boosting"]
    j = r.merge(m[["state", "record_ordinal", "cct_pred"]], on=["state", "record_ordinal"])
    truth = j["cct_true"].to_numpy(float)
    d = np.abs(truth - j["cct_weighted_per_location"].to_numpy(float)) - np.abs(truth - j["cct_pred"].to_numpy(float))
    rows.append(_report("A retrieval - gradient boosting, matched records", d, j["state"].to_numpy(str), gmap, rng))

    # B - causal availability penalty on causally covered records.
    base, caus = _keyed(BASELINE), _keyed(CAUSAL)
    caus = caus[caus["covered"] & caus["cct_weighted_per_location"].notna()]
    base = base[base["covered"] & base["cct_weighted_per_location"].notna()]
    jb = caus.merge(
        base[["state", "record_ordinal", "cct_weighted_per_location"]],
        on=["state", "record_ordinal"],
        suffixes=("_causal", "_retro"),
    )
    t = jb["cct_true"].to_numpy(float)
    db = np.abs(t - jb["cct_weighted_per_location_causal"].to_numpy(float)) - np.abs(
        t - jb["cct_weighted_per_location_retro"].to_numpy(float)
    )
    rows.append(_report("B causal - retrospective, causally covered", db, jb["state"].to_numpy(str), gmap, rng))

    # C - topology-filter ablation, per seed (filter off minus filter on).
    for suffix in ABLATION_SEEDS:
        wf = pd.DataFrame(joblib.load(TMP_DIR / f"report-eles-2026-06-lines_only-sample300{suffix}.joblib"))
        nf = pd.DataFrame(joblib.load(TMP_DIR / f"report-eles-2026-06-slovenia_only-sample300{suffix}.joblib"))
        key = ["state", "crit_gen_true", "location_true"]
        for f in (wf, nf):
            f["_occ"] = f.groupby(key).cumcount()
        k2 = key + ["_occ"]
        a = wf.dropna(subset=["cct_weighted_per_location"]).set_index(k2)
        b = nf.dropna(subset=["cct_weighted_per_location"]).set_index(k2)
        common = a.index.intersection(b.index)
        a, b = a.loc[common].reset_index(), b.loc[common].reset_index()
        err_on = (a["cct_true"] - a["cct_weighted_per_location"]).abs().to_numpy(float)
        err_off = (b["cct_true"] - b["cct_weighted_per_location"]).abs().to_numpy(float)
        label = f"C ablation off - on, seed {suffix.removeprefix('-seed') or '42'}"
        rows.append(_report(label, err_off - err_on, a["state"].astype(str).to_numpy(), gmap, rng))

    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
