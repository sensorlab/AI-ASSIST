"""Compares retrospective and causal ELES retrieval, the one-sided counterpart to the sweep.

The temporal-exclusion sweep withholds states on both sides of the query. Only the later side
is unavailable to a deployment: a state recorded earlier is legitimate reference data once its
contingencies have been simulated, and relying on it is regime tracking rather than leakage.
The sweep therefore measures robustness to a hole in the archive, not deployment, and it
overstates the operational penalty.

This compares the committed leave-one-state-out run against a causal run produced by

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only ELES_CAUSAL_LAG_HOURS=L \\
        uv run python scripts/evaluation/eles_generator_diagnostics_selected.py 0 <n_jobs>

in which a query may use only states recorded at or before its own timestamp minus L, never
including itself. Both arms are compared on the records covered in both, so the accuracy
comparison is paired; the coverage loss is reported separately because it is a result in its
own right.

Every causally covered record is also covered retrospectively, since causal availability is a
subset of retrospective availability. The script asserts that rather than assuming it.

Run from the repository root:
    uv run python scripts/paper/eles_causal_evaluation.py [lag_hours ...]
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.benchmarking import signed_wilcoxon_z  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_causal_evaluation.csv"
BASELINE: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"
# The comparator the Discussion quotes against the causal arm. Its folds are not causally
# restricted; see the note at the join below.
SUPERVISED: Final[Path] = TMP_DIR / "ml_benchmark_predictions-eles-2026-06.parquet"
SUPERVISED_MODEL: Final[str] = "hist_gradient_boosting"

N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5
KEY: Final[list[str]] = ["state", "record_ordinal"]


def _arm(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the {label} arm first (see module docstring).")
    df = pd.read_parquet(path)
    if "record_ordinal" not in df.columns:
        df = df.copy()
        df["record_ordinal"] = df.groupby("state", sort=False).cumcount()
    # gen_true_is_selected is the joint recovery indicator: false both when the wrong candidate
    # is chosen and when the recorded generator is not a candidate at all. The conditional rate
    # cannot be recomputed here, since the causal artifact carries no gen_true_rank, so the two
    # arms must be compared on this measure rather than against the conditional top-1 figure.
    cols = KEY + ["cct_true", "loc_true", "covered", "cct_weighted_per_location"]
    if "gen_true_is_selected" in df.columns:
        cols = cols + ["gen_true_is_selected"]
    df = df[cols].copy()
    df["state"] = df["state"].astype(str)
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    df["has_estimate"] = df["covered"] & df["cct_weighted_per_location"].notna()
    return df


def _discover_lags() -> list[float]:
    pattern = re.compile(r"_causal([0-9.]+)h\.parquet$")
    return sorted(float(m.group(1)) for p in TMP_DIR.glob("*_causal*h.parquet") if (m := pattern.search(p.name)))


def _window(lag: float) -> dict[str, object]:
    control = TMP_DIR / f"eles_generator_diagnostics_selected_eles-2026-06_lines_only_causal{lag:g}h.parquet"
    base, caus = _arm(BASELINE, "baseline"), _arm(control, "causal")
    merged = base.merge(caus, on=KEY, suffixes=("_base", "_caus"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(caus):
        raise ValueError(f"join dropped records: base={len(base)}, causal={len(caus)}, merged={len(merged)}")
    if not np.allclose(merged["cct_true_base"], merged["cct_true_caus"], rtol=0.0, atol=1e-12):
        raise ValueError("cct_true differs between arms after the record-key join")
    if not (merged["loc_true_base"] == merged["loc_true_caus"]).all():
        raise ValueError("loc_true differs between arms after the record-key join")

    only_causal = int((merged["has_estimate_caus"] & ~merged["has_estimate_base"]).sum())
    if only_causal:
        raise ValueError(f"{only_causal} records are covered causally but not retrospectively, which is impossible")

    paired = merged[merged["has_estimate_base"] & merged["has_estimate_caus"]]
    mae_base, mae_caus = float(paired["err_base"].mean()), float(paired["err_caus"].mean())
    diff = mae_caus - mae_base

    diffs = [(g["err_caus"] - g["err_base"]).to_numpy() for _s, g in paired.groupby("state")]
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(diffs), size=len(diffs))
        draws[b] = np.concatenate([diffs[i] for i in picked]).mean()
    ci_low, ci_high = float(np.percentile(draws, CI_LOW)), float(np.percentile(draws, CI_HIGH))

    per_state = paired.groupby("state")[["err_base", "err_caus"]].mean()
    w, p_value = stats.wilcoxon(per_state["err_caus"], per_state["err_base"], method="approx")
    z, n_nonzero = signed_wilcoxon_z((per_state["err_caus"] - per_state["err_base"]).to_numpy())

    # The supervised comparator on exactly the causally covered records. Its training folds are
    # NOT restricted causally, so retrieval is handicapped and the comparator is not; the
    # asymmetry works against retrieval's lead, which makes that lead conservative.
    gb_mae = float("nan")
    if SUPERVISED.exists():
        ml = pd.read_parquet(SUPERVISED)
        ml = ml[ml["model"] == SUPERVISED_MODEL].copy()
        ml["state"] = ml["state"].astype(str)
        if "record_ordinal" not in ml.columns:
            ml["record_ordinal"] = ml.groupby("state", sort=False).cumcount()
        ml = ml.drop_duplicates(subset=KEY, keep="first")
        covered = merged[merged["has_estimate_caus"]][KEY]
        g = covered.merge(ml[KEY + ["cct_true", "cct_pred"]], on=KEY, validate="one_to_one")
        if len(g) != len(covered):
            raise ValueError(f"supervised join covered {len(g)} of {len(covered)} causally covered records")
        gb_mae = float((g["cct_true"] - g["cct_pred"]).abs().mean())

    # Joint generator recovery within each arm's own covered set, the like-for-like measure.
    def _recovery(flag: str, mask: str) -> float:
        col = f"gen_true_is_selected_{flag}"
        if col not in merged.columns:
            return float("nan")
        sub = merged[merged[mask]]
        return float(sub[col].mean()) if len(sub) else float("nan")

    rec_base = _recovery("base", "has_estimate_base")
    rec_caus = _recovery("caus", "has_estimate_caus")

    logger.info(
        f"lag {lag:g} h: coverage {merged['has_estimate_base'].mean():.2%} -> {merged['has_estimate_caus'].mean():.2%}; "
        f"paired MAE {mae_base:.5f} -> {mae_caus:.5f} ({100 * diff / mae_base:+.1f}%, CI {ci_low:+.5f} {ci_high:+.5f}), "
        f"median {paired['err_base'].median():.5f} -> {paired['err_caus'].median():.5f}, p={p_value:.3e}"
    )
    return {
        "lag_hours": lag,
        "n_records": len(merged),
        "n_covered_retrospective": int(merged["has_estimate_base"].sum()),
        "n_covered_causal": int(merged["has_estimate_caus"].sum()),
        "coverage_retrospective": float(merged["has_estimate_base"].mean()),
        "coverage_causal": float(merged["has_estimate_caus"].mean()),
        "n_paired": len(paired),
        "n_states_paired": paired["state"].nunique(),
        "mae_retrospective_paired": mae_base,
        "mae_causal_paired": mae_caus,
        "mae_diff": diff,
        "mae_diff_ci_low": ci_low,
        "mae_diff_ci_high": ci_high,
        "mae_diff_pct": 100 * diff / mae_base,
        "median_ae_retrospective": float(paired["err_base"].median()),
        "median_ae_causal": float(paired["err_caus"].median()),
        "state_level_wilcoxon_W": float(w),
        "state_level_p_value": float(p_value),
        "state_level_wilcoxon_effect_size_r": float(z / np.sqrt(n_nonzero)),
        "mae_supervised_on_causal_covered": gb_mae,
        "supervised_model": SUPERVISED_MODEL,
        "retrieval_lead_over_supervised_pct": (100 * (gb_mae - mae_caus) / mae_caus)
        if gb_mae == gb_mae
        else float("nan"),
        "gen_recovery_retrospective": rec_base,
        "gen_recovery_causal": rec_caus,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }


def main() -> None:
    configure_logging()
    lags = [float(a) for a in sys.argv[1:]] or _discover_lags()
    if not lags:
        raise FileNotFoundError(f"no *_causal*h.parquet artifacts under {TMP_DIR}")
    logger.info(f"lags: {lags}")
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_window(lag) for lag in lags]).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
