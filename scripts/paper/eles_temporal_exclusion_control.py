"""Compares ELES retrieval accuracy with and without a temporal-exclusion window (review C5).

The ELES evaluation removes the query state from its own reference population but leaves every
other state, including the ones recorded an hour either side of it. ELES's same-topology groups
are largely contiguous hourly bursts spanning one to three days, so retrieval can return what is
nearly the same operating state shortly before or after the query. Two reviewers reached this
independently, and it is the mechanism most likely to explain the one place retrieval beats a
supervised baseline. The suggestive fingerprint is that ELES retrieval's median absolute error
equals the CCT bisection tolerance of 0.010 s.

This compares the committed leave-one-state-out run against a control produced by

    DATASET_NAME=eles/2026-06 TOPOLOGY_VARIANT=lines_only ELES_TEMPORAL_EXCLUSION_HOURS=24 \\
        uv run python scripts/evaluation/eles_generator_diagnostics_selected.py 0 <n_jobs>

which additionally excludes every state within the window. Both arms are restricted to the
records covered in both, so the comparison is paired and not a coverage artifact; coverage loss
in the control arm is reported separately, since it is itself a result - a state whose only
compatible neighbours were its own temporal burst has no support left once they are removed.

Run from the repository root:
    uv run python scripts/paper/eles_temporal_exclusion_control.py [hours]
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
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "eles_temporal_exclusion_control.csv"

BASELINE: Final[Path] = TMP_DIR / "eles_generator_diagnostics_selected_eles-2026-06_lines_only.parquet"
# Supervised predictions to score on the same surviving records. Every ml_benchmark run for
# this dataset is picked up, so a later run adding models extends the comparison without
# editing this script; (model, state, record_ordinal) is deduplicated keeping the newest file.
ML_PREDICTION_GLOB: Final[str] = "ml_benchmark_predictions-eles-2026-06*.parquet"
N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42
CI_LOW, CI_HIGH = 2.5, 97.5

KEY: Final[list[str]] = ["state", "record_ordinal"]


def _arm(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the {label} arm first (see module docstring).")
    df = pd.read_parquet(path)
    if "record_ordinal" not in df.columns:
        # The baseline artifact predates the record_ordinal column. The producing loop assigns
        # it as enumerate() over each state's TSA subset in file order, so cumcount within
        # state reconstructs it exactly. The cct_true/loc_true agreement check after the join
        # is what actually validates this, rather than the assumption on its own.
        logger.info(f"{label}: reconstructing record_ordinal by within-state cumcount")
        df = df.copy()
        df["record_ordinal"] = df.groupby("state", sort=False).cumcount()
    df = df[KEY + ["cct_true", "loc_true", "covered", "cct_weighted_per_location"]].copy()
    df["err"] = (df["cct_true"] - df["cct_weighted_per_location"]).abs()
    df["has_estimate"] = df["covered"] & df["cct_weighted_per_location"].notna()
    return df


def _window(hours: float) -> dict[str, object]:
    control_path = TMP_DIR / f"eles_generator_diagnostics_selected_eles-2026-06_lines_only_excl{hours:g}h.parquet"

    base = _arm(BASELINE, "baseline")
    ctrl = _arm(control_path, "control")

    merged = base.merge(ctrl, on=KEY, suffixes=("_base", "_ctrl"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(ctrl):
        raise ValueError(f"join dropped records: base={len(base)}, ctrl={len(ctrl)}, merged={len(merged)}")
    # Both arms must describe the same contingency on every joined row. This is the check that
    # validates the reconstructed record_ordinal above, not the reconstruction's own logic.
    if not np.allclose(merged["cct_true_base"], merged["cct_true_ctrl"], rtol=0.0, atol=1e-12):
        raise ValueError("cct_true differs between arms after the record-key join")
    if not (merged["loc_true_base"] == merged["loc_true_ctrl"]).all():
        mismatched = int((merged["loc_true_base"] != merged["loc_true_ctrl"]).sum())
        raise ValueError(f"loc_true differs on {mismatched} rows after the record-key join")

    n_records = len(merged)
    cov_base = int(merged["has_estimate_base"].sum())
    cov_ctrl = int(merged["has_estimate_ctrl"].sum())
    both = merged["has_estimate_base"] & merged["has_estimate_ctrl"]
    paired = merged[both]
    logger.info(
        f"records={n_records:,}  covered baseline={cov_base:,} ({cov_base / n_records:.2%})  "
        f"covered control={cov_ctrl:,} ({cov_ctrl / n_records:.2%})  paired={len(paired):,}"
    )

    mae_base_marginal = float(merged.loc[merged["has_estimate_base"], "err_base"].mean())
    mae_ctrl_marginal = float(merged.loc[merged["has_estimate_ctrl"], "err_ctrl"].mean())
    mae_base = float(paired["err_base"].mean())
    mae_ctrl = float(paired["err_ctrl"].mean())
    diff = mae_ctrl - mae_base

    # State-clustered bootstrap on the paired record-level difference, the paper's convention.
    diffs = [(grp["err_ctrl"] - grp["err_base"]).to_numpy() for _sid, grp in paired.groupby("state")]
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        picked = rng.integers(0, len(diffs), size=len(diffs))
        draws[b] = np.concatenate([diffs[i] for i in picked]).mean()
    ci_low, ci_high = float(np.percentile(draws, CI_LOW)), float(np.percentile(draws, CI_HIGH))

    state_means = paired.groupby("state")[["err_base", "err_ctrl"]].mean()
    w, p = stats.wilcoxon(state_means["err_ctrl"], state_means["err_base"], method="approx")
    z, n_nonzero = signed_wilcoxon_z((state_means["err_ctrl"] - state_means["err_base"]).to_numpy())

    logger.info(
        f"paired MAE: baseline {mae_base:.5f} -> control {mae_ctrl:.5f}  "
        f"diff {diff:+.5f} ({ci_low:+.5f}, {ci_high:+.5f}), {100 * diff / mae_base:+.2f}%"
    )
    logger.info(
        f"median AE: baseline {paired['err_base'].median():.5f} -> control {paired['err_ctrl'].median():.5f}  "
        f"| state-level W={w:.1f}, p={p:.3e}, r={z / np.sqrt(n_nonzero):.3f}"
    )

    # Supervised comparison on exactly the records that keep support after exclusion. Without
    # this the control arm's MAE has nothing to be read against: the question is not only
    # whether error rises but whether retrieval still beats the baselines it was reported to
    # beat, and that has to be scored on the same records rather than on full-population
    # figures from Table 2.
    ml_rows: list[dict[str, object]] = []
    surviving = paired[KEY].copy()
    surviving["state"] = surviving["state"].astype(str)
    ml_files = sorted(TMP_DIR.glob(ML_PREDICTION_GLOB), key=lambda p: p.stat().st_mtime)
    if ml_files:
        # A run's filename suffix becomes part of the model label. Without this, a temporally
        # blocked run and an unblocked one both carry model="hist_gradient_boosting" and the
        # newer file silently replaces the older under the deduplication below, changing a
        # reported number without any visible cause.
        frames = []
        for path in ml_files:
            frame = pd.read_parquet(path)
            suffix = path.stem[len("ml_benchmark_predictions-eles-2026-06") :]
            if suffix:
                frame["model"] = frame["model"].astype(str) + suffix
            frames.append(frame)
        ml = pd.concat(frames, ignore_index=True)
        ml["state"] = ml["state"].astype(str)
        ml = ml.drop_duplicates(subset=["model", *KEY], keep="last")
        for model, sub in ml.groupby("model"):
            joined = surviving.merge(sub[[*KEY, "cct_pred", "cct_true"]], on=KEY, how="inner")
            if joined.empty:
                continue
            ae = (joined["cct_pred"] - joined["cct_true"]).abs()
            ml_rows.append(
                {"model": str(model), "mae": float(ae.mean()), "median_ae": float(ae.median()), "n": len(joined)}
            )
            logger.info(
                f"  {str(model):<24} on surviving records: MAE={ae.mean():.5f}  median={ae.median():.5f}  n={len(joined):,}"
            )
    else:
        logger.warning(f"No {ML_PREDICTION_GLOB} under {TMP_DIR}; skipping the supervised comparison")

    row = {
        "exclusion_hours": hours,
        "n_records": n_records,
        "n_covered_baseline": cov_base,
        "n_covered_control": cov_ctrl,
        "coverage_baseline": cov_base / n_records,
        "coverage_control": cov_ctrl / n_records,
        "n_paired": len(paired),
        "n_states_paired": paired["state"].nunique(),
        "mae_baseline_marginal": mae_base_marginal,
        "mae_control_marginal": mae_ctrl_marginal,
        "mae_baseline_paired": mae_base,
        "mae_control_paired": mae_ctrl,
        "mae_diff": diff,
        "mae_diff_ci_low": ci_low,
        "mae_diff_ci_high": ci_high,
        "mae_diff_pct": 100 * diff / mae_base,
        "median_ae_baseline": float(paired["err_base"].median()),
        "median_ae_control": float(paired["err_ctrl"].median()),
        "state_level_wilcoxon_W": float(w),
        "state_level_p_value": float(p),
        "state_level_z": z,
        "state_level_wilcoxon_effect_size_r": float(z / np.sqrt(n_nonzero)),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }
    for ml_row in ml_rows:
        row[f"mae_{ml_row['model']}_surviving"] = ml_row["mae"]
        row[f"median_ae_{ml_row['model']}_surviving"] = ml_row["median_ae"]

    return row


def _discover_windows() -> list[float]:
    """Every exclusion window with a control artifact on disk, ascending."""
    pattern = re.compile(r"_excl([0-9.]+)h\.parquet$")
    found = [float(m.group(1)) for path in TMP_DIR.glob("*_excl*h.parquet") if (m := pattern.search(path.name))]
    return sorted(found)


def main() -> None:
    configure_logging()
    windows = [float(a) for a in sys.argv[1:]] or _discover_windows()
    if not windows:
        raise FileNotFoundError(f"no *_excl*h.parquet control artifacts under {TMP_DIR}")
    logger.info(f"windows: {windows}")
    rows = [_window(h) for h in windows]
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH} ({len(rows)} windows)")


if __name__ == "__main__":
    main()
