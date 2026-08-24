"""Paired common-support comparison of the two conditional critical-generator levels in
journal.tex Table 1 (oracle vs. highest-support candidate), the analysis the manuscript
previously declined to run and flagged as a prerequisite for interpreting the row difference.

Table 1 reports each level's MAE on its own covered population: the oracle row needs the
recorded critical generator to be present among the retrieved candidates, the selected row
needs only that some candidate exists at the recorded location. The populations therefore
differ by exactly the records where the recorded generator was never retrievable
(`n_true_gen_not_a_candidate` in generator_identification_summary.csv), which are
systematically the harder ones. Reading the marginal row difference as a generator-selection
cost confounds that composition shift with the selection effect itself.

This script restricts both levels to their common support - records where both predictions
exist - and reports:
  - the paired record-level MAE difference (selection minus oracle) with a state-clustered
    bootstrap CI, matching the state-level resampling convention used everywhere else in the
    paper (contingencies from one pre-fault state stay together);
  - a paired Wilcoxon signed-rank test on per-state mean differences, the same unit of
    analysis eles_topology_ablation_significance.py uses, since records within a state are
    not independent;
  - the fraction of states where selection is actually worse, which the mean alone hides.

The marginal (Table 1) figures are recomputed here too, so the CSV carries the composition
effect and the paired effect side by side rather than requiring a cross-file join.

Run from the repository root:
    uv run python scripts/paper/paired_common_support_generator.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmarking import signed_wilcoxon_z  # noqa: E402
from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
PAPER_DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PAPER_DATA_DIR / "paired_common_support_generator.csv"

N_BOOTSTRAP: Final[int] = 1000
SEED: Final[int] = 42
CI_LOW: Final[float] = 2.5
CI_HIGH: Final[float] = 97.5

# Per-dataset record files and their oracle/selection prediction column names. The two
# de-oracling scripts were written independently and use different column names for the same
# two quantities, so the mapping is explicit rather than inferred.
DATASETS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "bus39",
        # BUS39 leave-one-state-out, matching the ELES arm below.
        "generator_deoracled_records_loso.parquet",
        "pred_oracle_gen",
        "pred_selection",
    ),
    (
        "eles/2026-06",
        "eles_deoracled_records_eles-2026-06_lines_only.parquet",
        "pred_oracle_gen_and_loc",
        "pred_gen_deoracled_selection",
    ),
)


def _analyze(dataset: str, filename: str, oracle_col: str, selection_col: str) -> dict[str, object]:
    path = TMP_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with the corresponding de-oracling script "
            f"(generator_deoracled_bound.py for BUS39, eles_deoracled_bound.py for ELES)."
        )
    df = pd.read_parquet(path)

    has_oracle = df[oracle_col].notna()
    has_selection = df[selection_col].notna()
    common = has_oracle & has_selection

    # Marginal figures, reproducing Table 1's two rows on their own covered populations.
    mae_oracle_marginal = float((df.loc[has_oracle, oracle_col] - df.loc[has_oracle, "cct_true"]).abs().mean())
    mae_selection_marginal = float(
        (df.loc[has_selection, selection_col] - df.loc[has_selection, "cct_true"]).abs().mean()
    )

    # Paired figures on common support.
    paired = df.loc[common]
    ae_oracle = (paired[oracle_col] - paired["cct_true"]).abs()
    ae_selection = (paired[selection_col] - paired["cct_true"]).abs()
    mae_oracle_common = float(ae_oracle.mean())
    mae_selection_common = float(ae_selection.mean())

    per_state = pd.DataFrame(
        {"state": paired["state"].to_numpy(), "oracle": ae_oracle.to_numpy(), "selection": ae_selection.to_numpy()}
    ).groupby("state")
    state_means = per_state.mean()
    n_states = len(state_means)

    w_state, p_state = stats.wilcoxon(state_means["selection"], state_means["oracle"], method="approx")
    z_state, n_nonzero = signed_wilcoxon_z((state_means["selection"] - state_means["oracle"]).to_numpy())
    # z/sqrt(n) is the Wilcoxon/Rosenthal standardized effect size r, matching the convention
    # and the naming correction already applied in eles_topology_ablation_significance.py.
    # The sign comes from signed_wilcoxon_z rather than scipy's .zstatistic - see that
    # helper's docstring. This analysis is exactly the case the helper exists for: BUS39 and
    # ELES point in opposite directions, and scipy's z would report both as negative.
    # Positive r means selection is worse than the oracle.
    r_effect = float(z_state / np.sqrt(n_nonzero))
    frac_states_worse = float((state_means["selection"] > state_means["oracle"]).mean())

    # State-clustered bootstrap on the record-level paired difference: resample whole states
    # with replacement, keeping every contingency from a state together, then recompute the
    # record-weighted mean difference. This is the record-level quantity the MAE columns
    # report, not a mean of per-state means.
    state_ids = state_means.index.to_numpy()
    diff_by_state = {
        sid: (grp["selection"] - grp["oracle"]).to_numpy()
        for sid, grp in pd.DataFrame(
            {
                "state": paired["state"].to_numpy(),
                "oracle": ae_oracle.to_numpy(),
                "selection": ae_selection.to_numpy(),
            }
        ).groupby("state")
    }
    diff_arrays = [diff_by_state[sid] for sid in state_ids]

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOTSTRAP)
    n_state_ids = len(state_ids)
    for b in range(N_BOOTSTRAP):
        sampled = rng.integers(0, n_state_ids, size=n_state_ids)
        boot[b] = np.concatenate([diff_arrays[i] for i in sampled]).mean()

    point_diff = mae_selection_common - mae_oracle_common
    ci_low = float(np.percentile(boot, CI_LOW))
    ci_high = float(np.percentile(boot, CI_HIGH))

    logger.info(
        f"{dataset}: marginal {mae_oracle_marginal:.5f} (n={int(has_oracle.sum()):,}) vs "
        f"{mae_selection_marginal:.5f} (n={int(has_selection.sum()):,}), "
        f"rel {100 * (mae_selection_marginal - mae_oracle_marginal) / mae_oracle_marginal:+.2f}%"
    )
    logger.info(
        f"{dataset}: paired on {int(common.sum()):,} records / {n_states:,} states, "
        f"{mae_oracle_common:.5f} vs {mae_selection_common:.5f}, "
        f"diff {point_diff:+.5f} ({ci_low:+.5f}, {ci_high:+.5f}), "
        f"rel {100 * point_diff / mae_oracle_common:+.2f}%, "
        f"W={w_state:.1f}, p={p_state:.3e}, r={r_effect:.3f}, "
        f"states worse {100 * frac_states_worse:.1f}%"
    )

    return {
        "dataset": dataset,
        "n_records_common": int(common.sum()),
        "n_states_common": n_states,
        "n_records_oracle_only": int((has_oracle & ~has_selection).sum()),
        "n_records_selection_only": int((has_selection & ~has_oracle).sum()),
        "mae_oracle_marginal": mae_oracle_marginal,
        "n_oracle_marginal": int(has_oracle.sum()),
        "mae_selection_marginal": mae_selection_marginal,
        "n_selection_marginal": int(has_selection.sum()),
        "rel_diff_marginal_pct": 100 * (mae_selection_marginal - mae_oracle_marginal) / mae_oracle_marginal,
        "mae_oracle_common": mae_oracle_common,
        "mae_selection_common": mae_selection_common,
        "mean_paired_diff": point_diff,
        "mean_paired_diff_ci_low": ci_low,
        "mean_paired_diff_ci_high": ci_high,
        "rel_diff_common_pct": 100 * point_diff / mae_oracle_common,
        "state_level_wilcoxon_W": float(w_state),
        "state_level_p_value": float(p_state),
        "state_level_z": float(z_state),
        "state_level_wilcoxon_effect_size_r": r_effect,
        "frac_states_selection_worse": frac_states_worse,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }


def main() -> None:
    configure_logging()
    rows = [_analyze(dataset, filename, oracle, selection) for dataset, filename, oracle, selection in DATASETS]
    out = pd.DataFrame(rows)
    PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
