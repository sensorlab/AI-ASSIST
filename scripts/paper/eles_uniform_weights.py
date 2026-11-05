"""Do the kernel weights matter on ELES? Uniform (alpha = 0) against kernel (alpha = 1) weights (R2.10a, Q2).

Pre-specified in the manuscript repository's revision/analyses/03-distance-concentration.md, committed before any
number here was computed. On ELES, K = 100 never binds, so both runs weigh the same neighbour pool; alpha = 0 gives
every neighbour weight 1, each generator estimate becomes its group mean and the support mass a neighbour count.
Coverage must therefore be identical record for record, and the script stops otherwise.

Compared paired on covered records: MAE (primary: uniform minus kernel), error quantiles, generator top-1, and the
screening triple at 150 and 200 ms. Cluster bootstrap over state, topology group and calendar day decides (widest
interval); calendar week is sensitivity. Rule: the kernel improves estimation only if the difference is positive, its
interval excludes zero and it is at least 0.010 s (label resolution); excluding zero but below 0.010 s is "negligible".

Run from the repository root (the kernel run is the validated harness output):
    nice -n 19 uv run python scripts/paper/eles_uniform_weights.py [n_jobs]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_eles_state_timestamps  # noqa: E402
from _retrieval_harness import EXACT_LINES, Rule, causal_exclusions, run_rule  # noqa: E402
from referral_decomposition import (  # noqa: E402
    DECISION_UNITS,
    ELES_CAUSAL0,
    ELES_DIR,
    ELES_LOSO,
    RAW_ZIP,
    VALIDATION,
    Stat,
    _bootstrap,
    _count,
    _eles_clusters,
    _mean,
    _ratio,
    _state_frame,
)

from src.config.logging import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "eles_uniform_weights.csv"
UNIFORM: Final[Rule] = Rule("exact_lines_uniform", EXACT_LINES.key, EXACT_LINES.max_hamming, alpha=0.0)
THRESHOLDS: Final[tuple[float, ...]] = (0.15, 0.20)
LABEL_RESOLUTION: Final[float] = 0.010
QUANTILES: Final[tuple[float, ...]] = (0.50, 0.95, 0.99)


def _diff(a: Stat, b: Stat) -> Stat:
    return lambda x, w: a(x, w) - b(x, w)


def decision(rows: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-specified rule to the primary MAE difference rows (one per resampling unit)."""
    d = rows[rows["decides"]].assign(width=lambda x: x["ci_high"] - x["ci_low"])
    w = d.loc[d["width"].idxmax()]
    excludes_zero = w["ci_low"] > 0 or w["ci_high"] < 0
    if w["estimate"] > 0 and w["ci_low"] > 0 and w["estimate"] >= LABEL_RESOLUTION:
        verdict = "kernel_improves"
    elif excludes_zero and abs(w["estimate"]) < LABEL_RESOLUTION:
        verdict = "negligible"
    elif excludes_zero and w["estimate"] < 0:
        verdict = "uniform_improves"  # not foreseen by the pre-specification; a deviation if it occurs
    else:
        verdict = "no_difference"
    return {"resampling_unit": w["resampling_unit"], "ci_low": w["ci_low"], "ci_high": w["ci_high"], "verdict": verdict}


def compare(kernel: pd.DataFrame, uniform: pd.DataFrame, units: dict[str, np.ndarray | None], availability: str):
    key = ["state", "record_ordinal"]
    m = kernel.merge(uniform, on=key, suffixes=("_k", "_u"), validate="1:1")
    if not m["covered_k"].equals(m["covered_u"]):
        raise ValueError(f"{availability}: coverage differs between alpha = 1 and alpha = 0")
    states = pd.Index(kernel["state"].drop_duplicates())
    cov = m[m["covered_k"]]
    rows: list[dict[str, Any]] = []

    def point(quantity: str, value: float) -> None:
        rows.append({"availability": availability, "quantity": quantity, "estimate": value})

    for arm in ("k", "u"):
        err = (cov[f"pred_selection_{arm}"] - cov["cct_true_k"]).abs()
        for q in QUANTILES:
            point(f"abs_error_q{int(100 * q)}_{'kernel' if arm == 'k' else 'uniform'}", float(err.quantile(q)))
    point("n_covered_records", len(cov))
    point("share_top_tie_uniform", float((cov["gens_tied_top_u"].str.count(r"\|") > 0).mean()))
    point("share_top_tie_kernel", float((cov["gens_tied_top_k"].str.count(r"\|") > 0).mean()))
    point("share_selection_differs", float((cov["gen_selected_k"] != cov["gen_selected_u"]).mean()))

    g = (
        cov.assign(
            ae_k=(cov["pred_selection_k"] - cov["cct_true_k"]).abs(),
            ae_u=(cov["pred_selection_u"] - cov["cct_true_k"]).abs(),
            top_k=(cov["gen_selected_k"] == cov["gen_true_k"]).astype(float),
            top_u=(cov["gen_selected_u"] == cov["gen_true_k"]).astype(float),
            n=1.0,
        )
        .groupby("state")[["ae_k", "ae_u", "top_k", "top_u", "n"]]
        .sum()
    )
    g = g.reindex(states, fill_value=0.0)
    arrays = {c: g[c].to_numpy() for c in g.columns}
    stats: dict[str, Stat] = {
        "mae_kernel": _ratio("ae_k", "n"),
        "mae_uniform": _ratio("ae_u", "n"),
        "mae_uniform_minus_kernel": _diff(_ratio("ae_u", "n"), _ratio("ae_k", "n")),
        "top1_kernel": _ratio("top_k", "n"),
        "top1_uniform": _ratio("top_u", "n"),
        "top1_uniform_minus_kernel": _diff(_ratio("top_u", "n"), _ratio("top_k", "n")),
    }
    for tau in THRESHOLDS:
        t = round(1000 * tau)
        for arm, col in (("kernel", "k"), ("uniform", "u")):
            sf = _state_frame(
                m.rename(columns={f"pred_screening_min_{col}": "est", "covered_k": "cov", "cct_true_k": "cct_true"}),
                "est",
                "cov",
                tau,
            ).reindex(states)
            arrays[f"ref_{arm}_{t}"] = sf["referred"].to_numpy(float)
            arrays[f"miss_{arm}_{t}"] = (sf["unsafe"] & ~sf["referred"]).to_numpy(float)
            arrays[f"caught_{arm}_{t}"] = (sf["unsafe"] & sf["referred"]).to_numpy(float)
            arrays[f"unsafe_{t}"] = sf["unsafe"].to_numpy(float)
            stats[f"referral_load_{arm}_{t}ms"] = _mean(f"ref_{arm}_{t}")
            stats[f"missed_unsafe_{arm}_{t}ms"] = _count(f"miss_{arm}_{t}")
            stats[f"recall_{arm}_{t}ms"] = _ratio(f"caught_{arm}_{t}", f"unsafe_{t}")
        stats[f"referral_load_uniform_minus_kernel_{t}ms"] = _diff(
            stats[f"referral_load_uniform_{t}ms"], stats[f"referral_load_kernel_{t}ms"]
        )
        stats[f"missed_unsafe_uniform_minus_kernel_{t}ms"] = _diff(
            stats[f"missed_unsafe_uniform_{t}ms"], stats[f"missed_unsafe_kernel_{t}ms"]
        )

    ones = np.ones(len(states))
    est = {name: stat(arrays, ones) for name, stat in stats.items()}
    for unit, clusters in units.items():
        ci = _bootstrap(arrays, stats, clusters, np.random.default_rng(42))
        for name in stats:
            rows.append(
                {
                    "availability": availability,
                    "quantity": name,
                    "resampling_unit": unit,
                    "n_units": len(states) if clusters is None else int(clusters.max()) + 1,
                    "decides": unit in DECISION_UNITS,
                    "estimate": est[name],
                    "ci_low": ci[name][0],
                    "ci_high": ci[name][1],
                }
            )
    out = pd.DataFrame(rows)
    verdict = decision(out[out["quantity"] == "mae_uniform_minus_kernel"])
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {
                        "availability": availability,
                        "quantity": f"decision_{verdict.pop('verdict')}",
                        "estimate": est["mae_uniform_minus_kernel"],
                        "decides": True,
                        **verdict,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return out


def main() -> None:
    configure_logging()
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    state_ids = pd.read_pickle(ELES_DIR / "interim" / "lf.pkl").index.astype(str)
    timestamps = load_eles_state_timestamps(RAW_ZIP).reindex(state_ids)
    if timestamps.isna().any():
        raise ValueError(f"{int(timestamps.isna().sum())} states have no timestamp")

    results = []
    for availability, kernel_path, exclusions in (
        ("retrospective", ELES_LOSO, None),
        ("causal_lag0", ELES_CAUSAL0, causal_exclusions(timestamps, 0.0)),
    ):
        uniform_path = TMP_DIR / f"harness_records_uniform_{'loso' if exclusions is None else 'causal0h'}.parquet"
        uniform, _, _ = run_rule(UNIFORM, exclusions, n_jobs=n_jobs)
        uniform.to_parquet(uniform_path, index=False)
        logger.info(f"{availability}: uniform run saved to {uniform_path}")
        kernel = pd.read_parquet(kernel_path)
        units = _eles_clusters(pd.Index(kernel["state"].drop_duplicates()))
        results.append(compare(kernel, uniform, units, availability))

    out = pd.concat(results, ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)
    for _, r in out[out["quantity"].str.startswith("decision_")].iterrows():
        logger.info(
            f"{r['availability']}: MAE uniform - kernel {r['estimate']:.5f} s "
            f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}] under {r['resampling_unit']} -> {r['quantity']}"
        )
    logger.info(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
