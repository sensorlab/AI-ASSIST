"""How many screening outcomes hinge on one CCT label step? (R2.14c)

Pre-specified in the manuscript repository's revision/analyses/04-clustered-intervals-and-label-steps.md, committed
before any number here was computed. Labels lie on a 0.01 s grid. Only the ground truth of the evaluated state is
shifted; estimates, and therefore referral decisions, are unchanged. (Shifting archive and truth together moves every
estimate by the same step, which equals moving tau and is read from the threshold curve instead.)

- shift -1: every recorded CCT one step lower, the worst case of one-step optimism from the simulation horizon.
- shift +1: every recorded CCT one step higher, the largest true CCT the bisection allows (discretisation).

Comparisons are made in integer grid steps so that 0.19 + 0.01 compares equal to 0.20.

Run from the repository root:
    nice -n 19 uv run python scripts/paper/label_step_perturbation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from referral_decomposition import (  # noqa: E402
    BUS39_LOSO,
    DECISION_UNITS,
    ELES_CAUSAL0,
    ELES_LOSO,
    VALIDATION,
    Stat,
    _bootstrap,
    _count,
    _eles_clusters,
    _ratio,
    _state_frame,
)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
CURVE: Final[Path] = PROJECT_DIR / "results" / "data" / "screening_threshold_curve.csv"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "data" / "label_step_perturbation.csv"
STEP: Final[float] = 0.01
THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20)
SHIFTS: Final[tuple[int, ...]] = (-1, 0, 1)


def steps(x: pd.Series | float) -> Any:
    return np.rint(np.asarray(x) / STEP).astype(int)


def shifted(records: pd.DataFrame, tau: float, shift: int) -> pd.DataFrame:
    """Per state: referred (unchanged by any shift), unsafe and boundary under a ground-truth shift of `shift` steps."""
    base = _state_frame(records, "pred_screening_min", "covered", tau)
    y = steps(records.groupby("state", sort=False)["cct_true"].min().reindex(base.index))
    t = int(steps(tau))
    return pd.DataFrame(
        {
            "referred": base["referred"].to_numpy(),
            "unsafe": y + shift < t,
            "boundary": (y >= t - 1) & (y < t + 1),
        },
        index=base.index,
    )


def evaluate(records: pd.DataFrame, units: dict[str, np.ndarray | None], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in THRESHOLDS:
        frames = {s: shifted(records, tau, s) for s in SHIFTS}
        reference = _state_frame(records, "pred_screening_min", "covered", tau)
        if not frames[0]["unsafe"].equals(reference["unsafe"]):
            raise ValueError("the unshifted grid comparison does not reproduce the screening rule")
        if frames[0]["unsafe"].sum() == 0 and frames[-1]["unsafe"].sum() == 0:
            rows.append({**ctx, "tau_s": tau, "quantity": "n_unsafe_all_shifts", "estimate": 0})
            continue
        arrays: dict[str, np.ndarray] = {}
        stats: dict[str, Stat] = {}
        for s, f in frames.items():
            arrays[f"unsafe_{s}"] = f["unsafe"].to_numpy(float)
            arrays[f"missed_{s}"] = (f["unsafe"] & ~f["referred"]).to_numpy(float)
            arrays[f"caught_{s}"] = (f["unsafe"] & f["referred"]).to_numpy(float)
            stats[f"n_unsafe_shift{s:+d}"] = _count(f"unsafe_{s}")
            stats[f"missed_unsafe_shift{s:+d}"] = _count(f"missed_{s}")
            stats[f"recall_shift{s:+d}"] = _ratio(f"caught_{s}", f"unsafe_{s}")
        for s in (-1, 1):
            arrays[f"dmissed_{s}"] = arrays[f"missed_{s}"] - arrays["missed_0"]
            stats[f"missed_unsafe_shift{s:+d}_minus_unshifted"] = _count(f"dmissed_{s}")
        b = frames[0]["boundary"]
        arrays["boundary"] = b.to_numpy(float)
        arrays["boundary_referred"] = (b & frames[0]["referred"]).to_numpy(float)
        stats["n_boundary_states"] = _count("boundary")
        stats["n_boundary_states_referred"] = _count("boundary_referred")
        ones = np.ones(len(b))
        for unit, clusters in units.items():
            ci = _bootstrap(arrays, stats, clusters, np.random.default_rng(42))
            for name, stat in stats.items():
                rows.append(
                    {
                        **ctx,
                        "tau_s": tau,
                        "quantity": name,
                        "resampling_unit": unit,
                        "decides": unit in DECISION_UNITS or ctx["dataset"] == "BUS39",
                        "estimate": stat(arrays, ones),
                        "ci_low": ci[name][0],
                        "ci_high": ci[name][1],
                    }
                )
    return rows


def joint_shift(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Archive and truth shifted together: the threshold curve at tau - 0.01 (shift +1) and tau + 0.01 (shift -1)."""
    c = pd.read_csv(CURVE)
    c = c[(c["dataset"] == ctx["dataset"]) & (c["availability"] == ctx["availability"]) & (c["family"] == "P0")]
    c = c.assign(_steps=steps(c["tau_s"])).set_index("_steps")
    rows = []
    for tau in THRESHOLDS:
        for s in (-1, 1):
            r = c.loc[int(steps(tau)) - s]
            for q in ("referral_load", "missed_unsafe", "recall"):
                rows.append({**ctx, "tau_s": tau, "quantity": f"joint_{q}_shift{s:+d}", "estimate": r[q]})
    return rows


def main() -> None:
    if not pd.read_csv(VALIDATION)["passed"].all():
        raise RuntimeError(f"{VALIDATION.name} has failed checks; the harness output must not be used")
    rows: list[dict[str, Any]] = []
    bus39 = pd.read_parquet(BUS39_LOSO, columns=["state", "covered", "cct_true", "pred_screening_min"])
    ctx = {"dataset": "BUS39", "availability": "retrospective"}
    rows += evaluate(bus39, {"state": None}, ctx) + joint_shift(ctx)
    for availability, path in (("retrospective", ELES_LOSO), ("causal_lag0", ELES_CAUSAL0)):
        records = pd.read_parquet(path, columns=["state", "covered", "cct_true", "pred_screening_min"])
        units = _eles_clusters(pd.Index(records["state"].drop_duplicates()))
        ctx = {"dataset": "ELES", "availability": availability}
        rows += evaluate(records, units, ctx) + joint_shift(ctx)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
