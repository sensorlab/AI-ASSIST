"""Supplementary table for Note S11: feature-family ablation (analysis 11) and exact Shapley attribution (analysis 12)
of the ELES distance, both exploratory (R2.10).

Writes results/tables/family_relevance.tex (two tabular bodies, after the markers "% ablation" and "% shapley") from
results/data/eles_feature_families.csv and eles_family_shapley.csv. Every interval is the widest of state,
topology-group and calendar-day resampling. The caption lives in the manuscript; each block is inlined there between
begin/end markers.

Run from the repository root:
    uv run python scripts/paper/make_family_relevance_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_distance_concentration_table import NOTE, _tabular  # noqa: E402

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_DIR / "results" / "data"
OUTPUT_PATH: Final[Path] = PROJECT_DIR / "results" / "tables" / "family_relevance.tex"
FAMILY_LABELS: Final[dict[str, str]] = {
    "active": "Active power",
    "angle": "Voltage angle",
    "reactive": "Reactive power",
    "voltage": "Voltage magnitude",
    "short_circuit": "Short-circuit power",
}
AVAILABILITIES: Final[tuple[str, ...]] = ("retrospective", "causal_lag0")


def signed(x: float, scale: float = 1000, digits: int = 2) -> str:
    return f"${'-' if x < 0 else '+'}${abs(scale * x):.{digits}f}"


def interval(r: pd.Series, scale: float = 1000, digits: int = 2, sign: bool = True) -> str:
    fmt = (lambda x: signed(x, scale, digits)) if sign else (lambda x: f"{scale * x:.{digits}f}")
    return f"{fmt(r['estimate'])} ({fmt(r['ci_low'])}, {fmt(r['ci_high'])})"


def ablation() -> str:
    d = pd.read_csv(DATA_DIR / "eles_feature_families.csv")
    dims = d[d["quantity"] == "n_dimensions"].set_index("representation")["estimate"]
    decision = d[d["quantity"].str.startswith("decision_")].set_index(["representation", "availability"])
    rows = [
        f"{label} & {int(dims[f'only_{f}']):,} & "
        + " & ".join(interval(decision.loc[(f"{m}_{f}", a)]) for m in ("without", "only") for a in AVAILABILITIES)
        + " \\\\"
        for f, label in FAMILY_LABELS.items()
    ]
    header = [
        "& & \\multicolumn{2}{c}{Family removed (ms)} & \\multicolumn{2}{c}{Family alone (ms)} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Family & Dimensions & Retrospective & Causal & Retrospective & Causal \\\\",
    ]
    return _tabular("lrrrrr", header, rows)


def shapley() -> str:
    d = pd.read_csv(DATA_DIR / "eles_family_shapley.csv")
    d = d[d["decides"]].assign(width=lambda x: x["ci_high"] - x["ci_low"])
    w = d.sort_values("width").groupby(["availability", "quantity"]).tail(1).set_index(["availability", "quantity"])

    def cells(f: str) -> str:
        return " & ".join(
            f"{interval(w.loc[(a, f'shapley_{f}')], sign=False)} & "
            f"{interval(w.loc[(a, f'share_{f}')], scale=100, digits=1, sign=False)}"
            for a in AVAILABILITIES
        )

    rows = [f"{label} & {cells(f)} \\\\" for f, label in FAMILY_LABELS.items()]
    total = " & ".join(f"{interval(w.loc[(a, 'value_full')], sign=False)} & 100" for a in AVAILABILITIES)
    rows += ["\\midrule", f"Sum: kernel against uniform weights & {total} \\\\"]
    header = [
        "& \\multicolumn{2}{c}{Retrospective} & \\multicolumn{2}{c}{Causal} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Family & Shapley value (ms) & Share (\\%) & Shapley value (ms) & Share (\\%) \\\\",
    ]
    return _tabular("lrrrr", header, rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        f"{NOTE.replace('distance_concentration', 'family_relevance')}\n% ablation\n{ablation()}\n% shapley\n{shapley()}\n"
    )
    print(OUTPUT_PATH.read_text())


if __name__ == "__main__":
    main()
