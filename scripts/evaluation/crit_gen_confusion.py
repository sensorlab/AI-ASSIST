"""Leave-one-state-out critical-generator identification, as a confusion matrix.

The leave-one-state-out benchmark (scripts/evaluation/eles_benchmark.py) scores CCT error
inside the *recorded* critical generator's group, so it never asks whether retrieval would
have picked that generator in the first place. This script asks exactly that: at each
record's recorded fault location, rank the candidate generators and compare the top-ranked
one with the recorded outcome.

Ranking uses LocationGroupReport.crit_gen_likelihood, not the per-generator weight_mass:
weight_mass is normalized within each generator's own neighbor set and is therefore a share
of that generator's own group, not a quantity comparable across generators (see the model's
docstring in src/domain/estimation/models.py). crit_gen_likelihood is built from raw,
never-renormalized kernel mass precisely so it can be compared across generators.

Stable records - where the simulation reached its ceiling and no generator went critical -
carry a null Crit_gen. Retrieval has no "stable" class to predict, since the service groups
by the recorded generator, so those records are reported separately as their own row rather
than folded into the matrix as if a generator had been named.

Dataset-generic: everything resolves from DATASET_NAME through the production service path.
Outputs land beside the dataset for non-manuscript datasets, matching eles_benchmark.py.

Run from the repository root, e.g.:
    DATASET_NAME=interscada/pl QDRANT_URL=":memory:" \
        uv run python scripts/evaluation/crit_gen_confusion.py [n_states]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from scripts.evaluation.benchmark import normalize_label
from scripts.evaluation.eles_benchmark import MANUSCRIPT_DATASETS
from src.config.logging import configure_logging
from src.config.settings import get_app_settings
from src.domain.estimation.service import _dataset_paths, build_estimation_service
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
STABLE_LABEL: Final[str] = "stable"


def _gen_label(value: Any) -> str:
    """Canonical class label for a critical generator.

    Three spellings of the same class reach this function: the TSA table's nullable-integer
    bus number (30), the service's group key parsed from the same column as a float ("30.0"),
    and the null that a stable record carries. Comparing the first two directly is what makes
    a perfectly good confusion matrix report zero accuracy, so both collapse to "30" here.

    Stable records are a real class rather than an absence: the archive stores a stable
    outcome as a row whose critical generator is null, the service groups those rows like any
    other, and its ranking can therefore name "stable" as the most likely outcome.
    """
    if value is None:
        return STABLE_LABEL
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return STABLE_LABEL
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isnan(number):
        return STABLE_LABEL
    return str(int(number)) if number.is_integer() else str(number)


def _reports_dir(dataset_name: str) -> Path:
    if dataset_name in MANUSCRIPT_DATASETS:
        return PROJECT_DIR / "results" / "data"
    return PROJECT_DIR / "datasets" / dataset_name / "reports"


def _process_state(service: Any, state_id: Any, state: pd.Series, tsa_subset: pd.DataFrame) -> list[dict[str, Any]]:
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
    reports = service.estimate_by_location(state=state_dict, exclude_uids=[str(state_id)])
    by_location = {normalize_label(loc): rep for loc, rep in reports.items()}

    excluded = normalize_label(state_id)
    for location, report in by_location.items():
        if excluded in {normalize_label(uid) for uid in report.included_state_ids}:
            raise RuntimeError(f"Data leakage: query state `{excluded}` retrieved at location `{location}`")

    rows: list[dict[str, Any]] = []
    for _, record in tsa_subset.iterrows():
        location = normalize_label(record["Location"])
        stable = pd.isna(record["Crit_gen"])
        true_label = _gen_label(None if stable else record["Crit_gen"])

        report = by_location.get(location)
        ranking = list(report.crit_gen_likelihood) if report is not None else []
        ranked = [_gen_label(gen) for gen in ranking]
        predicted = ranked[0] if ranked else None
        true_rank = ranked.index(true_label) + 1 if true_label in ranked else 0

        rows.append(
            {
                "state": str(state_id),
                "location": location,
                "cct_true": float(record["CCT"]),
                "true_label": true_label,
                "stable_record": bool(stable),
                "predicted_label": predicted,
                "covered": report is not None,
                "n_candidates": len(ranked),
                "true_rank": true_rank,
                "top1_score": float(report.crit_gen_likelihood[ranking[0]]) if ranking else np.nan,
            }
        )
    return rows


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    dataset = config.dataset_name
    slug = dataset.replace("/", "-")
    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    lf_path, _, _ = _dataset_paths(get_app_settings().data_dir, dataset, topology_variant=config.topology_variant)
    lf = pd.read_pickle(lf_path)
    tsa = pd.read_pickle(lf_path.parent / "tsa.pkl")
    tsa["state"] = tsa["state"].astype(str)

    if n_states and n_states < len(lf):
        rng = np.random.default_rng(42)
        keep = set(rng.choice(sorted(lf.index.astype(str)), size=n_states, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(keep)]
    logger.info(f"Dataset {dataset}: {len(lf)} query states, {len(tsa):,} TSA records")

    service = build_estimation_service()
    by_state = dict(tsa.groupby("state").__iter__())

    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        subset = by_state.get(str(state_id))
        if subset is None or subset.empty:
            continue
        rows.extend(_process_state(service, state_id, state, subset))
        if (i + 1) % 100 == 0:
            logger.info(f"{i + 1}/{len(lf)} states, {len(rows):,} records ({time.monotonic() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    logger.info(f"Done: {len(df):,} records from {len(lf)} states in {time.monotonic() - t0:.0f}s")

    suffix = f"-sample{n_states}" if n_states else ""
    joblib.dump(df.to_dict("records"), TMP_DIR / f"crit-gen-confusion-{slug}{suffix}.joblib")

    out_dir = _reports_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = df[df["covered"]]
    matrix = pd.crosstab(scored["true_label"], scored["predicted_label"], dropna=False)
    matrix.to_csv(out_dir / f"crit_gen_confusion_{slug}{suffix}.csv")

    # Row-normalized: each row sums to 1 and the diagonal is that class's recall. Counts alone
    # are hard to read when class support spans an order of magnitude, which it does on both
    # interscada archives, so a large diagonal can simply be a large class.
    relative = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    relative.round(4).to_csv(out_dir / f"crit_gen_confusion_relative_{slug}{suffix}.csv")

    per_class = pd.DataFrame(
        {
            "support": scored.groupby("true_label").size(),
            "recall": scored.groupby("true_label").apply(
                lambda g: float((g["predicted_label"] == g.name).mean()), include_groups=False
            ),
            "top3": scored.groupby("true_label").apply(
                lambda g: float(g["true_rank"].between(1, 3).mean()), include_groups=False
            ),
            "mean_candidates": scored.groupby("true_label")["n_candidates"].mean(),
        }
    )
    predicted_counts = scored.groupby("predicted_label").size()
    per_class["predicted_as"] = predicted_counts.reindex(per_class.index).fillna(0).astype(int)
    per_class["precision"] = [
        float((scored.loc[scored["predicted_label"] == label, "true_label"] == label).mean())
        if predicted_counts.get(label, 0)
        else np.nan
        for label in per_class.index
    ]
    # F1 as the harmonic mean of the two columns above. A class that is never predicted has an
    # undefined precision but a recall of zero, and scores 0 here rather than NaN, matching the
    # zero_division convention: it contributes to the macro average as the failure it is.
    precision = per_class["precision"].fillna(0.0)
    denominator = precision + per_class["recall"]
    per_class["f1"] = np.where(denominator > 0, 2 * precision * per_class["recall"] / denominator, 0.0)
    per_class.to_csv(out_dir / f"crit_gen_per_class_{slug}{suffix}.csv")

    # Macro treats every class alike, weighted scales by support. Report both: support spans an
    # order of magnitude on these archives, so the two answer different questions and a gap
    # between them is itself the finding.
    macro_f1 = float(per_class["f1"].mean())
    weighted_f1 = float((per_class["f1"] * per_class["support"]).sum() / per_class["support"].sum())

    overall = {
        "dataset": dataset,
        "n_states": len(lf),
        "n_records": len(df),
        "n_stable_records": int(df["stable_record"].sum()),
        "n_scored": len(scored),
        "top1_accuracy": float((scored["predicted_label"] == scored["true_label"]).mean()),
        "top3_accuracy": float(scored["true_rank"].between(1, 3).mean()),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top1_accuracy_unstable_only": float(
            (
                scored.loc[~scored["stable_record"], "predicted_label"]
                == scored.loc[~scored["stable_record"], "true_label"]
            ).mean()
        ),
        "top1_accuracy_stable_only": float(
            (
                scored.loc[scored["stable_record"], "predicted_label"]
                == scored.loc[scored["stable_record"], "true_label"]
            ).mean()
        ),
        "true_generator_absent_from_candidates": float((scored["true_rank"] == 0).mean()),
        "mean_candidates": float(scored["n_candidates"].mean()),
    }
    pd.DataFrame([overall]).to_csv(out_dir / f"crit_gen_summary_{slug}{suffix}.csv", index=False)

    print(pd.DataFrame([overall]).T.to_string(header=False))
    print("\nConfusion matrix (rows: recorded, columns: predicted)\n")
    print(matrix.to_string())
    print("\nPer-class\n")
    print(per_class.round(4).to_string())


if __name__ == "__main__":
    main()
