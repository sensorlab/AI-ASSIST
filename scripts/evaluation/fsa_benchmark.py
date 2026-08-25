"""Leave-one-state-out FSA benchmark, run in-process against the real service.

The FSA analog of eles_benchmark.py. For each query state, estimate_by_observed_generator()
is asked for the frequency-stability outcome of every (failed_gen, measured_gen) pair, and
each prediction is scored against the recorded value for that pair.

Coverage means the share of recorded pairs that receive an estimate. Unlike TSA, a retrieved
state legitimately may carry no FSA record for a given pair (see CLAUDE.md: FSA is inner-
merged for exactly this reason), so an uncovered pair is a normal outcome rather than an
error, and coverage is the first thing worth reporting.

Metric columns differ by dataset and are discovered from the FSA table rather than hardcoded:
eles/2026-06 carries minF/maxF/maxRoCoF, bus39 adds M1/M2/M3. Errors are reported per metric
because the three quantities are on different scales and a pooled error would be meaningless.

Point error alone cannot decide whether retrieval has any signal here, because the eles/2026-06
label is nearly constant: 5,676,538 of its 5,969,844 rows carry the identical minF of 0.999, so
"always predict 0.999" is exactly right 95.1% of the time and every pooled MAE is decided by the
rows where both arms are trivially correct. What varies is rare, so this script also scores each
metric as an event-detection problem against a baseline that uses no state information at all.

The baseline is the pair's own historical event rate, P(event | failed_gen, measured_gen), which
is the right no-state predictor rather than a per-pair median (that median is 0.999 for every
pair). It is computed leave-one-state-out in closed form, (E_pair - y_i) / (N_pair - 1), over the
whole FSA table rather than the query subsample, so subsampling the query side never changes it.
Retrieval's comparable output is the kernel-weighted share of its retrieved neighbors for that
pair that crossed the same threshold, read off FsaReport.per_neighbor.

Both arms are scored by average precision, not ROC-AUC, since prevalence is well below 1% and
ROC-AUC flatters a detector at that imbalance. The comparison is what earns the conclusion: the
events are state-dependent rather than pair-determined in this archive (no pair always fires, and
a pair that ever fires does so in about 0.1% of its states), so a pair-rate baseline is a genuine
competitor and beating it is evidence that retrieving similar operating states adds information.

The thresholds below are fixed defaults, not fitted to the data, and each is overridable with
FSA_EVENT_<METRIC> (e.g. FSA_EVENT_MINF=0.996). Metrics with no rule keep the regression view
only, which is why bus39's M1/M2/M3 are scored as errors and not as events. Note that on
eles/2026-06 no event here is operationally significant: frequency is in pu, so the archive's
worst minF of 0.993 is 49.65 Hz, above the first UFLS stage, and the 0.001 pu storage resolution
is 0.05 Hz. This measures whether retrieval tracks the archive's largest frequency excursions,
not whether it predicts frequency instability.

Run from the repository root, e.g.:

    DATASET_NAME=eles/2026-06 QDRANT_URL=":memory:" \\
        uv run python scripts/evaluation/fsa_benchmark.py [n_states] [n_jobs]

n_states subsamples the query side (default 200, 0 means every state) because ELES holds
5,969,844 FSA rows over 4,393 states, a median of 1,260 pairs per state, so a full sweep is
roughly eighteen times the work of the ELES TSA benchmark. The sample is drawn with
FSA_BENCHMARK_SAMPLE_SEED (default 42) and the seed is recorded in the output filename
whenever it is not the default, so two seeds cannot overwrite each other.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config.logging import configure_logging
from src.domain.estimation.service import build_estimation_service
from src.services.qdrant.config import get_qdrant_config

logger = logging.getLogger(__name__)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parents[2]
TMP_DIR: Final[Path] = PROJECT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)
# Deliberately not results/data. FSA is out of scope for the Scientific Reports manuscript
# (CLAUDE.md: "Analysis: TSA (CCT) only"), so these outputs are project-report material and
# belong in tmp/ with the other non-paper artifacts, not among the paper's evidence CSVs.

SAMPLE_SEED: Final[int] = int(os.environ.get("FSA_BENCHMARK_SAMPLE_SEED", "42"))
# Resamples for the paired state-clustered bootstrap on the retrieval-minus-baseline gap.
BOOTSTRAP_RESAMPLES: Final[int] = int(os.environ.get("FSA_BOOTSTRAP_RESAMPLES", "1000"))
BOOTSTRAP_SEED: Final[int] = int(os.environ.get("FSA_BOOTSTRAP_SEED", "42"))
# Identity columns in the FSA table; everything else is a measured metric.
FSA_KEY_COLUMNS: Final[frozenset[str]] = frozenset({"state", "failed_gen", "measured_gen"})

# Event thresholds per metric, as (comparison, value). Fixed rather than derived from the data,
# so a run on a subsample cannot move the definition of a positive. Measured prevalence on
# eles/2026-06: minF 0.536%, maxRoCoF 1.048%, maxF 0.045%.
FSA_EVENT_RULES: Final[dict[str, tuple[str, float]]] = {
    "minF": ("<=", 0.997),
    "maxF": (">=", 1.002),
    "maxRoCoF": (">=", 0.006),
}


def _metric_columns(fsa: pd.DataFrame) -> list[str]:
    """Metric columns, discovered rather than hardcoded, since the set is dataset-specific."""
    return [c for c in fsa.columns if c not in FSA_KEY_COLUMNS and pd.api.types.is_numeric_dtype(fsa[c])]


def _event_rules(metric_cols: list[str]) -> dict[str, tuple[str, float]]:
    """Rules for the metrics this dataset actually has, with env overrides applied."""
    rules: dict[str, tuple[str, float]] = {}
    for metric in metric_cols:
        if metric not in FSA_EVENT_RULES:
            continue
        comparison, default = FSA_EVENT_RULES[metric]
        override = os.environ.get(f"FSA_EVENT_{metric.upper()}")
        rules[metric] = (comparison, float(override) if override else default)
    return rules


def _is_event(value: float, comparison: str, threshold: float) -> bool:
    return value <= threshold if comparison == "<=" else value >= threshold


def _weighted_event_share(report: Any, metric: str, comparison: str, threshold: float) -> float | None:
    """Retrieval's detector: the kernel-weighted share of retrieved neighbors that crossed.

    Divides by the weight actually seen rather than assuming the weights sum to one, since
    neighbors missing this metric are skipped and would otherwise deflate every score.
    """
    total = 0.0
    crossed = 0.0
    for neighbor in report.per_neighbor:
        value = neighbor.metrics.get(metric)
        if value is None:
            continue
        total += neighbor.weight
        if _is_event(float(value), comparison, threshold):
            crossed += neighbor.weight
    return crossed / total if total > 0 else None


def _pair_event_rates(fsa: pd.DataFrame, rules: dict[str, tuple[str, float]]) -> dict[str, tuple[pd.DataFrame, float]]:
    """Per-pair event counts and sizes over the whole table, plus the global rate.

    Computed on every state, not the query subsample, so the baseline a record is scored
    against does not depend on how many states this run happened to sample.
    """
    rates: dict[str, tuple[pd.DataFrame, float]] = {}
    keys = ["failed_gen", "measured_gen"]
    for metric, (comparison, threshold) in rules.items():
        recorded = fsa[fsa[metric].notna()]
        values = recorded[metric].astype(float)
        events = values <= threshold if comparison == "<=" else values >= threshold
        grouped = pd.DataFrame({"n": 1, "e": events.astype(int)}, index=recorded.index)
        grouped[keys] = recorded[keys].astype(str)
        totals = grouped.groupby(keys)[["n", "e"]].sum()
        rates[metric] = (totals, float(events.mean()) if len(events) else 0.0)
    return rates


def _process_state(
    service: Any,
    state_id: Any,
    state: pd.Series,
    fsa_subset: pd.DataFrame,
    metric_cols: list[str],
    rules: dict[str, tuple[str, float]],
) -> list[dict[str, Any]]:
    """Score one query state's recorded FSA pairs against the service's estimates."""
    state_dict = {k: (None if pd.isna(v) else v) for k, v in state.items()}
    reports = service.estimate_by_observed_generator(state=state_dict, exclude_uids=[str(state_id)])

    rows: list[dict[str, Any]] = []
    for _, record in fsa_subset.iterrows():
        failed_gen = str(record["failed_gen"])
        measured_gen = str(record["measured_gen"])
        report = reports.get(measured_gen, {}).get(failed_gen)
        summary = report.summary if report is not None else None
        stats = summary.stats if summary is not None else None
        predicted = summary.metrics_weighted if summary is not None else {}

        row: dict[str, Any] = {
            "state": state_id,
            "failed_gen": failed_gen,
            "measured_gen": measured_gen,
            "covered": report is not None,
            "n_neighbors": stats.n if stats is not None else 0,
            "n_eff": stats.n_eff if stats is not None else None,
            "neighborhood_compactness": (stats.neighborhood_compactness if stats is not None else None),
            "n_unique_states": stats.n_unique_states if stats is not None else None,
            "distance_min": (stats.distances or {}).get("min") if stats is not None else None,
            "distance_mean": (stats.distances or {}).get("mean") if stats is not None else None,
        }
        for metric in metric_cols:
            true_value = record[metric]
            pred_value = predicted.get(metric)
            row[f"{metric}_true"] = None if pd.isna(true_value) else float(true_value)
            row[f"{metric}_pred"] = pred_value
            if metric in rules:
                comparison, threshold = rules[metric]
                row[f"{metric}_event_true"] = (
                    None if pd.isna(true_value) else _is_event(float(true_value), comparison, threshold)
                )
                row[f"{metric}_event_score"] = (
                    _weighted_event_share(report, metric, comparison, threshold) if report is not None else None
                )
        rows.append(row)
    return rows


def _summarize(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Per-metric error on the covered pairs, with the population each figure rests on."""
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_cols:
        true_col, pred_col = f"{metric}_true", f"{metric}_pred"
        scored = df[df["covered"] & df[true_col].notna() & df[pred_col].notna()]
        if scored.empty:
            summary_rows.append({"metric": metric, "n_scored": 0})
            continue
        error = (scored[true_col] - scored[pred_col]).abs()
        summary_rows.append(
            {
                "metric": metric,
                "n_scored": int(len(scored)),
                "coverage": (
                    float(len(scored) / n_recorded) if (n_recorded := int(df[true_col].notna().sum())) else 0.0
                ),
                "mae": float(error.mean()),
                "rmse": float(np.sqrt(((scored[true_col] - scored[pred_col]) ** 2).mean())),
                "median_ae": float(error.median()),
                "ae_q95": float(error.quantile(0.95)),
                "ae_q99": float(error.quantile(0.99)),
                "max_ae": float(error.max()),
                "true_mean": float(scored[true_col].mean()),
                "true_std": float(scored[true_col].std()),
                # Pooled MAE above is dominated by the near-constant bulk; this is the same
                # error restricted to the records whose true value crossed the event rule.
                "mae_events": (
                    float(error[scored[event_col].astype(bool)].mean())
                    if (event_col := f"{metric}_event_true") in scored.columns and scored[event_col].any()
                    else None
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def _attach_baseline(df: pd.DataFrame, pair_rates: dict[str, tuple[pd.DataFrame, float]]) -> None:
    """Add the leave-one-state-out pair event rate for every scored record, in place.

    (E_pair - y_i) / (N_pair - 1) removes the record being scored from its own baseline, the
    same exclusion retrieval gets through exclude_uids. Pairs seen in only one state fall back
    to the global rate, which at millions of rows is indistinguishable from its own leave-one-out
    correction.
    """
    if df.empty:
        return
    key = pd.MultiIndex.from_arrays([df["failed_gen"].astype(str), df["measured_gen"].astype(str)])
    for metric, (totals, global_rate) in pair_rates.items():
        label_col = f"{metric}_event_true"
        if label_col not in df.columns:
            continue
        labels = df[label_col]
        n = totals["n"].reindex(key).to_numpy(dtype=float)
        e = totals["e"].reindex(key).to_numpy(dtype=float)
        y = labels.fillna(False).to_numpy(dtype=float)
        loo = np.divide(e - y, n - 1, out=np.full(len(df), global_rate), where=n > 1)
        df[f"{metric}_base_rate"] = np.where(labels.isna().to_numpy(), np.nan, loo)


def _precision_at_n(labels: np.ndarray, scores: np.ndarray, n_positive: int) -> float:
    """Precision among the n_positive highest-scoring records, n_positive being the number of
    true events, so flagging that many is the matched-workload comparison and precision here
    equals recall."""
    if n_positive <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[:n_positive]
    return float(labels[order].mean())


def _bootstrap_ap_gap(
    labels: np.ndarray,
    retrieval: np.ndarray,
    baseline: np.ndarray,
    states: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Paired bootstrap on the average-precision gap, resampling whole states.

    Records within a state share an operating point and are not independent, so the resample
    unit is the state, matching how every other benchmark in this repository bootstraps. The
    two arms are scored on the same resample, so the gap is paired and its interval is much
    tighter than differencing two independent intervals would suggest.
    """
    order = np.argsort(states, kind="stable")
    unique_states, starts = np.unique(states[order], return_index=True)
    blocks = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    retrieval_aps: list[float] = []
    baseline_aps: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in picked])
        resampled = labels[idx]
        # A resample can miss every event at these prevalences; average precision is then
        # undefined and the draw is skipped rather than counted as a zero.
        if not resampled.any() or resampled.all():
            continue
        ap_retrieval = float(average_precision_score(resampled, retrieval[idx]))
        ap_baseline = float(average_precision_score(resampled, baseline[idx]))
        retrieval_aps.append(ap_retrieval)
        baseline_aps.append(ap_baseline)
        gaps.append(ap_retrieval - ap_baseline)
    if not gaps:
        return {}
    return {
        "n_states_resampled": int(len(unique_states)),
        "n_boot_used": len(gaps),
        "ap_retrieval_ci_low": float(np.percentile(retrieval_aps, 2.5)),
        "ap_retrieval_ci_high": float(np.percentile(retrieval_aps, 97.5)),
        "ap_pair_rate_ci_low": float(np.percentile(baseline_aps, 2.5)),
        "ap_pair_rate_ci_high": float(np.percentile(baseline_aps, 97.5)),
        "ap_gap": float(np.mean(gaps)),
        "ap_gap_ci_low": float(np.percentile(gaps, 2.5)),
        "ap_gap_ci_high": float(np.percentile(gaps, 97.5)),
        "ap_gap_excludes_zero": bool(np.percentile(gaps, 2.5) > 0.0 or np.percentile(gaps, 97.5) < 0.0),
    }


def _summarize_detection(df: pd.DataFrame, rules: dict[str, tuple[str, float]]) -> pd.DataFrame:
    """Retrieval against the pair-rate baseline, on the pairs where both produce a score."""
    summary_rows: list[dict[str, Any]] = []
    for metric, (comparison, threshold) in rules.items():
        label_col = f"{metric}_event_true"
        score_col = f"{metric}_event_score"
        base_col = f"{metric}_base_rate"
        if label_col not in df.columns:
            continue
        scored = df[df["covered"] & df[label_col].notna() & df[score_col].notna() & df[base_col].notna()]
        labels = scored[label_col].to_numpy(dtype=bool)
        n_events = int(labels.sum())
        row: dict[str, Any] = {
            "metric": metric,
            "rule": f"{comparison}{threshold:g}",
            "n_scored": int(len(scored)),
            "n_events": n_events,
            "prevalence": float(labels.mean()) if len(scored) else float("nan"),
        }
        # Average precision is undefined without both classes present, which a small query
        # subsample can easily produce for a sub-1% event.
        if n_events and n_events < len(scored):
            retrieval = scored[score_col].to_numpy(dtype=float)
            baseline = scored[base_col].to_numpy(dtype=float)
            prevalence = row["prevalence"]
            row["ap_retrieval"] = float(average_precision_score(labels, retrieval))
            row["ap_pair_rate"] = float(average_precision_score(labels, baseline))
            row["lift_retrieval"] = row["ap_retrieval"] / prevalence
            row["lift_pair_rate"] = row["ap_pair_rate"] / prevalence
            row["precision_at_n_retrieval"] = _precision_at_n(labels, retrieval, n_events)
            row["precision_at_n_pair_rate"] = _precision_at_n(labels, baseline, n_events)
            row.update(
                _bootstrap_ap_gap(
                    labels,
                    retrieval,
                    baseline,
                    scored["state"].astype(str).to_numpy(),
                    BOOTSTRAP_RESAMPLES,
                    BOOTSTRAP_SEED,
                )
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main() -> None:
    configure_logging()
    config = get_qdrant_config()
    dataset = config.dataset_name
    dataset_slug = dataset.replace("/", "-")

    n_states = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    fsa_path = PROJECT_DIR / "datasets" / dataset / "interim" / "fsa.pkl"
    lf_path = PROJECT_DIR / "datasets" / dataset / "interim" / "lf.pkl"
    if not fsa_path.is_file():
        raise SystemExit(f"{dataset} has no FSA data at {fsa_path}; nothing to benchmark.")

    lf = pd.read_pickle(lf_path)
    fsa = pd.read_pickle(fsa_path)
    metric_cols = _metric_columns(fsa)
    rules = _event_rules(metric_cols)
    logger.info(f"Dataset {dataset}: {len(fsa):,} FSA rows, metrics {metric_cols}")
    if rules:
        logger.info("Event rules: " + ", ".join(f"{m}{c}{t:g}" for m, (c, t) in rules.items()))
    else:
        logger.info("No event rules apply to these metrics; reporting point error only.")

    if n_states and n_states < len(lf):
        rng = np.random.default_rng(SAMPLE_SEED)
        sampled = set(rng.choice(sorted(lf.index.astype(str)), size=n_states, replace=False))
        lf = lf.loc[lf.index.astype(str).isin(sampled)]
        logger.info(f"Query-side subsample: {len(lf)} states (seed={SAMPLE_SEED})")
    else:
        logger.info(f"Full query side: {len(lf)} states")

    fsa["state"] = fsa["state"].astype(str)
    # Baseline built from every state, before the query side is subsampled.
    pair_rates = _pair_event_rates(fsa, rules)
    query_ids = set(lf.index.astype(str))
    fsa_by_state = {str(sid): subset for sid, subset in fsa[fsa["state"].isin(query_ids)].groupby("state")}

    logger.info("Building EstimationService (embedded Qdrant)...")
    t0 = time.monotonic()
    service = build_estimation_service()
    if service.fsa is None:
        raise SystemExit(f"{dataset} exposes no FSA store; the service would return HTTP 501.")
    logger.info(f"Service ready in {time.monotonic() - t0:.1f}s")

    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i, (state_id, state) in enumerate(lf.iterrows()):
        subset = fsa_by_state.get(str(state_id))
        if subset is None or subset.empty:
            continue
        rows.extend(_process_state(service, state_id, state, subset, metric_cols, rules))
        if (i + 1) % 25 == 0:
            logger.info(f"{i + 1}/{len(lf)} states, {len(rows):,} pairs ({time.monotonic() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    _attach_baseline(df, pair_rates)
    elapsed = time.monotonic() - t0
    logger.info(f"Done: {len(df):,} pairs from {len(lf)} states in {elapsed:.1f}s")
    logger.info(f"Pair coverage: {df['covered'].mean():.4f} ({int(df['covered'].sum()):,} of {len(df):,})")

    suffix = f"-sample{n_states}" if n_states else "-full"
    if SAMPLE_SEED != 42:
        suffix += f"-seed{SAMPLE_SEED}"
    # df, not rows: the leave-one-out baseline is attached to the frame, and an artifact
    # without it cannot reproduce the detection comparison.
    joblib.dump(df.to_dict("records"), TMP_DIR / f"report-fsa-{dataset_slug}{suffix}.joblib")
    logger.info(f"Saved per-pair report to {TMP_DIR / f'report-fsa-{dataset_slug}{suffix}.joblib'}")

    summary = _summarize(df, metric_cols)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "n_states", len(lf))
    summary.insert(2, "n_pairs", len(df))
    summary.insert(3, "pair_coverage", float(df["covered"].mean()))
    out_csv = TMP_DIR / f"fsa_benchmark_{dataset_slug}{suffix}.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Saved summary to {out_csv}")
    print(summary.to_string(index=False))

    if rules:
        detection = _summarize_detection(df, rules)
        detection.insert(0, "dataset", dataset)
        detection.insert(1, "n_states", len(lf))
        detection_csv = TMP_DIR / f"fsa_detection_{dataset_slug}{suffix}.csv"
        detection.to_csv(detection_csv, index=False)
        logger.info(f"Saved detection summary to {detection_csv}")
        print(detection.to_string(index=False))


if __name__ == "__main__":
    main()
