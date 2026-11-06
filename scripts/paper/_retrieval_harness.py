"""Exact retrieval for ELES (and BUS39) with a pluggable compatibility rule, running the production estimator unchanged.

Two revision analyses need retrieval that the deployed Qdrant filter cannot express: the causal screening estimate
(the causal run stored only the selected generator's estimate, not the minimum over candidates) and alternative
compatibility rules (Hamming-relaxed, domestic-only, no filter). On ELES every query's candidate pool fits in memory
(4,393 states), so the neighbour search can be done exactly in NumPy.

Only the search is replaced. `BruteForceDatabase` is a drop-in for `DatabaseQdrant` (same `query()` signature, same
`QueryResult`), so `EstimationService.estimate_by_location()` runs as deployed: same fitted scaler, same Crit_gen and
location grouping, same kernel weights, same raw kernel mass for generator selection. Two production details are
reproduced deliberately: Qdrant stores vectors as float32 and returns them, so neighbour vectors are float32 upcast to
float64 while the query stays float64; and the query's topology bits come from the scaled row via `to_numpy(bool)`,
exactly as `DatabaseQdrant._get_topology_id` computes them.

BUS39 (analysis 06, R2.8) runs through the same path: `Rule.dataset` selects the corpus, and BUS39's single topology
file plays the deployed key's role (its flags are constant, so it never filters). `Rule.n_neighbors` sets K, which on
BUS39 binds. `scripts/paper/retrieval_harness_validation_bus39.py` checks the BUS39 path against production.

The production estimate for one record is reproduced by `evaluate_states()`, which merges the two per-record producers
`scripts/evaluation/eles_deoracled_bound.py` (selection, screening minimum, oracle) and
`scripts/evaluation/eles_generator_diagnostics_selected.py` (selected-generator diagnostics, causal exclusion).
`scripts/paper/retrieval_harness_validation.py` checks the result against their artifacts before anything else uses it.

Production weights are exp(-alpha * d) without rescaling. A group whose weights all underflow makes production either
raise (generator-level sum is zero) or fall back to an unweighted mean (location-level mass is zero). Relaxed rules
admit more distant neighbours, so both events are counted per query instead of being hidden by a rescaled kernel.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

import src.domain.estimation.service as service_module
from src.domain.estimation.service import EstimationService, _dataset_paths, _make_scaler_for_dataset
from src.domain.estimation.weights import query_distances
from src.services.qdrant.repository import RE_TOPO, QueryResult
from src.services.sqlite_store import SqliteRecordStore

logger = logging.getLogger(__name__)

DATASET: Final[str] = "eles/2026-06"
ALPHA: Final[float] = 1.0
N_NEIGHBORS: Final[int] = 100
QUERY_CHUNK: Final[int] = 256
DISTANCE_FEATURE: Final[str] = "_query_distance"
TIE_RTOL: Final[float] = 1e-12
# Topology keys available for ELES, by the name of their topology_cols_<variant>.json file.
KEYS: Final[tuple[str, ...]] = ("lines_only", "full", "slovenia_only")
# The key whose mismatches are reported for every rule: line status, the deployed key.
LINE_KEY: Final[str] = "lines_only"


@dataclass(frozen=True, slots=True)
class Rule:
    """A compatibility rule: candidates must differ from the query in at most `max_hamming` flags of `key`.

    `key=None` admits every state (no filter). `max_hamming=0` is exact equality on the key. `alpha` is the kernel
    bandwidth passed to production; 0 gives every neighbour weight 1 (uniform weights, analysis 03). `representation`
    names an alternative distance from _representations.py (analysis 05); None keeps production's distance.
    `n_neighbors` is K and `dataset` the corpus (analysis 06). `scaler_fit_states` fits the feature scaler on those
    states only instead of all of them (analysis 09, historically fitted preprocessing)."""

    name: str
    key: str | None
    max_hamming: int = 0
    alpha: float = ALPHA
    representation: str | None = None
    n_neighbors: int = N_NEIGHBORS
    dataset: str = DATASET
    scaler_fit_states: tuple[str, ...] | None = None


EXACT_LINES: Final[Rule] = Rule("exact_lines", "lines_only", 0)


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _tied(a: float, b: float) -> bool:
    """Kernel masses equal up to rounding. Production's choice among such generators follows its own last-bit noise,
    which depends on how pandas lays out each group, so the validation gate accepts any of them (TIE_RTOL)."""
    return abs(a - b) <= TIE_RTOL * max(abs(a), abs(b))


def _feature_map(topology_cols: Iterable[str], scaled_columns: Sequence[str]) -> list[tuple[str, str]]:
    """(raw, scaled) column pairs for a topology key, mapped as build_estimation_service() maps them.

    Sorted by the scaled name, as DatabaseQdrant sorts its significant topology columns. The order does not change
    Hamming counts; it keeps the per-flag mismatch table in production's column order."""
    mapped: list[tuple[str, str]] = []
    for col in topology_cols:
        candidates = [c for c in scaled_columns if c == col or c.startswith(col + "_")]
        if len(candidates) != 1:
            raise ValueError(f"cannot map topology col {col!r}: found {len(candidates)} candidates {candidates[:5]}")
        mapped.append((col, candidates[0]))
    return sorted(mapped, key=lambda pair: pair[1])


class BruteForceDatabase:
    """Exact nearest-neighbour search with the `DatabaseQdrant.query()` contract and a pluggable compatibility rule."""

    def __init__(self, lf_scaled: pd.DataFrame, key_columns: dict[str, list[str]], rule: Rule):
        self.embed_cols: list[str] = sorted(col for col in lf_scaled.columns if not RE_TOPO.match(col))
        self.default_limit: int = rule.n_neighbors
        self.index_name: str = lf_scaled.index.name or "_index"
        self.rule = rule
        self.key_columns = key_columns
        self.src_index: np.ndarray = lf_scaled.index.astype(str).to_numpy()
        self._position: dict[str, int] = {uid: i for i, uid in enumerate(self.src_index)}
        # Qdrant stores float32 and returns it upcast; holding the upcast copy avoids converting the corpus per query.
        vectors = lf_scaled.loc[:, self.embed_cols].to_numpy(dtype=np.float32).astype(np.float64)
        if not np.isfinite(vectors).all():
            raise ValueError("states contain non-finite values")
        self.vectors: np.ndarray = vectors
        self.bits: dict[str, np.ndarray] = {
            key: lf_scaled.loc[:, cols].to_numpy(dtype=bool) for key, cols in key_columns.items()
        }

    def query_bits(self, state: pd.DataFrame, key: str) -> np.ndarray:
        return state.iloc[0][self.key_columns[key]].to_numpy(dtype=bool)

    def hamming(self, state: pd.DataFrame, key: str) -> np.ndarray:
        return np.count_nonzero(self.bits[key] != self.query_bits(state, key)[None, :], axis=1)

    def positions(self, uids: Iterable[str]) -> np.ndarray:
        return np.fromiter((self._position[str(u)] for u in uids), dtype=np.int64)

    def query(
        self,
        state: pd.DataFrame,
        limit: int | None = None,
        exclude_source_index: Iterable[str] | str | None = None,
    ) -> QueryResult:
        if isinstance(exclude_source_index, (str, bytes)):
            exclude_source_index = [exclude_source_index]
        k = limit if limit is not None else self.default_limit

        allowed = np.ones(len(self.src_index), dtype=bool)
        if exclude_source_index:
            excluded = [self._position[str(u)] for u in exclude_source_index if str(u) in self._position]
            allowed[excluded] = False
        if self.rule.key is not None:
            allowed &= self.hamming(state, self.rule.key) <= self.rule.max_hamming

        candidates = np.flatnonzero(allowed)
        query = state.iloc[0][self.embed_cols].to_numpy(dtype=np.float32).astype(np.float64)
        sq = np.empty(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), QUERY_CHUNK):  # bounded temporaries; each row's sum is unchanged
            diff = self.vectors[candidates[start : start + QUERY_CHUNK]] - query
            sq[start : start + QUERY_CHUNK] = np.einsum("ij,ij->i", diff, diff)
        order = np.lexsort((candidates, sq))[:k]
        chosen = candidates[order]

        rows = pd.DataFrame(self.vectors[chosen], columns=self.embed_cols)
        rows.index = pd.Index(self.src_index[chosen], name=self.index_name)
        return QueryResult(scores=np.sqrt(sq[order]), rows=rows)


@dataclass(slots=True)
class Corpus:
    """Everything a worker needs that does not depend on the rule: scaled features, TSA store, raw frames."""

    lf: pd.DataFrame
    lf_scaled: pd.DataFrame
    scaler: Any
    tsa: SqliteRecordStore
    key_columns: dict[str, list[str]]  # scaled column names per topology key
    key_raw_columns: dict[str, list[str]]  # the same flags under their raw names, aligned with key_columns


def fit_scaler(scaler: Any, lf: pd.DataFrame, fit_states: Sequence[str] | None) -> pd.DataFrame:
    """Scale every state, fitting on all of them (production) or only on `fit_states`."""
    if fit_states is None:
        return cast(pd.DataFrame, scaler.fit_transform(lf))
    mask = lf.index.astype(str).isin(set(fit_states))
    if mask.sum() != len(set(fit_states)):
        raise ValueError(f"{len(set(fit_states)) - int(mask.sum())} fit states are not in the corpus")
    scaler.fit(lf[mask])
    return cast(pd.DataFrame, scaler.transform(lf))


def load_corpus(
    data_dir: Path | None = None, dataset: str = DATASET, fit_states: Sequence[str] | None = None
) -> Corpus:
    """Load and scale a dataset exactly as build_estimation_service() does (or fit the scaler on `fit_states`)."""
    data_dir = data_dir or Path(os.environ.get("DATA_DIR", "./datasets"))
    eles = dataset == DATASET
    lf_path, tsa_path, _ = _dataset_paths(data_dir, dataset, topology_variant=LINE_KEY if eles else None)
    lf = cast(pd.DataFrame, pd.read_pickle(lf_path))
    scaler = _make_scaler_for_dataset(dataset)
    lf_scaled = fit_scaler(scaler, lf, fit_states)
    scaler.n_jobs = None  # see build_estimation_service(): dispatch only, never the arithmetic

    key_columns: dict[str, list[str]] = {}
    key_raw_columns: dict[str, list[str]] = {}
    for key in KEYS if eles else (LINE_KEY,):  # BUS39 has one topology file, the deployed key
        _, _, cols_path = _dataset_paths(data_dir, dataset, topology_variant=key if eles else None)
        if eles and cols_path.stem != f"topology_cols_{key}":
            raise FileNotFoundError(f"no topology_cols_{key} file under {cols_path.parent}")
        pairs = _feature_map(json.loads(cols_path.read_text()), list(lf_scaled.columns))
        key_raw_columns[key] = [raw for raw, _ in pairs]
        key_columns[key] = [scaled for _, scaled in pairs]
    return Corpus(
        lf=lf,
        lf_scaled=lf_scaled,
        scaler=scaler,
        tsa=SqliteRecordStore(tsa_path, table="tsa"),
        key_columns=key_columns,
        key_raw_columns=key_raw_columns,
    )


def build_service(corpus: Corpus, rule: Rule) -> tuple[EstimationService, BruteForceDatabase]:
    # Neighbourhood compactness is a pairwise-distance diagnostic that no estimate, weight or kernel mass depends on,
    # and it costs about 80% of a query once K binds (no filter, wide Hamming). It is switched off in this process
    # only; retrieval_harness_validation.py runs with it off, so the gate covers the harness as used.
    service_module._neighborhood_compactness = lambda *args, **kwargs: None
    db = BruteForceDatabase(corpus.lf_scaled, corpus.key_columns, rule)
    service = EstimationService(
        columns=list(corpus.lf.columns),
        scaler=corpus.scaler,
        tsa=corpus.tsa,
        db=db,  # type: ignore[arg-type]  # duck-typed stand-in for DatabaseQdrant
    )
    production_query = service._query_neighbors
    alternative = None
    if rule.representation is not None:
        import _representations

        raw_columns = [c for c in corpus.lf.columns if not RE_TOPO.match(c)]
        alternative, _ = _representations.build(
            rule.representation, db.vectors, db.embed_cols, raw_columns, db.bits[LINE_KEY]
        )

    def narrow_query_neighbors(
        state: Any, exclude_uids: Iterable[str], n_neighbors: int | None = None
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        # Production merges all 11,452 features onto every retrieved record and copies that frame at each groupby,
        # which dominates a query once K binds. With compactness off, the features feed only the query distance, so
        # each neighbour's distance is computed once here with production's function and passed on as the only
        # feature against a zero query: sqrt(d * d) == d exactly in IEEE arithmetic, so weights are unchanged.
        rows, embed_cols, X_query = production_query(state, exclude_uids, n_neighbors)
        X_neighbors = rows[embed_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(X_neighbors).all():
            raise ValueError("retrieved neighbours contain non-finite values")
        if alternative is None:
            distance = query_distances(X_query=X_query, X_neighbor=X_neighbors)
        else:
            # A representation must not change which neighbours are retrieved, only their distances.
            if len(rows) >= db.default_limit:
                raise RuntimeError("K binds, so the neighbour set depends on the deployed distance")
            distance = alternative(X_query, X_neighbors)
        return pd.DataFrame({DISTANCE_FEATURE: distance}, index=rows.index), [DISTANCE_FEATURE], np.zeros((1, 1))

    service._query_neighbors = narrow_query_neighbors  # type: ignore[method-assign]

    # Retrieval and the TSA fetch do not depend on alpha, so a query evaluated at several alphas in a row (analysis
    # 06) reuses them: one entry, keyed on everything the enriched neighbours depend on. Production's weighting
    # writes onto copies, so the cached frame is never changed.
    enriched = service._query_enriched_neighbors
    last: dict[str, Any] = {}

    def cached_enriched_neighbors(
        state: Any, exclude_uids: Iterable[str], n_neighbors: int | None = None
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        key = (tuple(state.items()), tuple(exclude_uids), n_neighbors)
        if last.get("key") != key:
            last.update(key=key, value=enriched(state, exclude_uids, n_neighbors))
        return last["value"]

    service._query_enriched_neighbors = cached_enriched_neighbors  # type: ignore[method-assign]
    return service, db


def causal_exclusions(timestamps: pd.Series, lag_hours: float) -> dict[str, list[str]]:
    """Per query, every state not yet usable: recorded later than the query time minus the lag, plus the query itself.

    Same construction as eles_generator_diagnostics_selected.py (ELES_CAUSAL_LAG_HOURS)."""
    if lag_hours < 0:
        raise ValueError("lag_hours must be non-negative")
    order = timestamps.sort_values()
    times = order.to_numpy()
    uids = order.index.to_numpy()
    lag = np.timedelta64(int(lag_hours * 3600), "s")
    cutoff = np.searchsorted(times, times - lag, side="right")
    exclusions: dict[str, list[str]] = {}
    for i, uid in enumerate(uids):
        withheld = list(uids[cutoff[i] :])
        if uid not in withheld:
            withheld.append(uid)
        exclusions[str(uid)] = [str(u) for u in withheld]
    return exclusions


WorkItem = tuple[str, pd.DataFrame, list[str]]


def tsa_by_state(corpus: Corpus) -> dict[str, pd.DataFrame]:
    """TSA records per state in the order both production producers iterate them."""
    tsa = corpus.tsa.fetch(list(corpus.lf.index.astype(str)))
    return {str(state_id): subset.copy() for state_id, subset in tsa.groupby("state", observed=True)}


def evaluate_states(
    corpus: Corpus, rule: Rule, items: list[WorkItem], alphas: Sequence[float] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Per-record estimates, per-state retrieval facts and per-flag line mismatch counts for one rule.

    Records: `covered` means at least one candidate generator at the record's own location (the coverage both
    production producers use). `pred_selection` is the estimate of the generator with the largest raw kernel mass,
    `pred_screening_min` the minimum over candidate generators, `pred_oracle` the recorded generator's estimate.

    With `alphas`, every query is evaluated at each of them in turn (retrieval reused), and record and state rows
    carry an `alpha` column; without, `rule.alpha` alone and the rows are unchanged."""
    service, db = build_service(corpus, rule)
    records: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    n_line_flags = len(corpus.key_columns[LINE_KEY])
    flag_mismatch = np.zeros(n_line_flags, dtype=np.int64)

    row_of = {str(u): i for i, u in enumerate(corpus.lf.index)}
    alpha_list = list(alphas) if alphas is not None else [rule.alpha]
    for uid, tsa_subset, excluded_uids in items:
        state = {k: (None if pd.isna(v) else v) for k, v in corpus.lf.iloc[row_of[uid]].items()}
        for alpha in alpha_list:
            tag = {} if alphas is None else {"alpha": alpha}
            state_row: dict[str, Any] = {"state": uid, "query_error": None, "n_fallback_groups": 0, **tag}
            try:
                by_loc = service.estimate_by_location(
                    state=state, exclude_uids=excluded_uids, n_neighbors=rule.n_neighbors, alpha=alpha
                )
            except ValueError as exc:  # production raises when a generator group's kernel weights all underflow
                by_loc = {}
                state_row["query_error"] = str(exc)

            excluded = {_norm(u) for u in excluded_uids}
            pools: dict[str, dict[str, tuple[float, Any]]] = {}
            retrieved: set[str] = set()
            for loc_key, loc_group in by_loc.items():
                loc_pool = pools.setdefault(_norm(loc_key), {})
                retrieved.update(str(s) for s in loc_group.included_state_ids)
                for gen_key, report in loc_group.per_crit_gen.items():
                    if excluded & {_norm(s) for s in report.included_state_ids}:
                        raise RuntimeError(f"excluded states returned by retrieval for query {uid}")
                    if report.summary.stats.weight_mass <= 0.0:
                        state_row["n_fallback_groups"] += 1
                    mass = service._raw_kernel_mass(report.per_neighbor, alpha=alpha)
                    loc_pool[_norm(gen_key)] = (mass, report)

            if retrieved:
                # The query's own bits from the corpus row: the scaler is fitted on the corpus, so this equals the bits
                # of the re-scaled query without paying for a second transform.
                pos = db.positions(sorted(retrieved))
                differs = db.bits[LINE_KEY][pos] != db.bits[LINE_KEY][row_of[uid]][None, :]
                line_hamming = differs.sum(axis=1)
                if alpha == alpha_list[0]:  # the neighbour set does not depend on alpha; count it once
                    flag_mismatch += differs.sum(axis=0)
                state_row.update(
                    n_retrieved_states=len(pos),
                    n_retrieved_line_mismatch=int((line_hamming > 0).sum()),
                    mean_line_hamming=float(line_hamming.mean()),
                    max_line_hamming=int(line_hamming.max()),
                )
            else:
                state_row.update(n_retrieved_states=0, n_retrieved_line_mismatch=0)
            states.append(state_row)

            for record_ordinal, (_, rec) in enumerate(tsa_subset.iterrows()):
                loc_true, gen_true = _norm(rec["Location"]), _norm(rec["Crit_gen"])
                row: dict[str, Any] = {
                    "state": uid,
                    "record_ordinal": record_ordinal,
                    "cct_true": float(rec["CCT"]),
                    "loc_true": loc_true,
                    "gen_true": gen_true,
                    "covered": False,
                    **tag,
                }
                loc_pool = pools.get(loc_true)
                if loc_pool:
                    order = sorted(loc_pool, key=lambda g: loc_pool[g][0], reverse=True)
                    selected = loc_pool[order[0]][1]
                    stats = selected.summary.stats
                    masses = [loc_pool[g][0] for g in order]
                    row.update(
                        covered=True,
                        gens_tied_top="|".join(
                            sorted(g for g, m in zip(order, masses, strict=True) if _tied(m, masses[0]))
                        ),
                        mass_tie=any(_tied(a, b) for a, b in zip(masses[1:], masses[:-1], strict=True)),
                        n_candidate_gens=len(loc_pool),
                        gen_selected=order[0],
                        gen_true_rank=order.index(gen_true) + 1 if gen_true in loc_pool else -1,
                        pred_selection=float(selected.summary.cct_weighted),
                        pred_screening_min=min(float(r.summary.cct_weighted) for _, r in loc_pool.values()),
                        pred_oracle=float(loc_pool[gen_true][1].summary.cct_weighted) if gen_true in loc_pool else None,
                        sel_n_neighbors=stats.n,
                        sel_n_eff=stats.n_eff,
                        sel_distance_min=stats.distances.get("min"),
                    )
                records.append(row)
    return records, states, flag_mismatch


def run_rule(
    rule: Rule,
    exclusions: dict[str, list[str]] | None,
    *,
    n_jobs: int,
    limit: int = 0,
    alphas: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Evaluate every state of `rule.dataset` under one rule, in worker processes that each load the corpus once.

    `exclusions=None` is leave-one-state-out; otherwise the per-query withheld states (e.g. causal_exclusions()).
    `alphas` evaluates every query at each alpha, rows tagged by an `alpha` column (evaluate_states())."""
    import joblib

    corpus = load_corpus(dataset=rule.dataset, fit_states=rule.scaler_fit_states)
    by_state = tsa_by_state(corpus)
    # A state missing from `exclusions` is not evaluated (analysis 06 runs the inner states of an outer fold).
    uids = [str(u) for u in corpus.lf.index if str(u) in by_state and (exclusions is None or str(u) in exclusions)]
    if limit:
        uids = uids[:limit]
    work = [(uid, by_state[uid], exclusions[uid] if exclusions is not None else [uid]) for uid in uids]
    flag_names = corpus.key_raw_columns[LINE_KEY]
    del corpus

    chunks = [work[i::n_jobs] for i in range(n_jobs) if work[i::n_jobs]]
    results = joblib.Parallel(n_jobs=len(chunks))(joblib.delayed(_run_chunk)(rule, chunk, alphas) for chunk in chunks)
    records = pd.DataFrame([r for chunk_records, _, _ in results for r in chunk_records])
    states = pd.DataFrame([s for _, chunk_states, _ in results for s in chunk_states])
    mismatch = pd.Series(np.sum([m for _, _, m in results], axis=0), index=flag_names, name="n_retrieved_mismatch")
    order = {uid: i for i, uid in enumerate(uids)}
    by = [] if alphas is None else ["alpha"]
    records = records.sort_values(
        [*by, "state", "record_ordinal"], key=lambda s: s.map(order) if s.name == "state" else s
    )
    states = states.sort_values([*by, "state"], key=lambda s: s.map(order) if s.name == "state" else s)
    return records.reset_index(drop=True), states.reset_index(drop=True), mismatch


def _run_chunk(
    rule: Rule, items: list[WorkItem], alphas: Sequence[float] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    return evaluate_states(load_corpus(dataset=rule.dataset, fit_states=rule.scaler_fit_states), rule, items, alphas)
