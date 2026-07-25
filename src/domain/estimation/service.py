import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config.settings import get_app_settings
from src.domain.estimation.models import (
    FsaReport,
    FsaReportNeighbor,
    FsaReportSummary,
    LocationReport,
    LocationReportStats,
    LocationReportSummary,
    Report,
    ReportNeighbor,
    ReportStats,
    ReportSummary,
    SssaNeighbor,
    Stats,
)
from src.domain.estimation.weights import K, query_distances
from src.preprocessing import AngleSinCos
from src.services.qdrant.client import create_qdrant_client
from src.services.qdrant.config import get_qdrant_config
from src.services.qdrant.repository import DatabaseQdrant, QueryResult
from src.services.sqlite_store import SqliteRecordStore


def _resolve_dataset_file(base_dir: Path, base_name: str) -> Path:
    exact = base_dir / base_name
    if exact.exists():
        return exact

    candidates = sorted(base_dir.glob(f"{base_name}.*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Required dataset artifact not found in {base_dir}: expected '{base_name}' or '{base_name}.*'"
    )


def _resolve_optional_dataset_file(base_dir: Path, base_name: str) -> Path | None:
    """Like _resolve_dataset_file, but for artifacts not every dataset has (e.g. fsa.pkl,
    sssa.pkl) - returns None instead of raising when absent."""
    try:
        return _resolve_dataset_file(base_dir, base_name)
    except FileNotFoundError:
        return None


def _resolve_topology_cols_file(processed: Path, topology_variant: str | None) -> Path:
    """Resolve the topology_cols artifact, variant-aware. Datasets with multiple named
    variants (currently only eles/2026-06 - see its README's "Topology Variants" section)
    write topology_cols_<variant>.json/.joblib.z instead of a single topology_cols.json;
    datasets with only one definition (bus39, interscada/*, eles/2026-01) keep the old
    unversioned topology_cols.json. Try the variant-specific name first, fall back to the
    bare name so single-variant datasets need no changes at all."""
    if topology_variant:
        variant_specific = _resolve_optional_dataset_file(processed, f"topology_cols_{topology_variant}")
        if variant_specific is not None:
            return variant_specific
    return _resolve_dataset_file(processed, "topology_cols")


def _dataset_paths(
    data_dir: Path,
    dataset_name: str,
    *,
    topology_variant: str | None = None,
) -> tuple[Path, Path, Path]:
    # lf.pkl is an interim, analyst-facing artifact (also the bulk input used once at startup
    # to fit the scaler and populate Qdrant); tsa/topology_cols live under processed/ as the
    # formats EstimationService actually reads per request/at startup (indexed SQLite for tsa,
    # plain JSON for topology_cols - joblib copies of both still exist under interim/ for
    # analyst/notebook use). topology_variant selects among a dataset's named topology_cols
    # variants when it has more than one (see _resolve_topology_cols_file); None/absent falls
    # back to the single unversioned topology_cols.json every dataset originally had.
    interim = data_dir / dataset_name / "interim"
    processed = data_dir / dataset_name / "processed"
    return (
        _resolve_dataset_file(interim, "lf.pkl"),
        _resolve_dataset_file(processed, "tsa"),
        _resolve_topology_cols_file(processed, topology_variant),
    )


def _fsa_dataset_path(data_dir: Path, dataset_name: str) -> Path | None:
    processed = data_dir / dataset_name / "processed"
    return _resolve_optional_dataset_file(processed, "fsa")


def _sssa_dataset_path(data_dir: Path, dataset_name: str) -> Path | None:
    processed = data_dir / dataset_name / "processed"
    return _resolve_optional_dataset_file(processed, "sssa")


def make_scaler_eles() -> Any:
    def add_suffix_fn(transformer: Any, input_features: list[str]) -> np.ndarray:
        return np.array([f"{c}_scaled" for c in input_features], dtype=object)

    feature_scaler = make_column_transformer(
        (
            # voltages: (0.0, 3.1166)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 3.0, feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^U_"),
        ),
        (
            # electrical phases: (-180.0, 180.0)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                AngleSinCos(input_in_degrees=True),
                VarianceThreshold(threshold=0),  # remove feature with ZERO variance
                FunctionTransformer(lambda X: X * 20, feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^phi_"),
        ),
        (
            # scale all active powers (-7423.0, 14050.0)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),  # remove feature with ZERO variance
                FunctionTransformer(lambda X: X / 14_000, feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^P\d?_"),
        ),
        (
            # reactive power (-8654.47, 916.79)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),  # remove feature with ZERO variance
                FunctionTransformer(lambda X: X / 8_000, feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^Q\d?_"),
        ),
        (
            # short-circuit powers (0.0, 15996.28)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),  # remove feature with ZERO variance
                FunctionTransformer(lambda X: 1.0 / (1 + np.sqrt(X)), feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^Sk_"),
        ),
        (
            # oserv_ ~= in/out-of-service status flag. NOTE: for eles/2026-01 and eles/2026-06
            # specifically, empirical validation against power-flow columns showed the documented
            # "True ~= out of service" convention is backwards for LINES: True correlates with
            # nonzero power flow (in service/active) - this reversed-polarity reading is the one
            # to trust. For GENERATORS the same 100%-clean correlation (True <-> nonzero output)
            # was also observed, but is flagged as UNRESOLVED/possibly a partner-side export
            # artifact (oserv_Gen* may be derived FROM the power value rather than an independent
            # PowerFactory flag) - don't treat the generator case as confirmed the way the line
            # case is. See datasets/eles/2026-06/README.md "Topology Variants" section. Not
            # re-verified for bus39/interscada; don't assume this reversed polarity generalizes.
            # make_pipeline(
            #    SimpleImputer(strategy="constant", fill_value=True, keep_empty_features=True),
            #    # OneHotEncoder(drop="if_binary", sparse_output=False),
            # ),
            SimpleImputer(strategy="constant", fill_value=True, keep_empty_features=True),
            make_column_selector(pattern=r"^oserv_"),
        ),
        remainder="drop",  # "passthrough",
        n_jobs=-1,
        verbose=False,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return feature_scaler


def make_scaler_bus39() -> Any:
    def _add_suffix_fn(transformer, input_features: list[str]):
        return np.array([f"{c}_scaled" for c in input_features], dtype=object)

    return make_column_transformer(
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                FunctionTransformer(lambda X: X, feature_names_out=_add_suffix_fn),
            ),
            make_column_selector(pattern=r"^U_"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                AngleSinCos(input_in_degrees=True),
                FunctionTransformer(lambda X: X * 20, feature_names_out=_add_suffix_fn),
            ),
            make_column_selector(pattern=r"^phi_"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                FunctionTransformer(lambda X: X / 150.0, feature_names_out=_add_suffix_fn),
            ),
            make_column_selector(pattern=r"^P\d?_"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                FunctionTransformer(lambda X: X / 30.0, feature_names_out=_add_suffix_fn),
            ),
            make_column_selector(pattern=r"^Q\d?_"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                FunctionTransformer(lambda X: 1.0 / (1 + np.sqrt(X)), feature_names_out=_add_suffix_fn),
            ),
            make_column_selector(pattern=r"^Sk_"),
        ),
        (
            SimpleImputer(strategy="constant", fill_value=True, keep_empty_features=True),
            make_column_selector(pattern=r"^oserv_"),
        ),
        remainder="drop",
        n_jobs=-1,
        verbose=False,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_scaler_interscada_pl() -> Any:
    def _sf(transformer, input_features: list[str]) -> np.ndarray:
        return np.array([f"{c}_scaled" for c in input_features], dtype=object)

    return make_column_transformer(
        (
            # voltages: per-unit, range ~0.88–1.11 — pass through as-is
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                FunctionTransformer(lambda X: X, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*\[pu\]$"),
        ),
        (
            # angles: degrees, range ~-42 to +32
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                AngleSinCos(input_in_degrees=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X * 20, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*\[deg\]$"),
        ),
        (
            # generator active power: range 50–850 MW
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 850, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*Gen_P_MW"),
        ),
        (
            # generator reactive power: range -142 to +424 Mvar
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 450, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*Gen_Q_mvar"),
        ),
        (
            # load active power: range 0–1911 MW
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 2000, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*Load_P_MW"),
        ),
        (
            # load reactive power: range -134 to +465 Mvar
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 500, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r".*[Oo]ad_Q_mvar"),
        ),
        (
            # line and generator topology status (0/1)
            SimpleImputer(strategy="constant", fill_value=1, keep_empty_features=True),
            make_column_selector(pattern=r".*(?:line_status|gen_status)"),
        ),
        remainder="drop",
        n_jobs=-1,
        verbose=False,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_scaler_interscada_fr() -> Any:
    def _sf(transformer, input_features: list[str]) -> np.ndarray:
        return np.array([f"{c}_scaled" for c in input_features], dtype=object)

    return make_column_transformer(
        (
            # voltages: kV, range 380–429 kV
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 430, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^V_"),
        ),
        (
            # angles: degrees, range ~-42 to +62
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                AngleSinCos(input_in_degrees=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X * 20, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^angle_"),
        ),
        (
            # generator active power: range -1974 to +1224 MW
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 2000, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^Pgen_"),
        ),
        (
            # generator reactive power: range -489 to +944 Mvar
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 1000, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^Qgen_"),
        ),
        (
            # load active power: range -1912 to +2089 MW
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 2100, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^Pload_"),
        ),
        (
            # load reactive power: range -514 to +635 Mvar
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),
                FunctionTransformer(lambda X: X / 650, feature_names_out=_sf),
            ),
            make_column_selector(pattern=r"^Qload_"),
        ),
        (
            # line topology status (0/1); NaN-free but impute for safety
            SimpleImputer(strategy="constant", fill_value=1, keep_empty_features=True),
            make_column_selector(pattern=r"^[A-Z][^_]*$"),  # line names e.g. ALBERL71BATHI, no underscore
        ),
        remainder="drop",  # drops n_buses and any unrecognised cols
        n_jobs=-1,
        verbose=False,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _make_scaler_for_dataset(dataset_name: str) -> Any:
    name = dataset_name.strip().lower()
    scaler_by_dataset = {
        "bus39": make_scaler_bus39,
        "eles/2026-01": make_scaler_eles,
        "eles/2026-06": make_scaler_eles,
        "interscada/pl": make_scaler_interscada_pl,
        "interscada/fr": make_scaler_interscada_fr,
    }

    try:
        return scaler_by_dataset[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(scaler_by_dataset))
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. No scaler registered. Supported: {supported}"
        ) from exc


def _ensure_finite(name: str, values: np.ndarray, *, crit_gen: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in `{name}` for crit_gen={crit_gen}")


def _normalized_weights(distances: np.ndarray, *, crit_gen: str, alpha: float = 1.0) -> np.ndarray:
    if distances.size == 0:
        raise ValueError(f"No distances found for crit_gen={crit_gen}")

    _ensure_finite("query_distances", distances, crit_gen=crit_gen)

    weights = K(distances, alpha)
    _ensure_finite("query_weights", weights, crit_gen=crit_gen)

    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError(f"Invalid query weight sum for crit_gen={crit_gen}: {weight_sum}")

    normalized = weights / weight_sum
    _ensure_finite("normalized_query_weights", normalized, crit_gen=crit_gen)
    return normalized


def _effective_sample_size(weights: np.ndarray) -> float:
    """Effective number of contributing simulation records for normalized weights."""
    if weights.size == 0:
        raise ValueError("Effective sample size requires at least one weight")
    if not np.isfinite(weights).all():
        raise ValueError("Effective sample size weights must be finite")

    denominator = float(np.sum(weights**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"Invalid effective sample size denominator: {denominator}")

    return 1.0 / denominator


def _distance_summary(qds: np.ndarray) -> dict[str, float]:
    """Summarize query-to-record distances, not pairwise neighbor distances."""
    if qds.size == 0:
        raise ValueError("Distance summary requires at least one query distance")
    if not np.isfinite(qds).all():
        raise ValueError("Query distances must be finite")

    median = float(np.median(qds))
    return {
        "min": float(qds.min()),
        "mean": float(qds.mean()),
        "median": median,
        "spread": float(qds.max() - qds.min()),
        "norm": float(qds.min() / (median + 1e-12)),
    }


def _neighborhood_compactness(X_neighbors: np.ndarray, *, crit_gen: str, alpha: float = 1.0) -> float | None:
    """Normalized pairwise compactness within one critical-generator group.

    This is not a calibrated density estimate. It is the mean exponential
    kernel value over all unique off-diagonal pairs of retrieved records.
    """
    if X_neighbors.ndim != 2:
        raise ValueError(f"Expected 2D neighbor matrix for crit_gen={crit_gen}; got shape={X_neighbors.shape}")

    n = X_neighbors.shape[0]
    if n <= 1:
        return None

    _ensure_finite("X_neighbors", X_neighbors, crit_gen=crit_gen)

    pairwise_distances = pdist(X_neighbors, metric="euclidean")
    _ensure_finite("pairwise_distances", pairwise_distances, crit_gen=crit_gen)

    pairwise_weights = K(pairwise_distances, alpha)
    _ensure_finite("pairwise_weights", pairwise_weights, crit_gen=crit_gen)

    compactness = float(pairwise_weights.mean())
    if not np.isfinite(compactness):
        raise ValueError(f"Non-finite neighborhood compactness for crit_gen={crit_gen}")

    return compactness


class EstimationService:
    def __init__(
        self,
        columns: list[str],
        scaler: Any,
        tsa: SqliteRecordStore,
        db: DatabaseQdrant,
        fsa: SqliteRecordStore | None = None,
        sssa: SqliteRecordStore | None = None,
    ):
        self.columns = columns
        self.scaler = scaler
        self.tsa = tsa
        self.db = db
        # Not every dataset has FSA (frequency stability) data - None here means the active
        # dataset doesn't support it; estimate_by_contingency() raises NotImplementedError.
        self.fsa = fsa
        # Same for SSSA (small-signal stability) - None means the active dataset doesn't
        # support it; estimate_sssa_by_generator() raises NotImplementedError.
        self.sssa = sssa

    def ensure_columns(self, request_cols: Iterable[str]) -> None:
        inputs_cols = set(self.columns)
        request_cols = set(request_cols)
        if request_cols != inputs_cols:
            invalid_cols = inputs_cols - request_cols
            missing_cols = request_cols - inputs_cols
            raise ValueError(f"invalid_fields={list(invalid_cols)}, missing_fields={list(missing_cols)}")

    def _query_neighbors(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        """Scale the query state and retrieve nearest neighbors from Qdrant. Returns the
        raw LF neighbor rows (not yet merged with any target dataset), the embedding
        columns, and the query's own embedding vector. Shared by every estimate_* method -
        each merges the result with whichever target dataset (tsa/fsa) it needs.
        n_neighbors defaults to DatabaseQdrant.default_limit when not given."""
        sample = pd.DataFrame([state]).astype(np.float64)
        sample = self.scaler.transform(sample)
        sample = cast(pd.DataFrame, sample)

        results = self.db.query(state=sample, limit=n_neighbors, exclude_source_index=exclude_uids)
        assert isinstance(results, QueryResult)

        embed_cols = self.db.embed_cols
        if embed_cols is None:
            raise RuntimeError("Qdrant database is not fitted")

        X_query = sample[embed_cols].to_numpy(dtype=np.float64)
        return results.rows, embed_cols, X_query

    def _query_enriched_neighbors(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        # Retrieve nearest neighbors from Qdrant and enrich them with TSA metadata.
        rows, embed_cols, X_query = self._query_neighbors(state, exclude_uids, n_neighbors)

        tsa_subset = self.tsa.fetch(rows.index)
        lf_tsa = rows.merge(tsa_subset, how="left", left_index=True, right_on="state")
        if lf_tsa.empty:
            return lf_tsa, [], np.empty((1, 0), dtype=np.float64)

        if "CCT" not in lf_tsa.columns:
            raise ValueError("Merged TSA data does not contain required 'CCT' column")

        missing_tsa = lf_tsa["CCT"].isna()
        if missing_tsa.any():
            missing_states = lf_tsa.loc[missing_tsa, "state"].dropna().astype(str).unique().tolist()
            raise ValueError(
                "Retrieved states without matching TSA records: "
                f"{missing_states[:10]}" + ("..." if len(missing_states) > 10 else "")
            )

        return lf_tsa, embed_cols, X_query

    def _query_enriched_fsa_neighbors(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        """FSA analog of _query_enriched_neighbors. Unlike TSA (where every retrieved state
        must have a TSA record), a retrieved state legitimately may have no FSA coverage at
        all - some (failed_gen, measured_gen) pairs have no result for every state, and a
        given state may lack all of them. An inner join silently excludes such states rather
        than treating that as an error."""
        if self.fsa is None:
            raise NotImplementedError("This dataset does not provide FSA (frequency stability) data")

        rows, embed_cols, X_query = self._query_neighbors(state, exclude_uids, n_neighbors)

        fsa_subset = self.fsa.fetch(rows.index)
        lf_fsa = rows.merge(fsa_subset, how="inner", left_index=True, right_on="state")
        if lf_fsa.empty:
            return lf_fsa, [], np.empty((1, 0), dtype=np.float64)

        return lf_fsa, embed_cols, X_query

    def _query_enriched_sssa_neighbors(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        """SSSA analog of _query_enriched_fsa_neighbors. Coverage isn't universal per state -
        some states have no recorded SSSA modes at all - so an inner join silently excludes
        such states rather than treating that as an error, same as FSA."""
        if self.sssa is None:
            raise NotImplementedError("This dataset does not provide SSSA (small-signal stability) data")

        rows, embed_cols, X_query = self._query_neighbors(state, exclude_uids, n_neighbors)

        sssa_subset = self.sssa.fetch(rows.index)
        lf_sssa = rows.merge(sssa_subset, how="inner", left_index=True, right_on="state")
        if lf_sssa.empty:
            return lf_sssa, [], np.empty((1, 0), dtype=np.float64)

        return lf_sssa, embed_cols, X_query

    def _weight_group(
        self,
        subset: pd.DataFrame,
        embed_cols: list[str],
        X_query: np.ndarray,
        *,
        group_name: str,
        alpha: float = 1.0,
    ) -> tuple[pd.DataFrame, np.ndarray, float | None]:
        """Compute normalized query-distance weights and neighborhood compactness for one
        group, writing 'weight'/'distance' columns onto a copy of subset. Shared by every
        estimate_* method's per-group loop (estimate_by_location re-normalizes these within
        a tighter sub-group afterward rather than recomputing from scratch). `alpha` is the
        exponential-kernel decay rate (src/domain/estimation/weights.py::K); not exposed on
        the public API, only threaded through for internal sensitivity-analysis scripts."""
        subset = subset.copy()
        X_neighbors = subset[embed_cols].to_numpy(dtype=np.float64)
        _ensure_finite("X_neighbors", X_neighbors, crit_gen=group_name)

        qds = query_distances(X_query=X_query, X_neighbor=X_neighbors)
        qw_norm = _normalized_weights(qds, crit_gen=group_name, alpha=alpha)
        compactness = _neighborhood_compactness(X_neighbors, crit_gen=group_name, alpha=alpha)

        subset["weight"] = qw_norm
        subset["distance"] = qds
        return subset, qw_norm, compactness

    def _build_per_neighbor(self, subset: pd.DataFrame) -> list[ReportNeighbor]:
        per_neighbor: list[ReportNeighbor] = []
        for item in subset[[*self.tsa.columns, "weight", "distance"]].to_dict(orient="records"):
            per_neighbor.append(
                ReportNeighbor(
                    state=str(item["state"]),
                    cct=float(item["CCT"]),
                    location=str(item["Location"]),
                    terminal=str(item["Terminal"]) if item.get("Terminal") is not None else None,
                    type=str(item["Type"]) if item.get("Type") is not None else None,
                    weight=float(item["weight"]),
                    distance=float(item["distance"]),
                )
            )
        return sorted(per_neighbor, key=lambda x: x.weight, reverse=True)

    @staticmethod
    def _included_state_ids(subset: pd.DataFrame) -> list[str]:
        return list(dict.fromkeys(str(s) for s in subset["state"].dropna()))

    def _reports_by_generator(
        self,
        lf_tsa: pd.DataFrame,
        embed_cols: list[str],
        X_query: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> tuple[dict[str, Report], pd.DataFrame]:
        reports: dict[str, Report] = {}
        weighted_subsets: list[pd.DataFrame] = []

        for crit_gen_value, subset in lf_tsa.groupby(by="Crit_gen", dropna=False):
            crit_gen = str(crit_gen_value)
            subset, qw_norm, compactness = self._weight_group(
                subset, embed_cols, X_query, group_name=crit_gen, alpha=alpha
            )
            cct_neighbors = subset["CCT"].to_numpy(dtype=np.float64)
            X_neighbors = subset[embed_cols].to_numpy(dtype=np.float64)
            qds = subset["distance"].to_numpy(dtype=np.float64)
            _ensure_finite("cct_neighbors", cct_neighbors, crit_gen=crit_gen)

            weighted_subsets.append(subset)

            location_counts: dict[str, int] = {}
            for location in subset["Location"]:
                loc = str(location)
                location_counts[loc] = location_counts.get(loc, 0) + 1

            # Aggregate weighted CCT per location. We re-normalize weights inside
            # each location so every location gets an internally consistent mean.
            weighted_cct_per_location: dict[str, float] = {}
            location_weight_mass: dict[str, float] = {}
            for location, group in subset.groupby("Location", dropna=False):
                loc = str(location)
                w = group["weight"].to_numpy(dtype=np.float64)
                c = group["CCT"].to_numpy(dtype=np.float64)

                _ensure_finite("location_weights", w, crit_gen=crit_gen)
                _ensure_finite("location_cct", c, crit_gen=crit_gen)

                w_sum = float(w.sum())
                location_weight_mass[loc] = w_sum

                if not np.isfinite(w_sum) or w_sum <= 0.0:
                    weighted_cct_per_location[loc] = float(c.mean())
                else:
                    weighted_cct_per_location[loc] = float(np.sum((w / w_sum) * c))

            reports[crit_gen] = Report(
                summary=ReportSummary(
                    cct_weighted=float(np.sum(qw_norm * cct_neighbors)),
                    cct_weighted_per_location=weighted_cct_per_location,
                    stats=ReportStats(
                        location_weight_mass=location_weight_mass,
                        neighborhood_compactness=compactness,
                        n=int(X_neighbors.shape[0]),
                        # Effective number of contributing simulation records in this Crit_gen
                        # group, not the number of unique pre-fault states.
                        n_eff=_effective_sample_size(qw_norm),
                        n_unique_states=int(subset["state"].nunique()),
                        distances=_distance_summary(qds),
                        location_counts=location_counts,
                    ),
                ),
                included_state_ids=self._included_state_ids(subset),
                per_neighbor=self._build_per_neighbor(subset),
            )

        if not weighted_subsets:
            return reports, lf_tsa.iloc[0:0].copy()

        return reports, pd.concat(weighted_subsets, axis=0)

    def estimate_by_generator(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
        alpha: float = 1.0,
    ) -> dict[str, Report]:
        lf_tsa, embed_cols, X_query = self._query_enriched_neighbors(
            state=state,
            exclude_uids=exclude_uids,
            n_neighbors=n_neighbors,
        )
        if lf_tsa.empty:
            return {}

        reports, _ = self._reports_by_generator(lf_tsa, embed_cols, X_query, alpha=alpha)
        return reports

    def estimate_by_location(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
        alpha: float = 1.0,
    ) -> dict[str, dict[str, LocationReport]]:
        lf_tsa, embed_cols, X_query = self._query_enriched_neighbors(
            state=state,
            exclude_uids=exclude_uids,
            n_neighbors=n_neighbors,
        )
        if lf_tsa.empty:
            return {}

        _, weighted_lf_tsa = self._reports_by_generator(lf_tsa, embed_cols, X_query, alpha=alpha)
        reports: dict[str, dict[str, LocationReport]] = {}

        for (location_value, crit_gen_value), subset in weighted_lf_tsa.groupby(
            by=["Location", "Crit_gen"],
            dropna=False,
        ):
            location = str(location_value)
            crit_gen = str(crit_gen_value)
            group_name = f"{location}/{crit_gen}"
            subset = subset.copy()

            w = subset["weight"].to_numpy(dtype=np.float64)
            c = subset["CCT"].to_numpy(dtype=np.float64)
            qds = subset["distance"].to_numpy(dtype=np.float64)
            X_neighbors = subset[embed_cols].to_numpy(dtype=np.float64)

            _ensure_finite("location_generator_weights", w, crit_gen=group_name)
            _ensure_finite("location_generator_cct", c, crit_gen=group_name)
            _ensure_finite("location_generator_distances", qds, crit_gen=group_name)
            _ensure_finite("location_generator_neighbors", X_neighbors, crit_gen=group_name)

            weight_mass = float(w.sum())
            if not np.isfinite(weight_mass) or weight_mass <= 0.0:
                qw_norm = np.full(w.shape, 1.0 / w.size, dtype=np.float64)
                cct_weighted = float(c.mean())
            else:
                qw_norm = w / weight_mass
                cct_weighted = float(np.sum(qw_norm * c))

            _ensure_finite("location_generator_normalized_weights", qw_norm, crit_gen=group_name)
            subset["weight"] = qw_norm

            reports.setdefault(location, {})[crit_gen] = LocationReport(
                summary=LocationReportSummary(
                    cct_weighted=cct_weighted,
                    stats=LocationReportStats(
                        weight_mass=weight_mass,
                        neighborhood_compactness=_neighborhood_compactness(
                            X_neighbors,
                            crit_gen=group_name,
                            alpha=alpha,
                        ),
                        n=int(X_neighbors.shape[0]),
                        n_eff=_effective_sample_size(qw_norm),
                        n_unique_states=int(subset["state"].nunique()),
                        distances=_distance_summary(qds),
                    ),
                ),
                included_state_ids=self._included_state_ids(subset),
                per_neighbor=self._build_per_neighbor(subset),
            )

        return reports

    def _fsa_metric_cols(self) -> list[str]:
        assert self.fsa is not None
        return [c for c in self.fsa.columns if c not in ("state", "failed_gen", "measured_gen")]

    def _build_fsa_per_neighbor(self, subset: pd.DataFrame, metric_cols: list[str]) -> list[FsaReportNeighbor]:
        per_neighbor: list[FsaReportNeighbor] = []
        for item in subset[["state", *metric_cols, "weight", "distance"]].to_dict(orient="records"):
            per_neighbor.append(
                FsaReportNeighbor(
                    state=str(item["state"]),
                    metrics={m: float(item[m]) for m in metric_cols},
                    weight=float(item["weight"]),
                    distance=float(item["distance"]),
                )
            )
        return sorted(per_neighbor, key=lambda x: x.weight, reverse=True)

    def _fsa_reports_by_pair(
        self,
        lf_fsa: pd.DataFrame,
        embed_cols: list[str],
        X_query: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> dict[tuple[str, str], FsaReport]:
        """Compute one FsaReport per (failed_gen, measured_gen) pair - the shared
        computation behind both estimate_by_observed_generator and
        estimate_by_failed_generator, which just re-nest this same flat result differently."""
        metric_cols = self._fsa_metric_cols()
        reports: dict[tuple[str, str], FsaReport] = {}

        for (failed_gen_value, measured_gen_value), subset in lf_fsa.groupby(
            by=["failed_gen", "measured_gen"],
            dropna=False,
        ):
            failed_gen = str(failed_gen_value)
            measured_gen = str(measured_gen_value)
            group_name = f"{failed_gen}/{measured_gen}"

            subset, qw_norm, compactness = self._weight_group(
                subset, embed_cols, X_query, group_name=group_name, alpha=alpha
            )
            qds = subset["distance"].to_numpy(dtype=np.float64)

            metrics_weighted: dict[str, float] = {}
            for metric in metric_cols:
                values = subset[metric].to_numpy(dtype=np.float64)
                _ensure_finite(f"fsa_{metric}", values, crit_gen=group_name)
                metrics_weighted[metric] = float(np.sum(qw_norm * values))

            reports[(failed_gen, measured_gen)] = FsaReport(
                summary=FsaReportSummary(
                    metrics_weighted=metrics_weighted,
                    stats=Stats(
                        neighborhood_compactness=compactness,
                        n=int(len(subset)),
                        n_eff=_effective_sample_size(qw_norm),
                        n_unique_states=int(subset["state"].nunique()),
                        distances=_distance_summary(qds),
                    ),
                ),
                included_state_ids=self._included_state_ids(subset),
                per_neighbor=self._build_fsa_per_neighbor(subset, metric_cols),
            )

        return reports

    def estimate_by_observed_generator(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
        alpha: float = 1.0,
    ) -> dict[str, dict[str, FsaReport]]:
        """Primary FSA view: for each observed/measured generator, the frequency-stability
        outcome per failed generator - the FSA analog of estimate_by_generator. Raises
        NotImplementedError (mapped to HTTP 501 at the API layer) if the active dataset has
        no FSA data."""
        lf_fsa, embed_cols, X_query = self._query_enriched_fsa_neighbors(
            state=state, exclude_uids=exclude_uids, n_neighbors=n_neighbors
        )
        if lf_fsa.empty:
            return {}

        reports: dict[str, dict[str, FsaReport]] = {}
        for (failed_gen, measured_gen), report in self._fsa_reports_by_pair(
            lf_fsa, embed_cols, X_query, alpha=alpha
        ).items():
            reports.setdefault(measured_gen, {})[failed_gen] = report
        return reports

    def estimate_by_failed_generator(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
        alpha: float = 1.0,
    ) -> dict[str, dict[str, FsaReport]]:
        """Secondary FSA view: for each failed generator, the frequency-stability outcome
        per observed/measured generator - the FSA analog of estimate_by_location. Raises
        NotImplementedError (mapped to HTTP 501 at the API layer) if the active dataset has
        no FSA data."""
        lf_fsa, embed_cols, X_query = self._query_enriched_fsa_neighbors(
            state=state, exclude_uids=exclude_uids, n_neighbors=n_neighbors
        )
        if lf_fsa.empty:
            return {}

        reports: dict[str, dict[str, FsaReport]] = {}
        for (failed_gen, measured_gen), report in self._fsa_reports_by_pair(
            lf_fsa, embed_cols, X_query, alpha=alpha
        ).items():
            reports.setdefault(failed_gen, {})[measured_gen] = report
        return reports

    def _sssa_metric_cols(self) -> list[str]:
        assert self.sssa is not None
        return [c for c in self.sssa.columns if c not in ("state", "mode_id", "generator", "real_part", "imag_part")]

    @staticmethod
    def _transform_sssa_angles(metrics: dict[str, float]) -> dict[str, float]:
        """Passthrough placeholder for a future AngleSinCos-style treatment of SSSA's raw-degree
        angle columns (ConAng/ObsAng/ParAng and their state-variable-suffixed variants, e.g.
        ObsAng_speed - identified generically by "Ang" appearing in the column name, since the
        exact column set differs per dataset). Raw degrees have the same wraparound problem
        phi_Bus*_[deg] (LF) already gets AngleSinCos for - 179 and -179 degrees are 2 degrees
        apart, not ~358 - which only matters once something averages/compares these values
        directly. Nothing does yet: estimate_sssa_by_generator() only returns raw per-row
        values, so this is currently the identity function - a seam to fill in once an actual
        aggregation is built, not a fix being applied now."""
        return metrics

    def estimate_sssa_by_generator(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
    ) -> dict[str, list[SssaNeighbor]]:
        """Raw, unweighted SSSA retrieval grouped by generator - the only SSSA query exposed
        for now. mode_id is a per-state local identifier only (mode indices aren't comparable
        across operating states, per the data dictionary) and is never used as a grouping key -
        only generator is, since generator identity is comparable across states. No
        weighting/summary is computed here by design, pending domain input on what
        aggregation (if any) is wanted; every retrieved (state, mode_id, generator) row is
        returned as-is. Raises NotImplementedError (mapped to HTTP 501 at the API layer, if
        ever exposed there) if the active dataset has no SSSA data."""
        lf_sssa, embed_cols, X_query = self._query_enriched_sssa_neighbors(
            state=state, exclude_uids=exclude_uids, n_neighbors=n_neighbors
        )
        if lf_sssa.empty:
            return {}

        metric_cols = self._sssa_metric_cols()
        X_neighbors = lf_sssa[embed_cols].to_numpy(dtype=np.float64)
        _ensure_finite("X_neighbors", X_neighbors, crit_gen="sssa")

        lf_sssa = lf_sssa.copy()
        lf_sssa["distance"] = query_distances(X_query=X_query, X_neighbor=X_neighbors)

        reports: dict[str, list[SssaNeighbor]] = {}
        for generator_value, subset in lf_sssa.groupby("generator", dropna=False):
            generator = str(generator_value)
            subset = subset.sort_values("distance")
            reports[generator] = [
                SssaNeighbor(
                    state=str(item["state"]),
                    mode_id=int(item["mode_id"]),
                    real_part=float(item["real_part"]),
                    imag_part=float(item["imag_part"]),
                    metrics=self._transform_sssa_angles({m: float(item[m]) for m in metric_cols}),
                    distance=float(item["distance"]),
                )
                for item in subset[["state", "mode_id", "real_part", "imag_part", *metric_cols, "distance"]].to_dict(
                    orient="records"
                )
            ]

        return reports

    def estimate(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
        n_neighbors: int | None = None,
        alpha: float = 1.0,
    ) -> dict[str, Report]:
        return self.estimate_by_generator(state=state, exclude_uids=exclude_uids, n_neighbors=n_neighbors, alpha=alpha)


def build_estimation_service() -> EstimationService:
    config = get_qdrant_config()
    app_settings = get_app_settings()
    path_lf_dataset, path_tsa_dataset, path_topology_cols = _dataset_paths(
        app_settings.data_dir, config.dataset_name, topology_variant=config.topology_variant
    )
    path_fsa_dataset = _fsa_dataset_path(app_settings.data_dir, config.dataset_name)
    path_sssa_dataset = _sssa_dataset_path(app_settings.data_dir, config.dataset_name)
    use_population_lock = config.url.strip().lower() != ":memory:"

    lf: pd.DataFrame = pd.read_pickle(path_lf_dataset)
    tsa = SqliteRecordStore(path_tsa_dataset, table="tsa")

    fsa: SqliteRecordStore | None = None
    if path_fsa_dataset is not None:
        fsa = SqliteRecordStore(path_fsa_dataset, table="fsa")

    sssa: SqliteRecordStore | None = None
    if path_sssa_dataset is not None:
        sssa = SqliteRecordStore(path_sssa_dataset, table="sssa")

    scaler: Any = _make_scaler_for_dataset(config.dataset_name)
    lf_scaled = cast(pd.DataFrame, scaler.fit_transform(lf))

    si_topo_cols: Iterable[str] = json.loads(path_topology_cols.read_text())
    feature_map: dict[str, str] = {}
    for col in si_topo_cols:
        candidates = [c for c in lf_scaled.columns if c == col or c.startswith(col + "_")]
        if len(candidates) != 1:
            raise ValueError(f"cannot map topology col {col!r}: found {len(candidates)} candidates {candidates[:5]}")
        feature_map[col] = candidates[0]

    client = create_qdrant_client(config)
    db = DatabaseQdrant(
        client=client,
        collection_name=config.collection_name,
        subset_topology_cols=feature_map.values(),
        populate_lock_path=config.populate_lock_path,
        populate_lock_timeout_seconds=config.populate_lock_timeout_seconds,
        use_population_lock=use_population_lock,
    )
    db.fit(lf_scaled, force=False)

    return EstimationService(columns=list(lf.columns), scaler=scaler, tsa=tsa, db=db, fsa=fsa, sssa=sssa)
