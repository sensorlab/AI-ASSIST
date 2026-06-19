from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import joblib
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
    LocationReport,
    LocationReportSummary,
    Report,
    ReportNeighbor,
    ReportSummary,
)
from src.domain.estimation.weights import K, query_distances
from src.preprocessing import AngleSinCos
from src.services.qdrant.client import create_qdrant_client
from src.services.qdrant.config import get_qdrant_config
from src.services.qdrant.repository import DatabaseQdrant, QueryResult


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


def _dataset_paths(data_dir: Path, dataset_name: str) -> tuple[Path, Path, Path]:
    base = data_dir / dataset_name / "interim"
    return (
        _resolve_dataset_file(base, "lf.pkl"),
        _resolve_dataset_file(base, "tsa.pkl"),
        _resolve_dataset_file(base, "topology_cols.joblib"),
    )


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
            # oserv_ ~= out of service
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


def _normalized_weights(distances: np.ndarray, *, crit_gen: str) -> np.ndarray:
    if distances.size == 0:
        raise ValueError(f"No distances found for crit_gen={crit_gen}")

    _ensure_finite("query_distances", distances, crit_gen=crit_gen)

    weights = K(distances)
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


def _neighborhood_compactness(X_neighbors: np.ndarray, *, crit_gen: str) -> float | None:
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

    pairwise_weights = K(pairwise_distances)
    _ensure_finite("pairwise_weights", pairwise_weights, crit_gen=crit_gen)

    compactness = float(pairwise_weights.mean())
    if not np.isfinite(compactness):
        raise ValueError(f"Non-finite neighborhood compactness for crit_gen={crit_gen}")

    return compactness


class EstimationService:
    def __init__(self, columns: list[str], scaler: Any, tsa: pd.DataFrame, db: DatabaseQdrant):
        self.columns = columns
        self.scaler = scaler
        self.tsa = tsa
        self.db = db

    def ensure_columns(self, request_cols: Iterable[str]) -> None:
        inputs_cols = set(self.columns)
        request_cols = set(request_cols)
        if request_cols != inputs_cols:
            invalid_cols = inputs_cols - request_cols
            missing_cols = request_cols - inputs_cols
            raise ValueError(f"invalid_fields={list(invalid_cols)}, missing_fields={list(missing_cols)}")

    def _query_enriched_neighbors(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
    ) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        # Normalize incoming state into the same scaled feature space used by the index.
        sample = pd.DataFrame([state]).astype(np.float64)
        sample = self.scaler.transform(sample)
        sample = cast(pd.DataFrame, sample)

        # Retrieve nearest neighbors from Qdrant and enrich them with TSA metadata.
        results = self.db.query(state=sample, exclude_source_index=exclude_uids)
        assert isinstance(results, QueryResult)

        lf_tsa = results.rows.merge(self.tsa, how="left", left_index=True, right_on="state")
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

        embed_cols = self.db.embed_cols
        if embed_cols is None:
            raise RuntimeError("Qdrant database is not fitted")

        X_query = sample[embed_cols].to_numpy(dtype=np.float64)
        return lf_tsa, embed_cols, X_query

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
    ) -> tuple[dict[str, Report], pd.DataFrame]:
        reports: dict[str, Report] = {}
        weighted_subsets: list[pd.DataFrame] = []

        for crit_gen_value, subset in lf_tsa.groupby(by="Crit_gen", dropna=False):
            crit_gen = str(crit_gen_value)
            subset = subset.copy()

            cct_neighbors = subset["CCT"].to_numpy(dtype=np.float64)
            X_neighbors = subset[embed_cols].to_numpy(dtype=np.float64)

            _ensure_finite("cct_neighbors", cct_neighbors, crit_gen=crit_gen)
            _ensure_finite("X_neighbors", X_neighbors, crit_gen=crit_gen)

            # Query-to-record distances define record influence for this query sample.
            qds = query_distances(X_query=X_query, X_neighbor=X_neighbors)
            # Global normalization across all simulation records in this Crit_gen subset.
            qw_norm = _normalized_weights(qds, crit_gen=crit_gen)

            # Normalized pairwise compactness within this Crit_gen group.
            compactness = _neighborhood_compactness(X_neighbors, crit_gen=crit_gen)

            subset["weight"] = qw_norm
            subset["distance"] = qds
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
                    location_weight_mass=location_weight_mass,
                    neighborhood_compactness=compactness,
                    n=int(X_neighbors.shape[0]),
                    # Effective number of contributing simulation records in this Crit_gen group,
                    # not the number of unique pre-fault states.
                    n_eff=_effective_sample_size(qw_norm),
                    distances=_distance_summary(qds),
                    location_counts=location_counts,
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
    ) -> dict[str, Report]:
        lf_tsa, embed_cols, X_query = self._query_enriched_neighbors(
            state=state,
            exclude_uids=exclude_uids,
        )
        if lf_tsa.empty:
            return {}

        reports, _ = self._reports_by_generator(lf_tsa, embed_cols, X_query)
        return reports

    def estimate_by_location(
        self,
        state: Mapping[str, float | None],
        exclude_uids: Iterable[str],
    ) -> dict[str, dict[str, LocationReport]]:
        lf_tsa, embed_cols, X_query = self._query_enriched_neighbors(
            state=state,
            exclude_uids=exclude_uids,
        )
        if lf_tsa.empty:
            return {}

        _, weighted_lf_tsa = self._reports_by_generator(lf_tsa, embed_cols, X_query)
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
                    weight_mass=weight_mass,
                    neighborhood_compactness=_neighborhood_compactness(
                        X_neighbors,
                        crit_gen=group_name,
                    ),
                    n=int(X_neighbors.shape[0]),
                    n_eff=_effective_sample_size(qw_norm),
                    distances=_distance_summary(qds),
                ),
                included_state_ids=self._included_state_ids(subset),
                per_neighbor=self._build_per_neighbor(subset),
            )

        return reports

    def estimate(self, state: Mapping[str, float | None], exclude_uids: Iterable[str]) -> dict[str, Report]:
        return self.estimate_by_generator(state=state, exclude_uids=exclude_uids)


def build_estimation_service() -> EstimationService:
    config = get_qdrant_config()
    app_settings = get_app_settings()
    path_lf_dataset, path_tsa_dataset, path_topology_cols = _dataset_paths(app_settings.data_dir, config.dataset_name)
    use_population_lock = config.url.strip().lower() != ":memory:"

    lf: pd.DataFrame = pd.read_pickle(path_lf_dataset)
    tsa: pd.DataFrame = pd.read_pickle(path_tsa_dataset)
    tsa["state"] = tsa["state"].astype("str")

    scaler: Any = _make_scaler_for_dataset(config.dataset_name)
    lf_scaled = cast(pd.DataFrame, scaler.fit_transform(lf))

    si_topo_cols: Iterable[str] = joblib.load(path_topology_cols)
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

    return EstimationService(columns=list(lf.columns), scaler=scaler, tsa=tsa, db=db)
