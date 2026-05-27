from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config.settings import get_app_settings
from src.domain.estimation.models import Report, ReportNeighbor, ReportSummary
from src.domain.estimation.weights import K, cross_distances_efficient, query_distances
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


def make_scaler_eles():
    def add_suffix_fn(transformer, input_features: list[str]):
        return np.array([f"{c}_scaled" for c in input_features], dtype=object)

    feature_scaler = make_column_transformer(
        (
            # voltages: (0.0, 3.1166)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                VarianceThreshold(threshold=0),  # remove feature with ZERO variance
                FunctionTransformer(lambda X: X / 3.0, feature_names_out=add_suffix_fn),
            ),
            make_column_selector(pattern=r"^U_"),
        ),
        (
            # electrical phases: (-180.0, 180.0)
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                AngleSinCos(input_in_degrees=True),  # phases to sin(x), cos(x); values in [-1, 1] range
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


def _make_scaler_for_dataset(dataset_name: str) -> Any:
    name = dataset_name.strip().lower()
    scaler_by_dataset = {
        "bus39": make_scaler_bus39,
        "eles-2026-01": make_scaler_eles,
        "eles": make_scaler_eles,
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


class EstimationService:
    def __init__(self, columns: list[str], scaler, tsa: pd.DataFrame, db: DatabaseQdrant):
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

    def estimate(self, state: Mapping[str, float | None], exclude_uids: Iterable[str]) -> dict[str, Report]:
        # 1) Normalize incoming state into the same scaled feature space used by the index.
        sample = pd.DataFrame([state]).astype(np.float64)
        sample = self.scaler.transform(sample)
        sample = cast(pd.DataFrame, sample)

        # 2) Retrieve nearest neighbors from Qdrant and enrich them with TSA metadata.
        results = self.db.query(state=sample, exclude_source_index=exclude_uids)
        assert isinstance(results, QueryResult)

        lf_tsa = results.rows.merge(self.tsa, how="left", left_index=True, right_on="state")
        embed_cols = self.db.embed_cols
        if embed_cols is None:
            raise RuntimeError("Qdrant database is not fitted")

        X_query = sample[embed_cols].to_numpy(dtype=np.float64)

        # 3) Build one report per critical generator (`Crit_gen`).
        reports: dict[str, Report] = {}
        for crit_gen, subset in lf_tsa.groupby(by="Crit_gen"):
            crit_gen = str(crit_gen)
            subset = subset.copy()
            cct_neighbors = subset["CCT"].to_numpy(dtype=np.float64)
            X_neighbors = subset[embed_cols].to_numpy(dtype=np.float64)

            _ensure_finite("cct_neighbors", cct_neighbors, crit_gen=crit_gen)
            _ensure_finite("X_neighbors", X_neighbors, crit_gen=crit_gen)

            # Query-to-neighbor distances define neighbor influence for this query sample.
            qds = query_distances(X_query=X_query, X_neighbor=X_neighbors)
            _ensure_finite("query_distances", qds, crit_gen=crit_gen)
            qw = K(qds)
            _ensure_finite("query_weights", qw, crit_gen=crit_gen)
            # Global normalization across all neighbors in this Crit_gen subset.
            qw_norm = qw / np.sum(qw)
            _ensure_finite("normalized_query_weights", qw_norm, crit_gen=crit_gen)

            # Cross-neighbor distances measure local neighborhood density/compactness.
            cds = cross_distances_efficient(X_query=X_query, X_neighbor=X_neighbors)
            _ensure_finite("cross_distances", cds, crit_gen=crit_gen)
            cw = K(cds)
            _ensure_finite("cross_weights", cw, crit_gen=crit_gen)

            # 4) Build per-neighbor payload used by downstream consumers.
            subset["weight"] = qw_norm
            included_state_ids = list(dict.fromkeys(str(state) for state in subset["state"].dropna()))
            per_neighbor: list[ReportNeighbor] = []
            location_counts: dict[str, int] = {}
            for item in subset[[*self.tsa.columns, "weight"]].to_dict(orient="records"):
                per_neighbor.append(
                    ReportNeighbor(
                        state=item["state"],
                        cct=item["CCT"],
                        location=item["Location"],
                        terminal=item["Terminal"],
                        type=item["Type"],
                        weight=item["weight"],
                    )
                )
                location = item["Location"]
                location_counts[location] = location_counts.get(location, 0) + 1

            # 5) Aggregate weighted CCT per location.
            # We re-normalize weights inside each location so each location gets its own
            # internally consistent weighted mean, independent of global location mass.
            weighted_cct_per_location: dict[str, float] = {}
            location_weight_mass: dict[str, float] = {}
            for location, g in subset.groupby("Location", dropna=False):
                loc = str(location)
                w = g["weight"].to_numpy(dtype=np.float64)
                c = g["CCT"].to_numpy(dtype=np.float64)

                w_sum = float(w.sum())
                # Keep global weight mass as context (how much this location contributed overall).
                location_weight_mass[loc] = w_sum
                if w_sum <= 0.0:
                    # Degenerate case fallback: avoid division by zero.
                    weighted_cct_per_location[loc] = float(c.mean())
                else:
                    weighted_cct_per_location[loc] = float(np.sum((w / w_sum) * c))

            # 6) Assemble summary metrics and sorted neighbor details.
            reports[crit_gen] = Report(
                summary=ReportSummary(
                    cct_weighted=float(np.sum(qw_norm * cct_neighbors)),
                    cct_weighted_per_location=weighted_cct_per_location,
                    location_weight_mass=location_weight_mass,
                    neighborhood_density=float(cw.sum()),
                    n=int(X_neighbors.shape[0]),
                    n_eff=float(1.0 / (qw_norm**2).sum()),
                    distances={
                        "min": float(cds.min()),
                        "mean": float(cds.mean()),
                        "median": float(np.median(cds)),
                        "spread": float(cds.max() - cds.min()),
                        "norm": float(qds.min() / (np.median(cds) + 1e-12)),
                    },
                    location_counts=location_counts,
                ),
                included_state_ids=included_state_ids,
                per_neighbor=sorted(per_neighbor, key=lambda x: x.weight, reverse=True),
            )
        return reports


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
        candidates = [c for c in lf_scaled.columns if col in c]
        if len(candidates) != 1:
            raise ValueError(f"cannot map topology col {col}")
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
