import fcntl
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import joblib
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

RE_TOPO = re.compile(r".*oserv_.*", flags=re.IGNORECASE)
SRC_INDEX: Final[str] = "src_index"
N_THREADS: Final[int] = joblib.cpu_count()


@dataclass(frozen=True, kw_only=True, slots=True)
class QueryResult:
    scores: np.ndarray
    rows: pd.DataFrame


class DatabaseQdrant:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "states",
        subset_topology_cols: Iterable[str] | None = None,
        populate_lock_path: str = "/tmp/qdrant_populate.lock",
        populate_lock_timeout_seconds: float = 120.0,
        use_population_lock: bool = True,
    ):
        self.significant_topology_cols = sorted(subset_topology_cols) if subset_topology_cols is not None else None
        self.collection_name: Final[str] = collection_name
        self.distance: Final[models.Distance] = models.Distance.EUCLID
        self.topology_cols: list[str] | None = None
        self.embed_cols: list[str] | None = None
        self.default_limit: Final[int] = 100
        self._is_fitted: bool = False
        self._client: Final[QdrantClient] = client
        self.populate_lock_path: Final[Path] = Path(populate_lock_path)
        self.populate_lock_timeout_seconds: Final[float] = populate_lock_timeout_seconds
        self.use_population_lock: Final[bool] = use_population_lock

    def _acquire_population_lock(self):
        self.populate_lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.populate_lock_path.open("a+")
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except BlockingIOError as err:
                elapsed = time.monotonic() - start
                if elapsed >= self.populate_lock_timeout_seconds:
                    lock_file.close()
                    raise TimeoutError(
                        f"Timed out waiting for Qdrant population lock at {self.populate_lock_path}"
                    ) from err
                time.sleep(0.2)

    @property
    def columns(self) -> list[str]:
        assert self.topology_cols is not None
        assert self.embed_cols is not None
        return self.embed_cols + self.topology_cols

    def fit(self, X: pd.DataFrame, force: bool = False) -> Self:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("input `X` must be a pandas DataFrame")
        if X.empty:
            raise ValueError("input DataFrame must not be empty")

        columns = list(X.columns)
        self.embed_cols = sorted([col for col in columns if not RE_TOPO.match(col)])
        self.topology_cols = sorted([col for col in columns if RE_TOPO.match(col)])
        self.index_name = X.index.name or "_index"

        if not self.significant_topology_cols:
            self.significant_topology_cols = self.topology_cols

        vectors = X.loc[:, self.embed_cols].to_numpy(dtype=np.float32, copy=True)
        if np.isnan(vectors).any():
            raise ValueError("states contains NaN values; Qdrant vectors must be finite")

        ids: list[int] = []
        payload: list[dict[str, Any]] = []
        for idx, (src_idx, row) in enumerate(X.iterrows()):
            ids.append(idx)
            payload.append(
                {
                    SRC_INDEX: str(src_idx),
                    "topology": row[self.topology_cols].to_dict(),
                    "topology_id": self._get_topology_id(row),
                }
            )

        if self.use_population_lock:
            lock_file = self._acquire_population_lock()
            try:
                if force and self._client.collection_exists(self.collection_name):
                    self._client.delete_collection(self.collection_name)

                if not self._client.collection_exists(self.collection_name):
                    self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=vectors.shape[1], distance=self.distance, on_disk=False
                        ),
                    )
                    self._client.upload_collection(
                        collection_name=self.collection_name,
                        vectors=vectors,
                        payload=payload,
                        ids=ids,
                        parallel=N_THREADS,
                    )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
        else:
            if force and self._client.collection_exists(self.collection_name):
                self._client.delete_collection(self.collection_name)

            if not self._client.collection_exists(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=vectors.shape[1], distance=self.distance, on_disk=False),
                )
                self._client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=vectors,
                    payload=payload,
                    ids=ids,
                    parallel=N_THREADS,
                )

        self._is_fitted = True
        return self

    def _get_topology_id(self, row: pd.Series) -> str:
        assert self.significant_topology_cols
        vals = row[self.significant_topology_cols].to_numpy(dtype=bool)
        return "".join(np.where(vals, "1", "0"))

    def normalize_query_input(self, state: Any) -> pd.DataFrame:
        if isinstance(state, pd.DataFrame):
            qdf = state.copy()
        elif isinstance(state, pd.Series):
            qdf = state.to_frame().T
        elif isinstance(state, dict):
            qdf = pd.DataFrame([state])
        else:
            raise TypeError(f"state must be dict, Series, or DataFrame. Found {type(state)}")
        return qdf.loc[:, self.columns]

    def query(
        self,
        state: dict[str, Any] | pd.Series | pd.DataFrame,
        limit: int | None = None,
        exclude_source_index: Iterable[str] | str | None = None,
    ) -> QueryResult | list[QueryResult]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")

        if isinstance(exclude_source_index, (int, float, str, bytes)):
            exclude_source_index = {exclude_source_index}
        if isinstance(exclude_source_index, Iterable):
            exclude_source_index = set(exclude_source_index)

        k: int = limit if limit is not None else self.default_limit
        n: int = len(exclude_source_index) if exclude_source_index else 0

        qdf = self.normalize_query_input(state=state)
        out: list[QueryResult] = []
        for _, row in qdf.iterrows():
            assert self.embed_cols
            query = row[self.embed_cols].to_numpy(dtype=np.float32)
            topology_id = self._get_topology_id(row)
            query_filter = models.Filter(
                must=[models.FieldCondition(key="topology_id", match=models.MatchValue(value=topology_id))]
            )
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query,
                query_filter=query_filter,
                limit=k + n,
                with_payload=True,
                with_vectors=True,
            )
            points = response.points or []
            if exclude_source_index:
                points = [
                    p for p in points if p.payload is not None and p.payload.get(SRC_INDEX) not in exclude_source_index
                ][:k]
            scores = np.asarray([p.score for p in points], dtype=np.float64)
            vectors = [p.vector for p in points if p.vector is not None]
            payload_src_index = [
                str(p.payload[SRC_INDEX])
                for p in points
                if p.payload is not None and SRC_INDEX in p.payload and p.vector is not None
            ]
            rows = pd.DataFrame(vectors, columns=self.embed_cols)
            rows.index = payload_src_index
            rows.index.name = self.index_name
            out.append(QueryResult(scores=scores, rows=rows))
        return out[0] if len(out) == 1 else out
