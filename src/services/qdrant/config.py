from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    url: str = Field(default="127.0.0.1", alias="QDRANT_URL")
    api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    dataset_name: str = Field(default="bus39", alias="DATASET_NAME")
    collection_prefix: str = Field(default="states", alias="QDRANT_COLLECTION_PREFIX")
    collection_name_override: str | None = Field(default=None, alias="QDRANT_COLLECTION")
    prefer_grpc: bool = Field(default=True, alias="QDRANT_PREFER_GRPC")
    populate_lock_path: str = Field(default="/tmp/qdrant_populate.lock", alias="QDRANT_POPULATE_LOCK_PATH")
    populate_lock_timeout_seconds: float = Field(default=120.0, alias="QDRANT_POPULATE_LOCK_TIMEOUT")
    # QdrantClient's own default (unset -> a few seconds) is too short once per-query cost grows:
    # observed DEADLINE_EXCEEDED errors on eles/2026-01 under the v2 (per-column StandardScaler)
    # scaler, whose larger vectors (12,524 dims, up from 9,732 - VarianceThreshold no longer trims
    # zero-variance columns) combined with eles/2026-01's much larger same-topology candidate
    # pools (far less fragmented than eles/2026-06) push real query latency past the client's
    # short built-in default. Purely operational - does not change any computed value.
    client_timeout_seconds: float = Field(default=60.0, alias="QDRANT_CLIENT_TIMEOUT")
    topology_variant: str = Field(default="lines_only", alias="TOPOLOGY_VARIANT")
    # Bump whenever a make_scaler_*() formula changes in a way that alters stored vector values
    # or dimensionality - folded into collection_name below for the same reason topology_variant
    # is: DatabaseQdrant.fit(force=False) only (re)populates a collection that doesn't already
    # exist, so a stale collection built under an old scaler would otherwise silently keep
    # serving old-scaler vectors compared against new-scaler query vectors. v2: bus39/eles scalers
    # moved from hardcoded per-group range constants to per-column StandardScaler (2026-08-06). v3:
    # simplified further (2026-08-06) - dropped Sk_'s bespoke 1/(1+sqrt(X)) pre-transform and the
    # manual "_scaled" column-rename step, both no longer needed once every branch just imputes
    # then standardizes uniformly. Changes stored column names even where values are unchanged
    # (e.g. BUS39, where Sk_ was already constant either way), so still needs a fresh collection.
    # v4: applied the same v3-style fix to the two interscada scalers (2026-08-07) - they had the
    # same hardcoded-range-constant bug, worse than BUS39/ELES's pre-fix imbalance (angle branch
    # drove 99.55%/99.63% of total squared distance on pl/fr respectively). Global, not per-dataset,
    # so this also forces a one-time BUS39/ELES repopulation even though their formula is unchanged
    # - harmless, since it's a deterministic rebuild of identical vectors.
    scaler_version: str = Field(default="v4", alias="SCALER_VERSION")

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _normalize_dataset_name(cls, value: str | None) -> str:
        return str(value or "bus39").strip().lower()

    @field_validator("topology_variant", mode="before")
    @classmethod
    def _normalize_topology_variant(cls, value: str | None) -> str:
        return str(value or "lines_only").strip().lower()

    @field_validator("collection_prefix", mode="before")
    @classmethod
    def _normalize_collection_prefix(cls, value: str | None) -> str:
        return str(value or "states").strip()

    @field_validator("api_key", "collection_name_override", mode="before")
    @classmethod
    def _none_if_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        val = str(value).strip()
        return val or None

    @computed_field
    @property
    def collection_name(self) -> str:
        if self.collection_name_override:
            return self.collection_name_override
        safe_name = self.dataset_name.replace("/", "-")
        # topology_variant is always included, even for datasets with only one topology_cols
        # definition: DatabaseQdrant.fit(force=False) only (re)populates a collection that
        # doesn't exist yet, so a stale collection populated under a different variant would
        # otherwise silently keep serving topology_id payloads computed under the OLD variant.
        # Namespacing by variant makes that a self-healing "collection doesn't exist yet, create
        # it" case instead of a silent correctness bug. Qdrant is a rebuildable index, not a
        # source of truth, so the one-time repopulation cost is harmless.
        return f"{self.collection_prefix}_{safe_name}_{self.topology_variant}_{self.scaler_version}"


def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()
