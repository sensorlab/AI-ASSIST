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
    topology_variant: str = Field(default="lines_only", alias="TOPOLOGY_VARIANT")

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
        return f"{self.collection_prefix}_{safe_name}_{self.topology_variant}"


def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()
