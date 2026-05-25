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

    @field_validator("dataset_name", mode="before")
    @classmethod
    def _normalize_dataset_name(cls, value: str | None) -> str:
        return str(value or "bus39").strip().lower()

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
        return f"{self.collection_prefix}_{self.dataset_name}"


def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()
