from qdrant_client import QdrantClient

from src.services.qdrant.config import QdrantConfig


def create_qdrant_client(config: QdrantConfig) -> QdrantClient:
    return QdrantClient(
        location=config.url,
        api_key=config.api_key,
        prefer_grpc=config.prefer_grpc,
    )
