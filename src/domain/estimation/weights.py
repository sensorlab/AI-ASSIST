import numpy as np


def K(distances: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.exp(-alpha * distances)
    # return 1.0 / (1.0 + distances**2) # t-student
    # return 1.0 / (1.0 + np.abs(distances))


def query_distances(X_query: np.ndarray, X_neighbor: np.ndarray) -> np.ndarray:
    if X_query.shape[-1] != X_neighbor.shape[-1]:
        raise ValueError(f"shape mismatch: {X_query.shape} vs {X_neighbor.shape}")
    return np.sqrt(np.sum((X_neighbor - X_query) ** 2, axis=1))


def cross_distances_efficient(X_query: np.ndarray, X_neighbor: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist

    if X_query.shape[-1] != X_neighbor.shape[-1]:
        raise ValueError(f"shape mismatch: {X_query.shape} vs {X_neighbor.shape}")
    X = np.vstack([X_query, X_neighbor])
    return pdist(X, metric="euclidean")
