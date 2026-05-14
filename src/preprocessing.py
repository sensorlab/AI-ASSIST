import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AngleSinCos(BaseEstimator, TransformerMixin):
    """Transform angle features into sin(angle) and cos(angle)."""

    def __init__(self, input_in_degrees=True):
        self.input_in_degrees = input_in_degrees

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if self.input_in_degrees:
            arr = np.deg2rad(arr)

        sin_ = np.sin(arr)
        cos_ = np.cos(arr)
        return np.concatenate([sin_, cos_], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
            if input_features is None:
                raise ValueError("input_features must be provided")

        sin_names = [f"{feat}_sin" for feat in input_features]
        cos_names = [f"{feat}_cos" for feat in input_features]
        return np.array(sin_names + cos_names, dtype=object)
