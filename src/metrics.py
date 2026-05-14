import numpy as np


def median_absolute_deviation(y_true, y_pred) -> float:
    residuals = y_true - y_pred
    return float(np.median(np.abs(residuals - np.median(residuals))))
