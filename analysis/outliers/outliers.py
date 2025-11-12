import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import interp1d

def interpolate_outliers(
    x: np.ndarray,
    y: np.ndarray,
    outliers: np.ndarray,
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    x_interp = x.copy()
    y_interp = y.copy()
    
    if not np.any(outliers):
        return x_interp, y_interp
    
    indices = np.arange(len(x))
    non_outlier_indices = indices[~outliers]
    outlier_indices = indices[outliers]
    
    x_non_outlier = x[~outliers]
    y_non_outlier = y[~outliers]
    
    x_interp_func = interp1d(non_outlier_indices, x_non_outlier, kind=method, bounds_error=False, fill_value='extrapolate')
    y_interp_func = interp1d(non_outlier_indices, y_non_outlier, kind=method, bounds_error=False, fill_value='extrapolate')
    
    x_interp[outliers] = x_interp_func(outlier_indices)
    y_interp[outliers] = y_interp_func(outlier_indices)
    
    return x_interp, y_interp



def detect_outliers(
    values: np.ndarray,
    method: str = 'zscore',
    threshold: Optional[float] = None,
    factor: float = 3
) -> np.ndarray:
    if method == 'iqr':
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if threshold is None:
            threshold = q3 + factor * iqr
        return values > threshold
    elif method == 'zscore':
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        z_scores = np.abs((values - mean) / std)
        if threshold is None:
            z_threshold = 3.0
        else:
            z_threshold = threshold
        return z_scores > z_threshold
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

def calculate_coordinate_diffs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_diffs = np.diff(x)
    y_diffs = np.diff(y)
    x_diffs = np.concatenate([[0], x_diffs])
    y_diffs = np.concatenate([[0], y_diffs])
    distances = np.sqrt(x_diffs**2 + y_diffs**2)
    return x_diffs, y_diffs, distances
