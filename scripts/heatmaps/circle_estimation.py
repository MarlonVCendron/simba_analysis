import numpy as np
import pandas as pd
from typing import Dict, Tuple


def fit_circle_least_squares(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 3:
        center_x = np.mean(x) if len(x) > 0 else 0.0
        center_y = np.mean(y) if len(y) > 0 else 0.0
        radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2)) if len(x) > 0 else 0.0
        return (center_x, center_y, radius)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    z = x_centered**2 + y_centered**2
    
    B = np.column_stack([2*x_centered, 2*y_centered, np.ones(len(x_centered))])
    
    u, residuals, rank, s = np.linalg.lstsq(B, z, rcond=None)
    
    center_x_centered = u[0]
    center_y_centered = u[1]
    radius_squared = center_x_centered**2 + center_y_centered**2 + u[2]
    
    if radius_squared < 0:
        center_x = x_mean
        center_y = y_mean
        radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2))
    else:
        center_x = center_x_centered + x_mean
        center_y = center_y_centered + y_mean
        radius = np.sqrt(radius_squared)
    
    return (center_x, center_y, radius)


def fit_circle_robust(x: np.ndarray, y: np.ndarray, 
                      outlier_threshold: float = 2.0,
                      max_iterations: int = 10) -> Tuple[float, float, float]:
    if len(x) < 3:
        center_x = np.mean(x) if len(x) > 0 else 0.0
        center_y = np.mean(y) if len(y) > 0 else 0.0
        radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2)) if len(x) > 0 else 0.0
        return (center_x, center_y, radius)
    
    x_current = x.copy()
    y_current = y.copy()
    
    center_x, center_y, radius = fit_circle_least_squares(x_current, y_current)
    
    for iteration in range(max_iterations):
        distances_to_center = np.sqrt((x_current - center_x)**2 + (y_current - center_y)**2)
        distances_to_circle = np.abs(distances_to_center - radius)
        
        mad = np.median(np.abs(distances_to_circle - np.median(distances_to_circle)))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (distances_to_circle - np.median(distances_to_circle)) / mad
        else:
            modified_z_scores = np.zeros_like(distances_to_circle)
        
        inlier_mask = np.abs(modified_z_scores) <= outlier_threshold
        
        if np.sum(inlier_mask) < 3:
            break
        
        if np.sum(inlier_mask) == len(x_current):
            break
        
        x_current = x_current[inlier_mask]
        y_current = y_current[inlier_mask]
        
        center_x, center_y, radius = fit_circle_least_squares(x_current, y_current)
    
    return (center_x, center_y, radius)


def estimate_circle_center(data_dict: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    all_x = []
    all_y = []
    
    for session_data in data_dict.values():
        if session_data is None or len(session_data) == 0:
            continue
        all_x.extend(session_data['x'].values)
        all_y.extend(session_data['y'].values)
    
    if len(all_x) == 0:
        return (0.0, 0.0)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    valid = np.isfinite(all_x) & np.isfinite(all_y)
    if not np.any(valid):
        return (0.0, 0.0)
    
    all_x = all_x[valid]
    all_y = all_y[valid]
    
    if len(all_x) < 3:
        center_x = np.mean(all_x)
        center_y = np.mean(all_y)
    else:
        center_x, center_y, _ = fit_circle_robust(all_x, all_y, outlier_threshold=2.0)
    
    return (center_x, center_y)

