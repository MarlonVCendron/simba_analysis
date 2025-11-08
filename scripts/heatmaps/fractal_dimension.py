import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy import stats
import warnings

from .file_loading import (
    load_group_index, read_dlc_csv, find_rat_files,
    BASE_PATH, S1_DIR, S2_DIR, T_DIR
)
from .circle_estimation import estimate_circle_center

warnings.filterwarnings('ignore')

BIN_SIZE_PX = 5
MAX_DISTANCE_FROM_CENTER_PX = 175
BODY_PART = 'mid_mid'
INCLUDE_GROUPS = ['saline', 'muscimol']
SESSIONS = ['S1', 'S2', 'T']


def filter_points_by_distance(data: pd.DataFrame, center_x: float, center_y: float,
                              max_distance: float) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data
    
    x = data['x'].values
    y = data['y'].values
    
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    mask = distances <= max_distance
    
    filtered_data = data[mask].copy()
    
    return filtered_data if len(filtered_data) > 0 else None


def normalize_coordinates(data: pd.DataFrame, center_x: float, center_y: float) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data
    
    normalized_data = data.copy()
    normalized_data['x'] = data['x'] - center_x
    normalized_data['y'] = data['y'] - center_y
    
    return normalized_data


def compute_trajectory_length(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    
    diffs = np.diff(coordinates, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)


def compute_fractal_dimension_box_counting(coordinates: np.ndarray, 
                                           min_box_size: float = 1.0, 
                                           max_box_size: float = None,
                                           num_scales: int = 20) -> Optional[float]:
    if len(coordinates) < 2:
        return None
    
    if coordinates.shape[1] != 2:
        return None
    
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
    
    span_x = x_max - x_min
    span_y = y_max - y_min
    max_span = max(span_x, span_y)
    
    if max_span <= 0:
        return None
    
    if max_box_size is None:
        max_box_size = max_span / 2
    
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num_scales)
    
    counts = []
    valid_scales = []
    
    for box_size in box_sizes:
        if box_size <= 0:
            continue
        
        num_boxes_x = int(np.ceil(span_x / box_size)) + 1
        num_boxes_y = int(np.ceil(span_y / box_size)) + 1
        
        if num_boxes_x <= 0 or num_boxes_y <= 0:
            continue
        
        box_grid = np.zeros((num_boxes_y, num_boxes_x), dtype=bool)
        
        for point in coordinates:
            idx_x = int((point[0] - x_min) / box_size)
            idx_y = int((point[1] - y_min) / box_size)
            
            idx_x = np.clip(idx_x, 0, num_boxes_x - 1)
            idx_y = np.clip(idx_y, 0, num_boxes_y - 1)
            
            box_grid[idx_y, idx_x] = True
        
        num_boxes_occupied = np.sum(box_grid)
        
        if num_boxes_occupied > 0:
            counts.append(num_boxes_occupied)
            valid_scales.append(box_size)
    
    if len(counts) < 3:
        return None
    
    log_scales = np.log10(valid_scales)
    log_counts = np.log10(counts)
    
    valid_mask = np.isfinite(log_scales) & np.isfinite(log_counts) & (log_counts > 0)
    
    if np.sum(valid_mask) < 3:
        return None
    
    log_scales = log_scales[valid_mask]
    log_counts = log_counts[valid_mask]
    
    slope, intercept = np.polyfit(log_scales, log_counts, 1)
    
    fractal_dimension = -slope
    
    return fractal_dimension


def compute_fractal_dimension_higuchi(coordinates: np.ndarray, k_max: int = 10) -> Optional[float]:
    if len(coordinates) < k_max * 2:
        return None
    
    if coordinates.shape[1] != 2:
        return None
    
    trajectory = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    
    if len(trajectory) == 0:
        return None
    
    L_k = []
    k_values = []
    
    for k in range(1, min(k_max + 1, len(trajectory) // 2)):
        L_k_sum = 0.0
        count = 0
        
        for m in range(k):
            indices = np.arange(m, len(trajectory), k)
            if len(indices) < 2:
                continue
            
            segment_lengths = trajectory[indices[:-1]]
            L_k_sum += np.sum(segment_lengths) * (len(trajectory) - 1) / ((len(indices) - 1) * k * k)
            count += 1
        
        if count > 0:
            L_k.append(L_k_sum / count)
            k_values.append(k)
    
    if len(L_k) < 3:
        return None
    
    log_k = np.log10(k_values)
    log_L = np.log10(L_k)
    
    valid_mask = np.isfinite(log_k) & np.isfinite(log_L) & (log_L > 0)
    
    if np.sum(valid_mask) < 3:
        return None
    
    log_k = log_k[valid_mask]
    log_L = log_L[valid_mask]
    
    slope, _ = np.polyfit(log_k, log_L, 1)
    
    fractal_dimension = 2 - slope
    
    return fractal_dimension


def analyze_fractal_dimension():
    print("Loading group index...")
    group_map = load_group_index()
    
    all_rat_ids = set()
    for directory in [S1_DIR, S2_DIR, T_DIR]:
        for csv_file in directory.glob("R*G*.csv"):
            if 'original_filename' in str(csv_file):
                continue
            match = re.match(r'(R\d+G\d+)', csv_file.stem)
            if match:
                all_rat_ids.add(match.group(1))
    
    filtered_rat_ids = []
    for rat_id in sorted(all_rat_ids):
        match = re.match(r'R(\d+)G(\d+)', rat_id)
        if not match:
            continue
        
        if rat_id not in group_map:
            continue
        group = group_map[rat_id]
        if group not in INCLUDE_GROUPS:
            continue
        
        filtered_rat_ids.append(rat_id)
    
    print(f"Found {len(filtered_rat_ids)} rats to analyze")
    
    results = []
    
    for rat_id in filtered_rat_ids:
        group = group_map.get(rat_id, 'unknown')
        print(f"\nProcessing {rat_id} ({group})...")
        
        file_paths = find_rat_files(rat_id)
        
        data_dict = {}
        for session in SESSIONS:
            filepath = file_paths[session]
            if filepath and filepath.exists():
                data = read_dlc_csv(filepath, BODY_PART)
                data_dict[session] = data
            else:
                data_dict[session] = None
        
        center_x, center_y = estimate_circle_center(data_dict)
        
        for session in SESSIONS:
            data = data_dict[session]
            if data is None or len(data) == 0:
                continue
            
            filtered_data = filter_points_by_distance(
                data, center_x, center_y, MAX_DISTANCE_FROM_CENTER_PX
            )
            
            if filtered_data is None or len(filtered_data) < 10:
                continue
            
            normalized_data = normalize_coordinates(filtered_data, center_x, center_y)
            
            coordinates = normalized_data[['x', 'y']].values
            
            trajectory_length = compute_trajectory_length(coordinates)
            
            fd_box = compute_fractal_dimension_box_counting(coordinates)
            fd_higuchi = compute_fractal_dimension_higuchi(coordinates)
            
            results.append({
                'rat_id': rat_id,
                'group': group,
                'session': session,
                'trajectory_length': trajectory_length,
                'fractal_dimension_box': fd_box,
                'fractal_dimension_higuchi': fd_higuchi,
                'num_points': len(coordinates)
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("FRACTAL DIMENSION ANALYSIS")
    print("="*80)
    
    for session in SESSIONS:
        session_data = results_df[results_df['session'] == session]
        if len(session_data) == 0:
            continue
        
        print(f"\n{session}:")
        print("-" * 80)
        
        for group in ['saline', 'muscimol']:
            group_data = session_data[session_data['group'] == group]
            if len(group_data) == 0:
                continue
            
            box_data = group_data[group_data['fractal_dimension_box'].notna()]
            higuchi_data = group_data[group_data['fractal_dimension_higuchi'].notna()]
            
            print(f"\n  {group.capitalize()} (n={len(group_data)}):")
            
            if len(box_data) > 0:
                print(f"    Box-counting FD:  mean={box_data['fractal_dimension_box'].mean():.3f}, "
                      f"std={box_data['fractal_dimension_box'].std():.3f}, "
                      f"range=[{box_data['fractal_dimension_box'].min():.3f}, "
                      f"{box_data['fractal_dimension_box'].max():.3f}], "
                      f"n_valid={len(box_data)}")
            
            if len(higuchi_data) > 0:
                print(f"    Higuchi FD:        mean={higuchi_data['fractal_dimension_higuchi'].mean():.3f}, "
                      f"std={higuchi_data['fractal_dimension_higuchi'].std():.3f}, "
                      f"range=[{higuchi_data['fractal_dimension_higuchi'].min():.3f}, "
                      f"{higuchi_data['fractal_dimension_higuchi'].max():.3f}], "
                      f"n_valid={len(higuchi_data)}")
            
            print(f"    Trajectory length: mean={group_data['trajectory_length'].mean():.2f}, "
                  f"std={group_data['trajectory_length'].std():.2f}")
        
        saline_data = session_data[session_data['group'] == 'saline']
        muscimol_data = session_data[session_data['group'] == 'muscimol']
        
        if len(saline_data) > 0 and len(muscimol_data) > 0:
            saline_box = saline_data[saline_data['fractal_dimension_box'].notna()]['fractal_dimension_box']
            muscimol_box = muscimol_data[muscimol_data['fractal_dimension_box'].notna()]['fractal_dimension_box']
            
            saline_higuchi = saline_data[saline_data['fractal_dimension_higuchi'].notna()]['fractal_dimension_higuchi']
            muscimol_higuchi = muscimol_data[muscimol_data['fractal_dimension_higuchi'].notna()]['fractal_dimension_higuchi']
            
            print(f"\n  Comparison (Mann-Whitney U test):")
            
            if len(saline_box) > 0 and len(muscimol_box) > 0:
                box_stat, box_p = stats.mannwhitneyu(
                    saline_box, muscimol_box, alternative='two-sided'
                )
                print(f"    Box-counting FD:   U={box_stat:.2f}, p={box_p:.4f}")
            
            if len(saline_higuchi) > 0 and len(muscimol_higuchi) > 0:
                higuchi_stat, higuchi_p = stats.mannwhitneyu(
                    saline_higuchi, muscimol_higuchi, alternative='two-sided'
                )
                print(f"    Higuchi FD:         U={higuchi_stat:.2f}, p={higuchi_p:.4f}")
            
            length_stat, length_p = stats.mannwhitneyu(
                saline_data['trajectory_length'], muscimol_data['trajectory_length'], alternative='two-sided'
            )
            print(f"    Trajectory length:  U={length_stat:.2f}, p={length_p:.4f}")
    
    output_path = BASE_PATH / 'data' / 'processed' / 'fractal_dimension.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    
    return results_df


if __name__ == '__main__':
    results_df = analyze_fractal_dimension()

