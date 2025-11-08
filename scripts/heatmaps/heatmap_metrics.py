import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import stats
import matplotlib.pyplot as plt
import warnings

from .file_loading import (
    load_group_index, read_dlc_csv, find_rat_files,
    BASE_PATH, H1_DIR, H2_DIR, H3_DIR, S1_DIR, S2_DIR, T_DIR
)
from .circle_estimation import estimate_circle_center

warnings.filterwarnings('ignore')

BIN_SIZE_PX = 5
SMOOTH_SIGMA = 1
MAX_DISTANCE_FROM_CENTER_PX = 175
BODY_PART = 'mid_mid'
INCLUDE_GROUPS = ['saline', 'muscimol']
SESSIONS = ['S1', 'S2', 'T']
OUTPUT_DIR = BASE_PATH / 'figures' / 'heatmaps' / 'histograms'


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


def compute_normalized_grid(bin_size: int, max_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    extent = max_distance
    
    x_min = -extent
    x_max = extent
    y_min = -extent
    y_max = extent
    
    x_min = np.floor(x_min / bin_size) * bin_size
    x_max = np.ceil(x_max / bin_size) * bin_size
    y_min = np.floor(y_min / bin_size) * bin_size
    y_max = np.ceil(y_max / bin_size) * bin_size
    
    x_edges = np.arange(x_min, x_max + bin_size, bin_size)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size)
    
    if len(x_edges) < 2:
        x_edges = np.array([x_min, x_min + bin_size])
    if len(y_edges) < 2:
        y_edges = np.array([y_min, y_min + bin_size])
    
    return x_edges, y_edges


def compute_histogram(data: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray, 
                     smooth_sigma: float = 0) -> np.ndarray:
    if data is None or len(data) == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    x = data['x'].values
    y = data['y'].values
    valid = np.isfinite(x) & np.isfinite(y)
    
    if not np.any(valid):
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1))
    
    counts, _, _ = np.histogram2d(y[valid], x[valid], bins=[y_edges, x_edges])
    
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            counts = gaussian_filter(counts, sigma=smooth_sigma)
        except ImportError:
            pass
    
    return counts


def compute_trajectory_length(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    
    diffs = np.diff(coordinates, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)


def compute_unique_locations(coordinates: np.ndarray, bin_size: float = 5.0) -> int:
    if len(coordinates) == 0:
        return 0
    
    binned_coords = np.round(coordinates / bin_size) * bin_size
    unique_locs = np.unique(binned_coords, axis=0)
    return len(unique_locs)


def compute_spatial_spread(coordinates: np.ndarray) -> float:
    if len(coordinates) == 0:
        return 0.0
    
    distances_from_center = np.sqrt(np.sum(coordinates**2, axis=1))
    return np.std(distances_from_center)


def compute_path_efficiency(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    
    total_distance = compute_trajectory_length(coordinates)
    straight_distance = np.sqrt(np.sum((coordinates[-1] - coordinates[0])**2))
    
    if total_distance == 0:
        return 0.0
    
    return straight_distance / total_distance


def compute_area_coverage(coordinates: np.ndarray) -> float:
    if len(coordinates) < 3:
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coordinates)
        return hull.volume
    except:
        x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
        y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
        return x_range * y_range


def compute_revisit_ratio(coordinates: np.ndarray, bin_size: float = 5.0) -> float:
    if len(coordinates) == 0:
        return 0.0
    
    binned_coords = np.round(coordinates / bin_size) * bin_size
    unique_locs = len(np.unique(binned_coords, axis=0))
    total_points = len(coordinates)
    
    if total_points == 0:
        return 0.0
    
    return 1.0 - (unique_locs / total_points)


def analyze_heatmap_metrics():
    print("Loading group index...")
    group_map = load_group_index()
    
    all_rat_ids = set()
    for directory in [H1_DIR, H2_DIR, H3_DIR, S1_DIR, S2_DIR, T_DIR]:
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
        
        filtered_data_dict = {}
        for session in SESSIONS:
            data = data_dict[session]
            if data is not None:
                filtered_data = filter_points_by_distance(
                    data, center_x, center_y, MAX_DISTANCE_FROM_CENTER_PX
                )
                if filtered_data is not None:
                    normalized_data = normalize_coordinates(filtered_data, center_x, center_y)
                    filtered_data_dict[session] = normalized_data
                else:
                    filtered_data_dict[session] = None
            else:
                filtered_data_dict[session] = None
        
        x_edges, y_edges = compute_normalized_grid(BIN_SIZE_PX, MAX_DISTANCE_FROM_CENTER_PX)
        
        for session in SESSIONS:
            data = filtered_data_dict[session]
            if data is None or len(data) == 0:
                continue
            
            coordinates = data[['x', 'y']].values
            
            if len(coordinates) < 2:
                continue
            
            trajectory_length = compute_trajectory_length(coordinates)
            unique_locations = compute_unique_locations(coordinates, bin_size=BIN_SIZE_PX)
            spatial_spread = compute_spatial_spread(coordinates)
            path_efficiency = compute_path_efficiency(coordinates)
            area_coverage = compute_area_coverage(coordinates)
            revisit_ratio = compute_revisit_ratio(coordinates, bin_size=BIN_SIZE_PX)
            
            results.append({
                'rat_id': rat_id,
                'group': group,
                'session': session,
                'trajectory_length': trajectory_length,
                'unique_locations': unique_locations,
                'spatial_spread': spatial_spread,
                'path_efficiency': path_efficiency,
                'area_coverage': area_coverage,
                'revisit_ratio': revisit_ratio
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
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
            
            print(f"\n  {group.capitalize()} (n={len(group_data)}):")
            print(f"    Trajectory length:  mean={group_data['trajectory_length'].mean():.2f}, "
                  f"std={group_data['trajectory_length'].std():.2f}")
            print(f"    Unique locations:    mean={group_data['unique_locations'].mean():.1f}, "
                  f"std={group_data['unique_locations'].std():.1f}")
            print(f"    Spatial spread:      mean={group_data['spatial_spread'].mean():.2f}, "
                  f"std={group_data['spatial_spread'].std():.2f}")
            print(f"    Path efficiency:     mean={group_data['path_efficiency'].mean():.3f}, "
                  f"std={group_data['path_efficiency'].std():.3f}")
            print(f"    Area coverage:       mean={group_data['area_coverage'].mean():.2f}, "
                  f"std={group_data['area_coverage'].std():.2f}")
            print(f"    Revisit ratio:       mean={group_data['revisit_ratio'].mean():.3f}, "
                  f"std={group_data['revisit_ratio'].std():.3f}")
        
        saline_data = session_data[session_data['group'] == 'saline']
        muscimol_data = session_data[session_data['group'] == 'muscimol']
        
        if len(saline_data) > 0 and len(muscimol_data) > 0:
            metrics_to_test = [
                ('trajectory_length', 'Trajectory length'),
                ('unique_locations', 'Unique locations'),
                ('spatial_spread', 'Spatial spread'),
                ('path_efficiency', 'Path efficiency'),
                ('area_coverage', 'Area coverage'),
                ('revisit_ratio', 'Revisit ratio')
            ]
            
            print(f"\n  Comparison (Mann-Whitney U test):")
            for metric_key, metric_name in metrics_to_test:
                stat, p = stats.mannwhitneyu(
                    saline_data[metric_key], muscimol_data[metric_key], alternative='two-sided'
                )
                print(f"    {metric_name}:  U={stat:.2f}, p={p:.4f}")
    
    output_path = BASE_PATH / 'data' / 'processed' / 'heatmap_metrics.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    
    print("\nGenerating histograms...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for session in SESSIONS:
        session_data = results_df[results_df['session'] == session]
        if len(session_data) == 0:
            continue
        
        saline_data = session_data[session_data['group'] == 'saline']
        muscimol_data = session_data[session_data['group'] == 'muscimol']
        
        if len(saline_data) == 0 and len(muscimol_data) == 0:
            continue
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        labels = ['Saline', 'Muscimol']
        colors = ['#3498db', '#e74c3c']
        
        metrics_to_plot = [
            ('trajectory_length', 'Trajectory Length (px)', 'Trajectory Length'),
            ('unique_locations', 'Unique Locations (count)', 'Unique Locations'),
            ('spatial_spread', 'Spatial Spread (px)', 'Spatial Spread'),
            ('path_efficiency', 'Path Efficiency', 'Path Efficiency'),
            ('area_coverage', 'Area Coverage (px²)', 'Area Coverage'),
            ('revisit_ratio', 'Revisit Ratio', 'Revisit Ratio')
        ]
        
        for idx, (metric_key, xlabel, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            saline_values = saline_data[metric_key].values if len(saline_data) > 0 else np.array([])
            muscimol_values = muscimol_data[metric_key].values if len(muscimol_data) > 0 else np.array([])
            
            data_to_plot = []
            labels_to_plot = []
            colors_to_plot = []
            
            if len(saline_values) > 0:
                data_to_plot.append(saline_values)
                labels_to_plot.append(labels[0])
                colors_to_plot.append(colors[0])
            
            if len(muscimol_values) > 0:
                data_to_plot.append(muscimol_values)
                labels_to_plot.append(labels[1])
                colors_to_plot.append(colors[1])
            
            if len(data_to_plot) > 0:
                ax.hist(data_to_plot, bins=15, alpha=0.7, label=labels_to_plot, 
                       color=colors_to_plot, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{session} - {title}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if len(saline_values) > 0 and len(muscimol_values) > 0:
                stat, p = stats.mannwhitneyu(
                    saline_values, muscimol_values, alternative='two-sided'
                )
                ax.text(0.02, 0.98, f'U={stat:.2f}, p={p:.4f}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        figure_path = OUTPUT_DIR / f'heatmap_metrics_{session}.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {figure_path}")
    
    print(f"\nAll histograms saved to: {OUTPUT_DIR}")
    
    return results_df


if __name__ == '__main__':
    results_df = analyze_heatmap_metrics()

