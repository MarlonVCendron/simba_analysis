import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
import re
import ast

from .file_loading import (
    load_group_index, load_polygons, read_dlc_csv, find_rat_files,
    BASE_PATH, H1_DIR, H2_DIR, H3_DIR, S1_DIR, S2_DIR, T_DIR
)
from .circle_estimation import estimate_circle_center

warnings.filterwarnings('ignore')

OUTPUT_DIR = BASE_PATH / 'figures' / 'heatmaps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_SIZE_PX = 5
COLORMAP_NAME = 'turbo'
SHOW_COLORBAR = True
INVERT_Y_DISPLAY = True
CAXIS_LIMITS = None
SMOOTH_SIGMA = 1
MAX_DISTANCE_FROM_CENTER_PX = 175

NORMALIZE_COUNTS = True
NORMALIZATION_PERCENTILE = 100.0

BODY_PART = 'mid_mid'
INCLUDE_GROUPS = ['saline', 'muscimol']
INCLUDE_RATS = []
SESSIONS = ['H1', 'H2', 'H3', 'S1', 'S2', 'T']


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


def filter_rearing_data(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data
    
    if 'rearing' not in data.columns:
        return None
    
    rearing_mask = data['rearing'].values > 0
    rearing_data = data[rearing_mask].copy()
    
    return rearing_data if len(rearing_data) > 0 else None


def filter_after_rearing_data(data: pd.DataFrame, frames_after: int = 450) -> pd.DataFrame:
    if data is None or len(data) == 0:
        return data
    
    if 'rearing' not in data.columns:
        return None
    
    rearing_frames = np.where(data['rearing'].values > 0)[0]
    
    if len(rearing_frames) == 0:
        return None
    
    after_frames = set()
    for frame_idx in rearing_frames:
        end_frame = min(frame_idx + frames_after + 1, len(data))
        for after_frame in range(frame_idx + 1, end_frame):
            after_frames.add(after_frame)
    
    if len(after_frames) == 0:
        return None
    
    after_mask = np.array([i in after_frames for i in range(len(data))])
    after_data = data[after_mask].copy()
    
    return after_data if len(after_data) > 0 else None


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
            print("Warning: scipy not available, skipping smoothing")
    
    return counts


def normalize_counts_by_percentile(counts: np.ndarray, percentile: float = 95.0) -> Tuple[np.ndarray, float]:
    if counts.size == 0 or np.all(counts == 0):
        return counts, 0.0
    
    non_zero_counts = counts[counts > 0]
    
    if len(non_zero_counts) == 0:
        return counts, 0.0
    
    normalization_factor = np.percentile(non_zero_counts, percentile)
    
    if normalization_factor <= 0:
        return counts, 0.0
    
    normalized_counts = counts / normalization_factor
    normalized_counts = np.clip(normalized_counts, 0, 1)
    
    return normalized_counts, normalization_factor


def resolve_caxis(limits: Optional[List[float]], rat_max: float) -> Tuple[float, float]:
    if limits is None:
        return (0, max(1, rat_max))
    if len(limits) == 1:
        return (0, max(0, limits[0]))
    if len(limits) == 2:
        lo, hi = limits[0], limits[1]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (lo, hi)
    return (0, max(1, rat_max))


def parse_polygon_vertices(vertices_str: str) -> Optional[np.ndarray]:
    if pd.isna(vertices_str) or not vertices_str:
        return None
    try:
        vertices_str_clean = str(vertices_str).strip()
        if not vertices_str_clean.startswith('['):
            return None
        vertices_str_clean = re.sub(r'\n', ' ', vertices_str_clean)
        vertices_str_clean = re.sub(r'\[(\d+)\s+(\d+)\]', r'[\1, \2]', vertices_str_clean)
        vertices_str_clean = re.sub(r'\]\s+\[', '], [', vertices_str_clean)
        vertices_str_clean = re.sub(r'\s+', ' ', vertices_str_clean)
        vertices_list = ast.literal_eval(vertices_str_clean)
        if isinstance(vertices_list, list) and len(vertices_list) > 0:
            vertices_array = np.array(vertices_list)
            if vertices_array.ndim == 2 and vertices_array.shape[1] == 2:
                return vertices_array.astype(float)
            elif vertices_array.ndim == 1 and len(vertices_array) >= 2:
                return vertices_array.reshape(-1, 2).astype(float)
    except Exception as e:
        pass
    return None


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    if polygon is None or len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def get_a1_polygon(polygons_dict: Dict[str, Optional[pd.DataFrame]], 
                   session: str, rat_id: str) -> Optional[np.ndarray]:
    """Get the A1 polygon vertices for a specific rat and session."""
    if session != 'T':
        return None
    
    session_polygons_df = polygons_dict.get(session)
    if session_polygons_df is None:
        return None
    
    video_pattern = f"{rat_id}{session}"
    matching_rows = session_polygons_df[
        (session_polygons_df['Video'] == video_pattern) & 
        (session_polygons_df['Name'] == 'A1')
    ]
    
    if len(matching_rows) == 0:
        return None
    
    # Get the first A1 polygon found
    row = matching_rows.iloc[0]
    vertices_str = row.get('vertices', None)
    if pd.isna(vertices_str):
        return None
    
    vertices = parse_polygon_vertices(vertices_str)
    return vertices


def filter_rearing_in_a1(data: pd.DataFrame, a1_polygon: np.ndarray) -> Optional[pd.DataFrame]:
    """Filter rearing data to only include points where the rat was rearing in the A1 area."""
    if data is None or len(data) == 0:
        return None
    
    if 'rearing' not in data.columns:
        return None
    
    if a1_polygon is None:
        return None
    
    # Filter for rearing events
    rearing_mask = data['rearing'].values > 0
    if not np.any(rearing_mask):
        return None
    
    rearing_data = data[rearing_mask].copy()
    
    # Check which rearing points are inside A1 polygon
    inside_mask = np.array([
        point_in_polygon((x, y), a1_polygon) 
        for x, y in zip(rearing_data['x'].values, rearing_data['y'].values)
    ])
    
    if not np.any(inside_mask):
        return None
    
    a1_rearing_data = rearing_data[inside_mask].copy()
    return a1_rearing_data if len(a1_rearing_data) > 0 else None


def filter_after_a1_rearing(data: pd.DataFrame, a1_polygon: np.ndarray, 
                           frames_after: int = 600) -> Optional[pd.DataFrame]:
    """Filter data to include 20 seconds (600 frames) after rearing events in A1 area."""
    if data is None or len(data) == 0:
        return None
    
    if 'rearing' not in data.columns:
        return None
    
    if a1_polygon is None:
        return None
    
    # Find rearing frames that are in A1 area
    rearing_mask = data['rearing'].values > 0
    if not np.any(rearing_mask):
        return None
    
    rearing_data = data[rearing_mask].copy()
    # Use positional indices (0-based) since data may have been filtered
    rearing_indices = np.where(rearing_mask)[0]
    
    # Check which rearing points are inside A1 polygon
    inside_mask = np.array([
        point_in_polygon((x, y), a1_polygon) 
        for x, y in zip(rearing_data['x'].values, rearing_data['y'].values)
    ])
    
    if not np.any(inside_mask):
        return None
    
    # Get the frame indices where rearing occurred in A1 (positional indices)
    a1_rearing_frame_indices = rearing_indices[inside_mask]
    
    # Collect all frames in the 20 seconds after each A1 rearing event
    after_frames = set()
    for frame_idx in a1_rearing_frame_indices:
        end_frame = min(frame_idx + frames_after + 1, len(data))
        for after_frame in range(frame_idx + 1, end_frame):
            after_frames.add(after_frame)
    
    if len(after_frames) == 0:
        return None
    
    after_mask = np.array([i in after_frames for i in range(len(data))])
    after_data = data[after_mask].copy()
    
    return after_data if len(after_data) > 0 else None


def plot_rat_heatmaps(rat_id: str, group: str, data_dict: Dict[str, pd.DataFrame],
                      x_edges: np.ndarray, y_edges: np.ndarray, 
                      clim: Tuple[float, float], output_path: Path,
                      polygons_dict: Optional[Dict[str, pd.DataFrame]] = None,
                      center_x: Optional[float] = None, center_y: Optional[float] = None,
                      title_suffix: str = ""):
    sessions = SESSIONS
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    title = f'{rat_id} — Group: {group} — H1/H2/H3/S1/S2/T occupancy'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontweight='bold', fontsize=12)
    axes = axes.flatten()
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    for idx, session in enumerate(sessions):
        ax = axes[idx]
        data = data_dict[session]
        
        if data is None or len(data) == 0:
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        counts = compute_histogram(data, x_edges, y_edges, SMOOTH_SIGMA)
        
        if np.all(counts == 0):
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        if NORMALIZE_COUNTS:
            counts_normalized, norm_factor = normalize_counts_by_percentile(
                counts, NORMALIZATION_PERCENTILE
            )
            display_counts = counts_normalized
            display_clim = (0, 1)
        else:
            display_counts = counts
            display_clim = clim
        
        im = ax.imshow(display_counts, extent=[x_edges[0], x_edges[-1], 
                                      y_edges[-1], y_edges[0]],
                      cmap=COLORMAP_NAME, aspect='equal', origin='upper')
        im.set_clim(display_clim[0], display_clim[1])
        
        # Plot polygon centers if available
        if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
            session_polygons_df = polygons_dict.get(session)
            if session_polygons_df is not None:
                video_pattern = f"{rat_id}{session}"
                matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                
                if len(matching_rows) > 0:
                    center_x_coords = []
                    center_y_coords = []
                    names = []
                    for _, row in matching_rows.iterrows():
                        center_x = row.get('Center_X', None)
                        center_y = row.get('Center_Y', None)
                        if pd.notna(center_x) and pd.notna(center_y):
                            center_x_coords.append(center_x)
                            center_y_coords.append(center_y)
                            # Get object name if available (especially for session T)
                            name = row.get('Name', '')
                            names.append(name if pd.notna(name) else '')
                    
                    if len(center_x_coords) > 0:
                        ax.scatter(center_x_coords, center_y_coords, 
                                  color='white', marker='x', s=100, 
                                  linewidths=2, alpha=0.9, label='Polygon centers')
                        # Add text labels at center coordinates
                        for x, y, name in zip(center_x_coords, center_y_coords, names):
                            if name:  # Only add text if name exists
                                ax.text(x, y, name, fontsize=8, color='white',
                                       ha='center', va='center', weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                               alpha=0.6, edgecolor='white', linewidth=0.5))
        
        ax.set_title(session, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (px)')
        
        if idx == 0 or idx == 3:
            yl = 'y (px)'
            if INVERT_Y_DISPLAY:
                yl += ' (inv)'
            ax.set_ylabel(yl)
        
        if SHOW_COLORBAR and idx == 5:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if NORMALIZE_COUNTS:
                cbar.set_label(f'Occupancy (normalized by {NORMALIZATION_PERCENTILE:.0f}th percentile)', 
                             rotation=270, labelpad=15)
            else:
                cbar.set_label('Nose occupancy (count)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_rearing_heatmaps(rat_id: str, group: str, data_dict: Dict[str, pd.DataFrame],
                          x_edges: np.ndarray, y_edges: np.ndarray, 
                          clim: Tuple[float, float], output_path: Path,
                          polygons_dict: Optional[Dict[str, pd.DataFrame]] = None,
                          center_x: Optional[float] = None, center_y: Optional[float] = None,
                          title_suffix: str = ""):
    sessions = ['S1', 'S2', 'T']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    title = f'{rat_id} — Group: {group} — S1/S2/T occupancy'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontweight='bold', fontsize=12)
    axes = axes.flatten()
    
    for idx, session in enumerate(sessions):
        ax = axes[idx]
        data = data_dict.get(session)
        
        if data is None or len(data) == 0:
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        counts = compute_histogram(data, x_edges, y_edges, SMOOTH_SIGMA)
        
        if np.all(counts == 0):
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        if NORMALIZE_COUNTS:
            counts_normalized, norm_factor = normalize_counts_by_percentile(
                counts, NORMALIZATION_PERCENTILE
            )
            display_counts = counts_normalized
            display_clim = (0, 1)
        else:
            display_counts = counts
            display_clim = clim
        
        im = ax.imshow(display_counts, extent=[x_edges[0], x_edges[-1], 
                                      y_edges[-1], y_edges[0]],
                      cmap=COLORMAP_NAME, aspect='equal', origin='upper')
        im.set_clim(display_clim[0], display_clim[1])
        
        # Plot polygon centers if available
        if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
            session_polygons_df = polygons_dict.get(session)
            if session_polygons_df is not None:
                video_pattern = f"{rat_id}{session}"
                matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                
                if len(matching_rows) > 0:
                    center_x_coords = []
                    center_y_coords = []
                    names = []
                    for _, row in matching_rows.iterrows():
                        center_x = row.get('Center_X', None)
                        center_y = row.get('Center_Y', None)
                        if pd.notna(center_x) and pd.notna(center_y):
                            center_x_coords.append(center_x)
                            center_y_coords.append(center_y)
                            # Get object name if available (especially for session T)
                            name = row.get('Name', '')
                            names.append(name if pd.notna(name) else '')
                    
                    if len(center_x_coords) > 0:
                        ax.scatter(center_x_coords, center_y_coords, 
                                  color='white', marker='x', s=100, 
                                  linewidths=2, alpha=0.9, label='Polygon centers')
                        # Add text labels at center coordinates
                        for x, y, name in zip(center_x_coords, center_y_coords, names):
                            if name:  # Only add text if name exists
                                ax.text(x, y, name, fontsize=8, color='white',
                                       ha='center', va='center', weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                               alpha=0.6, edgecolor='white', linewidth=0.5))
        
        ax.set_title(session, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (px)')
        
        if idx == 0:
            yl = 'y (px)'
            if INVERT_Y_DISPLAY:
                yl += ' (inv)'
            ax.set_ylabel(yl)
        
        if SHOW_COLORBAR and idx == 2:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if NORMALIZE_COUNTS:
                cbar.set_label(f'Occupancy (normalized by {NORMALIZATION_PERCENTILE:.0f}th percentile)', 
                             rotation=270, labelpad=15)
            else:
                cbar.set_label('Nose occupancy (count)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_rat_trajectories(rat_id: str, group: str, data_dict: Dict[str, pd.DataFrame],
                          output_path: Path,
                          polygons_dict: Optional[Dict[str, pd.DataFrame]] = None,
                          center_x: Optional[float] = None, center_y: Optional[float] = None,
                          title_suffix: str = ""):
    """Plot trajectories as continuous lines for each session."""
    sessions = SESSIONS
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    title = f'{rat_id} — Group: {group} — H1/H2/H3/S1/S2/T trajectories'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontweight='bold', fontsize=12)
    axes = axes.flatten()
    
    for idx, session in enumerate(sessions):
        ax = axes[idx]
        data = data_dict.get(session)
        
        if data is None or len(data) == 0:
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        x = data['x'].values
        y = data['y'].values
        
        # Plot trajectory as a line
        ax.plot(x, y, color='blue', linewidth=0.5, alpha=0.6, label='Trajectory')
        
        # Plot start point
        if len(x) > 0:
            ax.scatter(x[0], y[0], color='green', marker='o', s=50, 
                      label='Start', zorder=5)
        
        # Plot end point
        if len(x) > 0:
            ax.scatter(x[-1], y[-1], color='red', marker='s', s=50, 
                      label='End', zorder=5)
        
        # Plot polygon centers if available
        if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
            session_polygons_df = polygons_dict.get(session)
            if session_polygons_df is not None:
                video_pattern = f"{rat_id}{session}"
                matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                
                if len(matching_rows) > 0:
                    center_x_coords = []
                    center_y_coords = []
                    for _, row in matching_rows.iterrows():
                        center_x = row.get('Center_X', None)
                        center_y = row.get('Center_Y', None)
                        if pd.notna(center_x) and pd.notna(center_y):
                            center_x_coords.append(center_x)
                            center_y_coords.append(center_y)
                    
                    if len(center_x_coords) > 0:
                        ax.scatter(center_x_coords, center_y_coords, 
                                  color='orange', marker='x', s=100, 
                                  linewidths=2, alpha=0.9, label='Polygon centers', zorder=4)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(session, fontweight='bold')
        ax.set_xlabel('x (px)')
        
        if idx == 0 or idx == 3:
            yl = 'y (px)'
            if INVERT_Y_DISPLAY:
                yl += ' (inv)'
            ax.set_ylabel(yl)
        
        # Add legend only for the first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_rearing_trajectories(rat_id: str, group: str, data_dict: Dict[str, pd.DataFrame],
                              output_path: Path,
                              polygons_dict: Optional[Dict[str, pd.DataFrame]] = None,
                              center_x: Optional[float] = None, center_y: Optional[float] = None,
                              title_suffix: str = ""):
    """Plot trajectories for S1, S2, T sessions."""
    sessions = ['S1', 'S2', 'T']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    title = f'{rat_id} — Group: {group} — S1/S2/T trajectories'
    if title_suffix:
        title += f' — {title_suffix}'
    fig.suptitle(title, fontweight='bold', fontsize=12)
    axes = axes.flatten()
    
    for idx, session in enumerate(sessions):
        ax = axes[idx]
        data = data_dict.get(session)
        
        if data is None or len(data) == 0:
            ax.axis('off')
            ax.set_title(f'{session} (no data)', fontweight='bold')
            continue
        
        x = data['x'].values
        y = data['y'].values
        
        # Plot trajectory as a line
        ax.plot(x, y, color='blue', linewidth=0.5, alpha=0.6, label='Trajectory')
        
        # Plot start point
        if len(x) > 0:
            ax.scatter(x[0], y[0], color='green', marker='o', s=50, 
                      label='Start', zorder=5)
        
        # Plot end point
        if len(x) > 0:
            ax.scatter(x[-1], y[-1], color='red', marker='s', s=50, 
                      label='End', zorder=5)
        
        # Plot polygon centers if available
        if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
            session_polygons_df = polygons_dict.get(session)
            if session_polygons_df is not None:
                video_pattern = f"{rat_id}{session}"
                matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                
                if len(matching_rows) > 0:
                    center_x_coords = []
                    center_y_coords = []
                    for _, row in matching_rows.iterrows():
                        center_x = row.get('Center_X', None)
                        center_y = row.get('Center_Y', None)
                        if pd.notna(center_x) and pd.notna(center_y):
                            center_x_coords.append(center_x)
                            center_y_coords.append(center_y)
                    
                    if len(center_x_coords) > 0:
                        ax.scatter(center_x_coords, center_y_coords, 
                                  color='orange', marker='x', s=100, 
                                  linewidths=2, alpha=0.9, label='Polygon centers', zorder=4)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(session, fontweight='bold')
        ax.set_xlabel('x (px)')
        
        if idx == 0:
            yl = 'y (px)'
            if INVERT_Y_DISPLAY:
                yl += ' (inv)'
            ax.set_ylabel(yl)
        
        # Add legend only for the first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_a1_rearing_heatmaps(rat_id: str, group: str, 
                             rearing_data: pd.DataFrame, after_rearing_data: pd.DataFrame,
                             x_edges: np.ndarray, y_edges: np.ndarray, 
                             clim: Tuple[float, float], output_path: Path,
                             polygons_dict: Optional[Dict[str, pd.DataFrame]] = None,
                             center_x: Optional[float] = None, center_y: Optional[float] = None,
                             session: str = "T"):
    """Plot heatmaps for rearing in A1 area and 20 seconds after."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    title = f'{rat_id} — Group: {group} — A1 Rearing Heatmaps ({session})'
    fig.suptitle(title, fontweight='bold', fontsize=12)
    
    # Plot rearing in A1
    ax = axes[0]
    if rearing_data is None or len(rearing_data) == 0:
        ax.axis('off')
        ax.set_title('Rearing in A1 (no data)', fontweight='bold')
    else:
        counts = compute_histogram(rearing_data, x_edges, y_edges, SMOOTH_SIGMA)
        
        if np.all(counts == 0):
            ax.axis('off')
            ax.set_title('Rearing in A1 (no data)', fontweight='bold')
        else:
            if NORMALIZE_COUNTS:
                counts_normalized, norm_factor = normalize_counts_by_percentile(
                    counts, NORMALIZATION_PERCENTILE
                )
                display_counts = counts_normalized
                display_clim = (0, 1)
            else:
                display_counts = counts
                display_clim = clim
            
            im = ax.imshow(display_counts, extent=[x_edges[0], x_edges[-1], 
                                          y_edges[-1], y_edges[0]],
                          cmap=COLORMAP_NAME, aspect='equal', origin='upper')
            im.set_clim(display_clim[0], display_clim[1])
            
            # Plot polygon centers if available
            if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
                session_polygons_df = polygons_dict.get(session)
                if session_polygons_df is not None:
                    video_pattern = f"{rat_id}{session}"
                    matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                    
                    if len(matching_rows) > 0:
                        center_x_coords = []
                        center_y_coords = []
                        names = []
                        for _, row in matching_rows.iterrows():
                            center_x_val = row.get('Center_X', None)
                            center_y_val = row.get('Center_Y', None)
                            if pd.notna(center_x_val) and pd.notna(center_y_val):
                                center_x_coords.append(center_x_val)
                                center_y_coords.append(center_y_val)
                                name = row.get('Name', '')
                                names.append(name if pd.notna(name) else '')
                        
                        if len(center_x_coords) > 0:
                            ax.scatter(center_x_coords, center_y_coords, 
                                      color='white', marker='x', s=100, 
                                      linewidths=2, alpha=0.9)
                            # Add text labels
                            for x, y, name in zip(center_x_coords, center_y_coords, names):
                                if name:
                                    ax.text(x, y, name, fontsize=8, color='white',
                                           ha='center', va='center', weight='bold',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                                   alpha=0.6, edgecolor='white', linewidth=0.5))
            
            ax.set_title('Rearing in A1', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x (px)')
            ax.set_ylabel('y (px)')
    
    # Plot 20 seconds after rearing in A1
    ax = axes[1]
    if after_rearing_data is None or len(after_rearing_data) == 0:
        ax.axis('off')
        ax.set_title('20s after A1 Rearing (no data)', fontweight='bold')
    else:
        counts = compute_histogram(after_rearing_data, x_edges, y_edges, SMOOTH_SIGMA)
        
        if np.all(counts == 0):
            ax.axis('off')
            ax.set_title('20s after A1 Rearing (no data)', fontweight='bold')
        else:
            if NORMALIZE_COUNTS:
                counts_normalized, norm_factor = normalize_counts_by_percentile(
                    counts, NORMALIZATION_PERCENTILE
                )
                display_counts = counts_normalized
                display_clim = (0, 1)
            else:
                display_counts = counts
                display_clim = clim
            
            im = ax.imshow(display_counts, extent=[x_edges[0], x_edges[-1], 
                                          y_edges[-1], y_edges[0]],
                          cmap=COLORMAP_NAME, aspect='equal', origin='upper')
            im.set_clim(display_clim[0], display_clim[1])
            
            # Plot polygon centers if available
            if session in ['S1', 'S2', 'T'] and polygons_dict is not None:
                session_polygons_df = polygons_dict.get(session)
                if session_polygons_df is not None:
                    video_pattern = f"{rat_id}{session}"
                    matching_rows = session_polygons_df[session_polygons_df['Video'] == video_pattern]
                    
                    if len(matching_rows) > 0:
                        center_x_coords = []
                        center_y_coords = []
                        names = []
                        for _, row in matching_rows.iterrows():
                            center_x_val = row.get('Center_X', None)
                            center_y_val = row.get('Center_Y', None)
                            if pd.notna(center_x_val) and pd.notna(center_y_val):
                                center_x_coords.append(center_x_val)
                                center_y_coords.append(center_y_val)
                                name = row.get('Name', '')
                                names.append(name if pd.notna(name) else '')
                        
                        if len(center_x_coords) > 0:
                            ax.scatter(center_x_coords, center_y_coords, 
                                      color='white', marker='x', s=100, 
                                      linewidths=2, alpha=0.9)
                            # Add text labels
                            for x, y, name in zip(center_x_coords, center_y_coords, names):
                                if name:
                                    ax.text(x, y, name, fontsize=8, color='white',
                                           ha='center', va='center', weight='bold',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                                   alpha=0.6, edgecolor='white', linewidth=0.5))
            
            ax.set_title('20s after A1 Rearing', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x (px)')
            
            if SHOW_COLORBAR:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if NORMALIZE_COUNTS:
                    cbar.set_label(f'Occupancy (normalized by {NORMALIZATION_PERCENTILE:.0f}th percentile)', 
                                 rotation=270, labelpad=15)
                else:
                    cbar.set_label('Nose occupancy (count)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print("Loading group index...")
    group_map = load_group_index()
    
    print("Loading polygons...")
    polygons_dict = load_polygons()
    total_polygons = 0
    for session, df in polygons_dict.items():
        if df is not None:
            count = len(df)
            total_polygons += count
            print(f"  Loaded {count} polygon entries for {session}")
    if total_polygons > 0:
        print(f"  Total: {total_polygons} polygon entries")
    
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
        rat_num = f"R{match.group(1)}"
        
        if INCLUDE_RATS and rat_num not in INCLUDE_RATS:
            continue
        if rat_id not in group_map:
            continue
        group = group_map[rat_id]
        if group not in INCLUDE_GROUPS:
            continue
        
        filtered_rat_ids.append(rat_id)
    
    print(f"Found {len(filtered_rat_ids)} rats to process:")
    print(", ".join(filtered_rat_ids))
    
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
                if data is not None:
                    print(f"  {session}: {len(data)} valid points")
                else:
                    print(f"  {session}: no valid data")
            else:
                data_dict[session] = None
                print(f"  {session}: file not found")
        
        center_x, center_y = estimate_circle_center(data_dict)
        print(f"  Estimated circle center: ({center_x:.1f}, {center_y:.1f})")
        
        filtered_data_dict = {}
        for session in SESSIONS:
            data = data_dict[session]
            if data is not None:
                filtered_data = filter_points_by_distance(
                    data, center_x, center_y, MAX_DISTANCE_FROM_CENTER_PX
                )
                if filtered_data is not None:
                    original_count = len(data)
                    filtered_count = len(filtered_data)
                    removed_count = original_count - filtered_count
                    if removed_count > 0:
                        print(f"  {session}: removed {removed_count} points ({removed_count/original_count*100:.1f}%) beyond {MAX_DISTANCE_FROM_CENTER_PX}px")
                filtered_data_dict[session] = filtered_data
            else:
                filtered_data_dict[session] = None
        
        normalized_data_dict = {}
        for session in SESSIONS:
            data = filtered_data_dict[session]
            if data is not None:
                normalized_data = normalize_coordinates(data, center_x, center_y)
                normalized_data_dict[session] = normalized_data
            else:
                normalized_data_dict[session] = None
        
        data_dict = normalized_data_dict
        
        x_edges, y_edges = compute_normalized_grid(BIN_SIZE_PX, MAX_DISTANCE_FROM_CENTER_PX)
        
        rat_max = 0
        for session in SESSIONS:
            counts = compute_histogram(data_dict[session], x_edges, y_edges, SMOOTH_SIGMA)
            if counts.size > 0:
                rat_max = max(rat_max, np.max(counts))
        
        clim = resolve_caxis(CAXIS_LIMITS, rat_max)
        
        normalized_polygons_dict = None
        if polygons_dict is not None:
            normalized_polygons_dict = {}
            for session in ['S1', 'S2', 'T']:
                session_polygons_df = polygons_dict.get(session)
                if session_polygons_df is not None:
                    normalized_df = session_polygons_df.copy()
                    video_pattern = f"{rat_id}{session}"
                    matching_rows = normalized_df['Video'] == video_pattern
                    if matching_rows.any():
                        normalized_df.loc[matching_rows, 'Center_X'] = (
                            normalized_df.loc[matching_rows, 'Center_X'] - center_x
                        )
                        normalized_df.loc[matching_rows, 'Center_Y'] = (
                            normalized_df.loc[matching_rows, 'Center_Y'] - center_y
                        )
                        for idx in normalized_df[matching_rows].index:
                            vertices_str = normalized_df.loc[idx, 'vertices']
                            vertices_array = parse_polygon_vertices(vertices_str)
                            if vertices_array is not None:
                                vertices_array[:, 0] = vertices_array[:, 0] - center_x
                                vertices_array[:, 1] = vertices_array[:, 1] - center_y
                                normalized_df.loc[idx, 'vertices'] = str(vertices_array.tolist())
                    normalized_polygons_dict[session] = normalized_df
        
        output_filename = f"{rat_id} — {group} — H1_H2_H3_S1_S2_T occupancy.png"
        output_path = OUTPUT_DIR / output_filename
        
        plot_rat_heatmaps(rat_id, group, data_dict, x_edges, y_edges, clim, output_path, 
                         normalized_polygons_dict, center_x=0, center_y=0)
        print(f"  Saved: {output_path}")
        
        # Plot trajectories
        trajectories_output_dir = OUTPUT_DIR / 'trajectories'
        trajectories_output_dir.mkdir(parents=True, exist_ok=True)
        trajectories_output_filename = f"{rat_id} — {group} — H1_H2_H3_S1_S2_T trajectories.png"
        trajectories_output_path = trajectories_output_dir / trajectories_output_filename
        plot_rat_trajectories(rat_id, group, data_dict, trajectories_output_path,
                             normalized_polygons_dict, center_x=0, center_y=0)
        print(f"  Saved trajectories: {trajectories_output_path}")
        
        rearing_output_dir = OUTPUT_DIR / 'rearing_heatmap'
        rearing_output_dir.mkdir(parents=True, exist_ok=True)
        after_rearing_output_dir = OUTPUT_DIR / 'after_rearing_heatmap'
        after_rearing_output_dir.mkdir(parents=True, exist_ok=True)
        
        rearing_sessions = ['S1', 'S2', 'T']
        rearing_data_dict = {}
        after_rearing_data_dict = {}
        
        for session in rearing_sessions:
            filepath = file_paths[session]
            if filepath and filepath.exists():
                data_with_rearing = read_dlc_csv(filepath, BODY_PART, include_rearing=True)
                
                if data_with_rearing is not None and 'rearing' in data_with_rearing.columns:
                    filtered_rearing = filter_points_by_distance(
                        data_with_rearing, center_x, center_y, MAX_DISTANCE_FROM_CENTER_PX
                    )
                    if filtered_rearing is not None:
                        rearing_data = filter_rearing_data(filtered_rearing)
                        after_rearing_data = filter_after_rearing_data(filtered_rearing, frames_after=450)
                        
                        if rearing_data is not None:
                            rearing_data = normalize_coordinates(rearing_data, center_x, center_y)
                            rearing_data_dict[session] = rearing_data
                        
                        if after_rearing_data is not None:
                            after_rearing_data = normalize_coordinates(after_rearing_data, center_x, center_y)
                            after_rearing_data_dict[session] = after_rearing_data
                else:
                    rearing_data_dict[session] = None
                    after_rearing_data_dict[session] = None
            else:
                rearing_data_dict[session] = None
                after_rearing_data_dict[session] = None
        
        rearing_rat_max = 0
        for session in rearing_sessions:
            if rearing_data_dict.get(session) is not None:
                counts = compute_histogram(rearing_data_dict[session], x_edges, y_edges, SMOOTH_SIGMA)
                if counts.size > 0:
                    rearing_rat_max = max(rearing_rat_max, np.max(counts))
        
        rearing_clim = resolve_caxis(CAXIS_LIMITS, rearing_rat_max)
        
        rearing_output_filename = f"{rat_id} — {group} — S1_S2_T rearing occupancy.png"
        rearing_output_path = rearing_output_dir / rearing_output_filename
        
        rearing_plot_data = {s: rearing_data_dict.get(s) for s in rearing_sessions}
        plot_rearing_heatmaps(rat_id, group, rearing_plot_data, x_edges, y_edges, rearing_clim, 
                            rearing_output_path, normalized_polygons_dict, center_x=0, center_y=0)
        print(f"  Saved rearing heatmap: {rearing_output_path}")
        
        after_rearing_rat_max = 0
        for session in rearing_sessions:
            if after_rearing_data_dict.get(session) is not None:
                counts = compute_histogram(after_rearing_data_dict[session], x_edges, y_edges, SMOOTH_SIGMA)
                if counts.size > 0:
                    after_rearing_rat_max = max(after_rearing_rat_max, np.max(counts))
        
        after_rearing_clim = resolve_caxis(CAXIS_LIMITS, after_rearing_rat_max)
        
        after_rearing_output_filename = f"{rat_id} — {group} — S1_S2_T after_rearing occupancy.png"
        after_rearing_output_path = after_rearing_output_dir / after_rearing_output_filename
        
        after_rearing_plot_data = {s: after_rearing_data_dict.get(s) for s in rearing_sessions}
        plot_rearing_heatmaps(rat_id, group, after_rearing_plot_data, x_edges, y_edges, after_rearing_clim,
                            after_rearing_output_path, normalized_polygons_dict, center_x=0, center_y=0,
                            title_suffix="after rearing (15s)")
        print(f"  Saved after-rearing heatmap: {after_rearing_output_path}")
        
        # Plot rearing trajectories
        rearing_trajectories_output_dir = OUTPUT_DIR / 'trajectories' / 'rearing_trajectories'
        rearing_trajectories_output_dir.mkdir(parents=True, exist_ok=True)
        rearing_trajectories_output_filename = f"{rat_id} — {group} — S1_S2_T rearing trajectories.png"
        rearing_trajectories_output_path = rearing_trajectories_output_dir / rearing_trajectories_output_filename
        rearing_trajectories_plot_data = {s: rearing_data_dict.get(s) for s in rearing_sessions}
        plot_rearing_trajectories(rat_id, group, rearing_trajectories_plot_data, rearing_trajectories_output_path,
                                 normalized_polygons_dict, center_x=0, center_y=0, title_suffix="rearing")
        print(f"  Saved rearing trajectories: {rearing_trajectories_output_path}")
        
        # Plot after-rearing trajectories
        after_rearing_trajectories_output_filename = f"{rat_id} — {group} — S1_S2_T after_rearing trajectories.png"
        after_rearing_trajectories_output_path = rearing_trajectories_output_dir / after_rearing_trajectories_output_filename
        after_rearing_trajectories_plot_data = {s: after_rearing_data_dict.get(s) for s in rearing_sessions}
        plot_rearing_trajectories(rat_id, group, after_rearing_trajectories_plot_data, after_rearing_trajectories_output_path,
                                 normalized_polygons_dict, center_x=0, center_y=0, title_suffix="after rearing (15s)")
        print(f"  Saved after-rearing trajectories: {after_rearing_trajectories_output_path}")
        
        # A1 Rearing Heatmaps
        a1_rearing_output_dir = OUTPUT_DIR / 'a1_rearing_heatmap'
        a1_rearing_output_dir.mkdir(parents=True, exist_ok=True)
        
        a1_rearing_sessions = ['S1', 'S2', 'T']
        for session in a1_rearing_sessions:
            filepath = file_paths[session]
            if filepath and filepath.exists():
                data_with_rearing = read_dlc_csv(filepath, BODY_PART, include_rearing=True)
                
                if data_with_rearing is not None and 'rearing' in data_with_rearing.columns:
                    # Get A1 polygon (needs to be in original coordinates before normalization)
                    a1_polygon_original = get_a1_polygon(polygons_dict, session, rat_id)
                    
                    if a1_polygon_original is not None:
                        # Filter data by distance first
                        filtered_data = filter_points_by_distance(
                            data_with_rearing, center_x, center_y, MAX_DISTANCE_FROM_CENTER_PX
                        )
                        
                        if filtered_data is not None:
                            # Filter for rearing in A1 area (before normalization)
                            a1_rearing_data = filter_rearing_in_a1(filtered_data, a1_polygon_original)
                            a1_after_rearing_data = filter_after_a1_rearing(filtered_data, a1_polygon_original, frames_after=600)
                            
                            if a1_rearing_data is not None or a1_after_rearing_data is not None:
                                # Normalize coordinates
                                if a1_rearing_data is not None:
                                    a1_rearing_data = normalize_coordinates(a1_rearing_data, center_x, center_y)
                                
                                if a1_after_rearing_data is not None:
                                    a1_after_rearing_data = normalize_coordinates(a1_after_rearing_data, center_x, center_y)
                                
                                # Get normalized A1 polygon for plotting
                                a1_polygon_normalized = a1_polygon_original.copy()
                                a1_polygon_normalized[:, 0] = a1_polygon_normalized[:, 0] - center_x
                                a1_polygon_normalized[:, 1] = a1_polygon_normalized[:, 1] - center_y
                                
                                # Calculate max counts for color scaling
                                a1_rearing_max = 0
                                if a1_rearing_data is not None:
                                    counts = compute_histogram(a1_rearing_data, x_edges, y_edges, SMOOTH_SIGMA)
                                    if counts.size > 0:
                                        a1_rearing_max = max(a1_rearing_max, np.max(counts))
                                
                                if a1_after_rearing_data is not None:
                                    counts = compute_histogram(a1_after_rearing_data, x_edges, y_edges, SMOOTH_SIGMA)
                                    if counts.size > 0:
                                        a1_rearing_max = max(a1_rearing_max, np.max(counts))
                                
                                a1_rearing_clim = resolve_caxis(CAXIS_LIMITS, a1_rearing_max)
                                
                                # Generate output filename
                                a1_rearing_output_filename = f"{rat_id} — {group} — A1_rearing_heatmap_{session}.png"
                                a1_rearing_output_path = a1_rearing_output_dir / a1_rearing_output_filename
                                
                                # Plot A1 rearing heatmaps
                                plot_a1_rearing_heatmaps(rat_id, group, a1_rearing_data, a1_after_rearing_data,
                                                        x_edges, y_edges, a1_rearing_clim, a1_rearing_output_path,
                                                        normalized_polygons_dict, center_x=0, center_y=0, session=session)
                                print(f"  Saved A1 rearing heatmap ({session}): {a1_rearing_output_path}")
    
    print(f"\nDone! All figures saved to: {OUTPUT_DIR}")


def run_statistics():
    from .statistics import run_statistical_comparisons
    
    stats_output_dir = OUTPUT_DIR / 'statistics'
    stats_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Running statistical comparisons...")
    print("="*60)
    
    results = run_statistical_comparisons(stats_output_dir)
    
    summary_data = []
    for result in results:
        if 'error' not in result:
            summary_data.append({
                'Session': result['session'],
                'Group1': result['group1'],
                'Group2': result['group2'],
                'N1': result['n1'],
                'N2': result['n2'],
                'Mean_Difference': result['mean_difference'],
                'Significant_Pixels': result['n_significant_pixels'],
                'Pct_Significant': result['pct_significant_pixels']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = stats_output_dir / 'statistical_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nStatistical summary saved to: {summary_path}")
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    
    return results


if __name__ == '__main__':
    main()
