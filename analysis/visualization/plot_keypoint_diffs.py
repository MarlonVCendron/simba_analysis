import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from analysis.consts import BODYPARTS, BASE_PATH
from analysis.load import iter_bodyparts
from analysis.outliers.outliers import detect_outliers, interpolate_outliers, calculate_coordinate_diffs

def plot_keypoint_diffs(
    dlc_data,
    output_dir: Optional[Path] = None,
    scatter_outliers: bool = False,
    limit_scale_to_interpolated: bool = False,
    plot_x: bool = False
):
    if output_dir is None:
        output_dir = BASE_PATH / 'analysis' / 'figures' / 'keypoint_diffs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_sessions_data = []
    for rat_id in dlc_data.keys():
        for session in dlc_data[rat_id]:
            data = dlc_data[rat_id][session]
            all_sessions_data.append((rat_id, session, data))
    
    for rat_id, session, data in all_sessions_data:
        n_bodyparts = len(BODYPARTS)
        
        fig, axes = plt.subplots(n_bodyparts, 1, figsize=(15, 5 * n_bodyparts), sharey=False)
        
        for idx, bodypart in enumerate(BODYPARTS):
            ax = axes[idx]
            
            for bp, x, y in iter_bodyparts(data, [bodypart]):
                if bp != bodypart:
                  continue

                x_values = x.values
                y_values = y.values
                x_diffs, y_diffs, distances = calculate_coordinate_diffs(x_values, y_values)
                frames = np.arange(len(x_values))
                break
            
            outliers = detect_outliers(distances)
            
            x_values_interp, y_values_interp = interpolate_outliers(x_values, y_values, outliers, method='cubic')
            x_diffs_interp, y_diffs_interp, distances_interp = calculate_coordinate_diffs(x_values_interp, y_values_interp)
            
            if plot_x:
                ax.plot(x_values, alpha=0.8, label='Original x')
                ax.plot(x_values_interp, label='Interpolated x')
                ax.set_ylabel('x (pixels)')
            else:
                ax.plot(distances, alpha=0.8, label='Original distance')
                ax.plot(distances_interp, label='Interpolated distance')
                ax.set_ylabel('Distance (pixels)')
            
            if scatter_outliers and np.any(outliers):
                outlier_frames = frames[outliers]
                outlier_distances = distances[outliers]
                ax.scatter(outlier_frames, outlier_distances, c='red', s=20, alpha=0.8, label='Outliers', zorder=5)
            
            if limit_scale_to_interpolated:
                y_min = np.min(distances_interp)
                y_max = np.max(distances_interp)
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
                else:
                    ax.set_ylim(y_min - 1, y_max + 1)
                
            ax.set_xlabel('Frame')
            ax.set_title(f'{bodypart}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f'keypoint_diffs_{rat_id}_{session}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")