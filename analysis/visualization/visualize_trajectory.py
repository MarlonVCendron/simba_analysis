import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from analysis.consts import BODYPARTS, BASE_PATH
from analysis.load import iter_bodyparts
from analysis.outliers.outliers import detect_outliers, interpolate_outliers, calculate_coordinate_diffs


def visualize_trajectory(x_arrays, y_arrays, output_path, max_duration_seconds=None):
    n_frames = len(x_arrays[0])
    if max_duration_seconds is not None:
        max_frames = int(max_duration_seconds * 30)
        n_frames = min(n_frames, max_frames)
    n_bodyparts = len(x_arrays)
    
    x_min = min(np.nanmin(x) for x in x_arrays)
    x_max = max(np.nanmax(x) for x in x_arrays)
    y_min = min(np.nanmin(y) for y in y_arrays)
    y_max = max(np.nanmax(y) for y in y_arrays)
    
    valid_mask = [~(np.isnan(x) | np.isnan(y)) for x, y in zip(x_arrays, y_arrays)]
    valid_indices = [np.where(mask)[0] for mask in valid_mask]
    
    trail_length = 50
    line_data = [None] * n_frames
    trail_data = [None] * n_frames
    
    for frame in tqdm(range(n_frames), desc=f"Precomputing {output_path.name}"):
        frame_lines = []
        frame_trails = []
        for i in range(n_bodyparts):
            if len(valid_indices[i]) == 0:
                frame_lines.append((None, None))
                frame_trails.append((None, None))
                continue
                
            idx = np.searchsorted(valid_indices[i], frame, side='right')
            if idx > 0:
                last_valid_idx = valid_indices[i][idx - 1]
                frame_lines.append((x_arrays[i][last_valid_idx], y_arrays[i][last_valid_idx]))
                
                start_idx = max(0, idx - trail_length)
                trail_frames = valid_indices[i][start_idx:idx]
                if len(trail_frames) > 0:
                    frame_trails.append((x_arrays[i][trail_frames], y_arrays[i][trail_frames]))
                else:
                    frame_trails.append((None, None))
            else:
                frame_lines.append((None, None))
                frame_trails.append((None, None))
        
        line_data[frame] = frame_lines
        trail_data[frame] = frame_trails
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_bodyparts))
    lines = [ax.plot([], [], 'o-', color=colors[i], markersize=4)[0] for i in range(n_bodyparts)]
    trails = [ax.plot([], [], '-', color=colors[i], alpha=0.3, linewidth=1)[0] for i in range(n_bodyparts)]
    
    pbar = tqdm(total=n_frames, desc=f"Rendering {output_path.name}")
    
    def animate(frame):
        for i in range(n_bodyparts):
            x_line, y_line = line_data[frame][i]
            x_trail, y_trail = trail_data[frame][i]
            
            if x_line is not None:
                lines[i].set_data([x_line], [y_line])
            if x_trail is not None:
                trails[i].set_data(x_trail, y_trail)
        
        pbar.update(1)
        return lines + trails
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=33, blit=True, repeat=False)
    writer = animation.FFMpegWriter(fps=30, bitrate=1800, codec='libx264', extra_args=['-preset', 'fast'])
    anim.save(output_path, writer=writer, dpi=72)
    pbar.close()
    plt.close()


def visualize_all_trajectories(dlc_data, max_duration_seconds=60):
    output_dir = BASE_PATH / 'analysis' / 'figures' / 'trajectories'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_sessions_data = []
    for rat_id in dlc_data.keys():
        for session in dlc_data[rat_id]:
            data = dlc_data[rat_id][session]
            all_sessions_data.append((rat_id, session, data))
    
    for rat_id, session, data in all_sessions_data:
        x_arrays = []
        y_arrays = []
        x_interp_arrays = []
        y_interp_arrays = []
        for idx, bodypart in enumerate(BODYPARTS):
            for bp, x, y in iter_bodyparts(data, [bodypart]):
                if bp != bodypart:
                  continue

                x_values = x.values
                y_values = y.values
                x_diffs, y_diffs, distances = calculate_coordinate_diffs(x_values, y_values)
                break
            
            outliers = detect_outliers(distances)
            
            x_values_interp, y_values_interp = interpolate_outliers(x_values, y_values, outliers, method='cubic')
            x_diffs_interp, y_diffs_interp, _ = calculate_coordinate_diffs(x_values_interp, y_values_interp)

            x_arrays.append(x_values)
            y_arrays.append(y_values)
            x_interp_arrays.append(x_values_interp)
            y_interp_arrays.append(y_values_interp)

        visualize_trajectory(x_arrays, y_arrays, output_dir / f'trajectory_{rat_id}_{session}_original.mp4', max_duration_seconds)
        visualize_trajectory(x_interp_arrays, y_interp_arrays, output_dir / f'trajectory_{rat_id}_{session}.mp4', max_duration_seconds)
            