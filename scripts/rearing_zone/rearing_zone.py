import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import load_data, get_area_and_direction_columns, fig_path, session_types, get_rearing, groups, group_areas_and_directions
from scripts.rearing_zone.plots import (
    plot_rearing_episode, plot_mean_duration, plot_sum_duration, plot_total_episodes, 
    plot_area_during_rearing_counts, plot_area_during_rearing_mean_duration, 
    plot_area_after_rearing_durations
)

fps = 30
after_rearing_window_start = 2 # seconds
after_rearing_window_time = 10 # seconds

def calculate_durations(video_df):
    video_df = video_df.sort_values('frame').reset_index(drop=True)
    session_type = video_df['session'].iloc[0]
    video_df, area_columns, direction_columns = group_areas_and_directions(video_df, session_type)
    rearing = video_df['rearing'].values
    
    diff = np.diff(rearing, prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0 or len(ends) == 0:
        return None

    if len(starts) != len(ends):
        # Rearing não terminou
        ends = np.append(ends, len(rearing))
    
    valid_mask = (ends - starts) > 2
    starts = starts[valid_mask]
    ends = ends[valid_mask]
    if len(starts) == 0:
        return None

    total_area_durations = video_df[area_columns].sum() / fps
    
    total_episodes = len(starts)
    durations = (ends - starts) / fps

    area_during_rearing = []
    # area_during_rearing_ratio = []
    for start, end in zip(starts, ends):
        episode_areas = video_df.loc[start:end-1, area_columns]
        area_sums = episode_areas.sum()
        most_prominent_area = area_sums.idxmax()
        area_during_rearing.append(most_prominent_area)
        # total_sum = area_sums.sum()
        # most_prominent_area_ratio = area_sums[most_prominent_area] / total_sum
        # area_during_rearing_ratio.append(most_prominent_area_ratio)
    area_during_rearing = np.array(area_during_rearing)
    area_during_rearing_counts = pd.Series(area_during_rearing).value_counts()
    
    df_areas_during_rearing_durations = pd.DataFrame({'area': area_during_rearing, 'duration': durations})
    area_during_rearing_mean_duration = df_areas_during_rearing_durations.groupby('area')['duration'].mean()
    df_areas_during_rearing_durations_normalized = pd.DataFrame({'area': area_during_rearing, 'duration': durations / total_area_durations[area_during_rearing]})
    area_during_rearing_mean_duration_normalized = df_areas_during_rearing_durations_normalized.groupby('area')['duration'].mean()

    after_rearing_window_start_frames = after_rearing_window_start * fps
    after_rearing_window_frames = after_rearing_window_time * fps
    # Considerar apenas os finais que não têm início de outro rearing na window
    valid_rearing_ends = []
    for end in ends:
        window_start = end + after_rearing_window_start_frames
        window_end = window_start + after_rearing_window_frames
        next_starts = starts[starts >= window_start]
        if window_end > len(video_df):
            continue
        if len(next_starts) == 0 or next_starts[0] >= window_end:
            valid_rearing_ends.append(end)
    valid_rearing_ends = np.array(valid_rearing_ends)

    after_rearing_area_sums = []
    for end in valid_rearing_ends:
        window_start = end + after_rearing_window_start_frames
        window_end = window_start + after_rearing_window_frames
        window_areas = video_df.loc[window_start:window_end-1, area_columns]
        area_sums = window_areas.sum()
        after_rearing_area_sums.append(area_sums)
    
    if len(after_rearing_area_sums) > 0:
        area_after_rearing_durations = pd.DataFrame(after_rearing_area_sums) / fps
        area_after_rearing_durations_normalized = (pd.DataFrame(after_rearing_area_sums) / fps) / total_area_durations
    else:
        area_after_rearing_durations = pd.DataFrame(dtype=float)


    return (
        np.mean(durations),
        np.sum(durations),
        total_episodes,
        area_during_rearing_counts,
        area_during_rearing_mean_duration,
        area_during_rearing_mean_duration_normalized,
        area_after_rearing_durations,
        area_after_rearing_durations_normalized,
        rearing
    )
    
def process_video(group_df):
    result = calculate_durations(group_df)

    if result is None:
        return None
    
    mean_dur, sum_dur, total_episodes, area_during_rearing_counts, area_during_rearing_mean_duration, area_after_rearing_durations, rearing_arr = result
    return pd.Series({
        'mean_duration': mean_dur,
        'sum_duration': sum_dur,
        'total_episodes': total_episodes,
        'area_during_rearing_counts': area_during_rearing_counts,
        'area_during_rearing_mean_duration': area_during_rearing_mean_duration,
        'area_after_rearing_durations': area_after_rearing_durations,
        'rearing_array': rearing_arr
    })
    

def main():
    df = load_data()

    result = df.groupby(['session', 'group', 'video'], group_keys=False).apply(
        process_video, include_groups=True
    ).reset_index()
    
    result = result.dropna()
    for idx in result.index:
        video_name = result.loc[idx, 'video']
        rearing = result.loc[idx, 'rearing_array']
        plot_rearing_episode(video_name, rearing)
    
    plot_mean_duration(result)
    plot_sum_duration(result)
    plot_total_episodes(result)
    plot_area_during_rearing_counts(result)
    plot_area_during_rearing_mean_duration(result)
    plot_area_after_rearing_durations(result)
    
if __name__ == '__main__':
    main()