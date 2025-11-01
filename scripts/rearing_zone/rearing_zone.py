import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import load_data, get_area_and_direction_columns, fig_path, session_types, get_rearing, groups, group_areas_and_directions
from scripts.rearing_zone.plots import plot_rearing_episode, plot_mean_duration, plot_sum_duration, plot_total_episodes, plot_area_during_rearing_counts

fps = 30
after_rearing_window = 10 # seconds

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
    
    total_episodes = len(starts)
    durations = (ends - starts) / fps
    
    after_rearing_window_frames = after_rearing_window * fps

    area_during_rearing = []
    area_during_rearing_ratio = []
    for start, end in zip(starts, ends):
        episode_areas = video_df.loc[start:end-1, area_columns]
        area_sums = episode_areas.sum()
        most_prominent_area = area_sums.idxmax()
        total_sum = area_sums.sum()
        most_prominent_area_ratio = area_sums[most_prominent_area] / total_sum
        area_during_rearing.append(most_prominent_area)
        area_during_rearing_ratio.append(most_prominent_area_ratio)
    area_during_rearing = np.array(area_during_rearing)
    area_during_rearing_counts = pd.Series(area_during_rearing).value_counts()

    # Considerar apenas os finais que não têm início de outro rearing na window
    valid_rearing_ends = []
    for end in ends:
        window_start = end
        window_end = end + after_rearing_window_frames
        next_starts = starts[starts >= window_start]
        if len(next_starts) == 0 or next_starts[0] >= window_end:
            valid_rearing_ends.append(end)
    valid_rearing_ends = np.array(valid_rearing_ends)


    
    # print(video_df[area_columns])
    # print(video_df[starts])

    # print(video_df.head())
    
    
    return np.mean(durations), np.sum(durations), total_episodes, area_during_rearing_counts, rearing
    
def process_video(group_df):
    result = calculate_durations(group_df)

    if result is None:
        return None
    
    mean_dur, sum_dur, total_episodes, area_during_rearing_counts, rearing_arr = result
    return pd.Series({
        'mean_duration': mean_dur,
        'sum_duration': sum_dur,
        'total_episodes': total_episodes,
        'area_during_rearing_counts': area_during_rearing_counts,
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
    
if __name__ == '__main__':
    main()