import os
import pandas as pd
import numpy as np
from scripts.utils import base_path

def save_to_csv(results_df):
    all_episodes = []
    
    for idx, row in results_df.iterrows():
        session = row['session']
        group = row['group']
        video = row['video']
        
        episode_durations = row['episode_durations']
        episode_areas = row['episode_areas']
        direction_during_rearing = row['direction_during_rearing']
        area_after_rearing_durations = row['area_after_rearing_durations']
        
        n_episodes = len(episode_durations)
        
        if len(area_after_rearing_durations) > 0:
            area_columns = area_after_rearing_durations.columns.tolist()
        else:
            if session == 's1':
                area_columns = ['OBJ', 'NO_OBJ']
            elif session == 's2':
                area_columns = ['NOVEL', 'FORMER', 'SAME', 'NEVER']
            elif session == 't':
                area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER_1', 'NEVER_2', 'NEVER_3']
            else:
                area_columns = []
        
        for i in range(n_episodes):
            episode_data = {
                'session': session,
                'video': video,
                'group': group,
                'duration': episode_durations[i],
                'area_during_rearing': episode_areas[i],
            }
            
            for direction_col in direction_during_rearing.columns:
                episode_data[f'direction_{direction_col}'] = direction_during_rearing.iloc[i][direction_col]
            
            if len(area_after_rearing_durations) > 0 and i < len(area_after_rearing_durations):
                for area_col in area_after_rearing_durations.columns:
                    episode_data[f'area_after_{area_col}'] = area_after_rearing_durations.iloc[i][area_col]
            else:
                for area_col in area_columns:
                    episode_data[f'area_after_{area_col}'] = np.nan
            
            all_episodes.append(episode_data)
    
    episodes_df = pd.DataFrame(all_episodes)
    
    output_path = os.path.join(base_path, 'data', 'processed', 'rearing_episodes.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    episodes_df.to_csv(output_path, index=False)