import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict

from utils import load_data, base_path, session_types, groups

def calculate_rearing_metrics(df):
    """Calculate rearing frequency, duration, and other metrics per video"""
    rearing_metrics = []
    
    for video in df['video'].unique():
        video_data = df[df['video'] == video].copy()
        session = video_data['session'].iloc[0]
        group = video_data['group'].iloc[0]
        
        # Basic video info
        total_frames = len(video_data)
        total_duration = total_frames  # Assuming 1 frame = 1 time unit
        
        # Rearing metrics
        rearing_frames = video_data[video_data['rearing'] == 1]
        rearing_count = len(rearing_frames)
        rearing_frequency = rearing_count / total_duration if total_duration > 0 else 0
        
        # Calculate rearing duration (consecutive rearing frames)
        rearing_duration = 0
        current_duration = 0
        rearing_episodes = []
        
        for idx, row in video_data.iterrows():
            if row['rearing'] == 1:
                current_duration += 1
            else:
                if current_duration > 0:
                    rearing_episodes.append(current_duration)
                    rearing_duration += current_duration
                    current_duration = 0
        
        # Handle case where rearing continues to the end
        if current_duration > 0:
            rearing_episodes.append(current_duration)
            rearing_duration += current_duration
        
        mean_rearing_duration = np.mean(rearing_episodes) if rearing_episodes else 0
        max_rearing_duration = max(rearing_episodes) if rearing_episodes else 0
        
        rearing_metrics.append({
            'session': session,
            'video': video,
            'group': group,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'rearing_count': rearing_count,
            'rearing_frequency': rearing_frequency,
            'rearing_duration': rearing_duration,
            'mean_rearing_duration': mean_rearing_duration,
            'max_rearing_duration': max_rearing_duration,
            'rearing_episodes_count': len(rearing_episodes)
        })
    
    return pd.DataFrame(rearing_metrics)

def calculate_area_metrics(df):
    """Calculate time spent in each area per video"""
    area_metrics = []
    
    for video in df['video'].unique():
        video_data = df[df['video'] == video].copy()
        session = video_data['session'].iloc[0]
        group = video_data['group'].iloc[0]
        total_frames = len(video_data)
        
        # Get area columns based on session type
        if session == 's1':
            area_columns = ['OBJ_1', 'OBJ_2', 'OBJ_3', 'OBJ_4', 'NO_OBJ_1', 'NO_OBJ_2', 'NO_OBJ_3', 'NO_OBJ_4']
        elif session == 's2':
            area_columns = ['NOVEL_1', 'NOVEL_2', 'FORMER_1', 'FORMER_2', 'SAME_1', 'SAME_2', 'NEVER_1', 'NEVER_2']
        elif session == 't':
            area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER_1', 'NEVER_2', 'NEVER_3']
        else:
            continue
        
        # Calculate time spent in each area
        area_data = {
            'session': session,
            'video': video,
            'group': group,
            'total_frames': total_frames
        }
        
        for area in area_columns:
            if area in video_data.columns:
                time_in_area = video_data[area].sum()
                percentage_in_area = (time_in_area / total_frames) * 100 if total_frames > 0 else 0
                area_data[f'time_{area}'] = time_in_area
                area_data[f'percentage_{area}'] = percentage_in_area
        
        area_metrics.append(area_data)
    
    return pd.DataFrame(area_metrics)

def calculate_rearing_by_area(df):
    """Calculate rearing events and duration by area"""
    rearing_by_area = []
    
    for video in df['video'].unique():
        video_data = df[df['video'] == video].copy()
        session = video_data['session'].iloc[0]
        group = video_data['group'].iloc[0]
        
        # Get area columns based on session type
        if session == 's1':
            area_columns = ['OBJ_1', 'OBJ_2', 'OBJ_3', 'OBJ_4', 'NO_OBJ_1', 'NO_OBJ_2', 'NO_OBJ_3', 'NO_OBJ_4']
        elif session == 's2':
            area_columns = ['NOVEL_1', 'NOVEL_2', 'FORMER_1', 'FORMER_2', 'SAME_1', 'SAME_2', 'NEVER_1', 'NEVER_2']
        elif session == 't':
            area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER_1', 'NEVER_2', 'NEVER_3']
        else:
            continue
        
        # Calculate rearing in each area
        area_data = {
            'session': session,
            'video': video,
            'group': group
        }
        
        for area in area_columns:
            if area in video_data.columns:
                # Rearing events in this area
                rearing_in_area = video_data[(video_data['rearing'] == 1) & (video_data[area] == 1)]
                rearing_count = len(rearing_in_area)
                rearing_duration = rearing_in_area['rearing'].sum()
                
                # Time spent in area
                time_in_area = video_data[area].sum()
                
                # Rearing frequency per time in area
                rearing_frequency_in_area = rearing_count / time_in_area if time_in_area > 0 else 0
                
                area_data[f'rearing_count_{area}'] = rearing_count
                area_data[f'rearing_duration_{area}'] = rearing_duration
                area_data[f'time_{area}'] = time_in_area
                area_data[f'rearing_frequency_{area}'] = rearing_frequency_in_area
        
        rearing_by_area.append(area_data)
    
    return pd.DataFrame(rearing_by_area)

def calculate_direction_metrics(df):
    """Calculate rearing direction metrics"""
    direction_metrics = []
    
    for video in df['video'].unique():
        video_data = df[df['video'] == video].copy()
        session = video_data['session'].iloc[0]
        group = video_data['group'].iloc[0]
        
        # Get direction columns
        direction_columns = [col for col in video_data.columns if col.startswith('dir_')]
        
        direction_data = {
            'session': session,
            'video': video,
            'group': group
        }
        
        for direction_col in direction_columns:
            if direction_col in video_data.columns:
                # Rearing events facing this direction
                rearing_facing_direction = video_data[(video_data['rearing'] == 1) & (video_data[direction_col] == 1)]
                rearing_count = len(rearing_facing_direction)
                rearing_duration = rearing_count
                
                # Time facing this direction
                time_facing_direction = video_data[direction_col].sum()
                
                # Rearing frequency when facing this direction
                rearing_frequency = rearing_count / time_facing_direction if time_facing_direction > 0 else 0
                
                direction_data[f'rearing_count_{direction_col}'] = rearing_count
                direction_data[f'rearing_duration_{direction_col}'] = rearing_duration
                direction_data[f'time_{direction_col}'] = time_facing_direction
                direction_data[f'rearing_frequency_{direction_col}'] = rearing_frequency
        
        direction_metrics.append(direction_data)
    
    return pd.DataFrame(direction_metrics)

def calculate_session_summary(df):
    """Calculate summary statistics by session and group"""
    summary_data = []
    
    for session in session_types:
        session_data = df[df['session'] == session]
        if len(session_data) == 0:
            continue
            
        for group in groups:
            group_data = session_data[session_data['group'] == group]
            if len(group_data) == 0:
                continue
            
            # Basic metrics
            total_videos = group_data['video'].nunique()
            total_frames = group_data['frame'].nunique()
            
            # Rearing metrics
            rearing_events = group_data[group_data['rearing'] == 1]
            total_rearing = len(rearing_events)
            rearing_frequency = total_rearing / total_frames if total_frames > 0 else 0
            
            # Per video averages
            video_metrics = group_data.groupby('video').agg({
                'rearing': ['sum', 'count']
            }).reset_index()
            
            avg_rearing_per_video = video_metrics[('rearing', 'sum')].mean()
            avg_duration_per_video = video_metrics[('rearing', 'count')].mean()
            
            summary_data.append({
                'session': session,
                'group': group,
                'total_videos': total_videos,
                'total_frames': total_frames,
                'total_rearing_events': total_rearing,
                'rearing_frequency': rearing_frequency,
                'avg_rearing_per_video': avg_rearing_per_video,
                'avg_duration_per_video': avg_duration_per_video
            })
    
    return pd.DataFrame(summary_data)

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows of data")
    
    # Create output directory
    output_dir = os.path.join(base_path, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Calculating rearing metrics...")
    rearing_metrics = calculate_rearing_metrics(df)
    rearing_metrics.to_csv(os.path.join(output_dir, 'rearing_metrics.csv'), index=False)
    print(f"Saved rearing metrics: {len(rearing_metrics)} videos")
    
    print("Calculating area metrics...")
    area_metrics = calculate_area_metrics(df)
    area_metrics.to_csv(os.path.join(output_dir, 'area_metrics.csv'), index=False)
    print(f"Saved area metrics: {len(area_metrics)} videos")
    
    print("Calculating rearing by area...")
    rearing_by_area = calculate_rearing_by_area(df)
    rearing_by_area.to_csv(os.path.join(output_dir, 'rearing_by_area.csv'), index=False)
    print(f"Saved rearing by area: {len(rearing_by_area)} videos")
    
    print("Calculating direction metrics...")
    direction_metrics = calculate_direction_metrics(df)
    direction_metrics.to_csv(os.path.join(output_dir, 'direction_metrics.csv'), index=False)
    print(f"Saved direction metrics: {len(direction_metrics)} videos")
    
    print("Calculating session summary...")
    session_summary = calculate_session_summary(df)
    session_summary.to_csv(os.path.join(output_dir, 'session_summary.csv'), index=False)
    print(f"Saved session summary: {len(session_summary)} sessions")
    
    print("\nPreprocessing complete! Files saved to:", output_dir)
    print("\nGenerated files:")
    print("- rearing_metrics.csv: Rearing frequency, duration, and episodes per video")
    print("- area_metrics.csv: Time spent in each area per video")
    print("- rearing_by_area.csv: Rearing events and duration by area")
    print("- direction_metrics.csv: Rearing direction analysis")
    print("- session_summary.csv: Summary statistics by session and group")

if __name__ == '__main__':
    main()