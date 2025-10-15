import os
import pandas as pd
import numpy as np
import re

base_path = '/home/marlon/edu/mestrado/simba_analysis/simba'

direction_paths = {
  't': 'TEST_zones/project_folder/logs/ROI_directionality_summary_20251013234406.csv',
  's1': 's1_zones/project_folder/logs/ROI_directionality_summary_20251013235719.csv',
  's2': 's2_zones/project_folder/logs/ROI_directionality_summary_20251013235740.csv',
}

rearing_paths = {
  't': 'TEST_zones/project_folder/csv/machine_results',
  's1': 's1_zones/project_folder/csv/machine_results',
  's2': 's2_zones/project_folder/csv/machine_results',
}

roi_paths = {
  't': 'TEST_zones/project_folder/csv/features_extracted',
  's1': 's1_zones/project_folder/csv/features_extracted',
  's2': 's2_zones/project_folder/csv/features_extracted',
}

group_index_path = os.path.join(base_path, 'src/data/group_index.csv')

def main():
  group_index = pd.read_csv(group_index_path)
  
  video_to_group = {}
  for _, row in group_index.iterrows():
    full_name = row['name']
    video_id = full_name.split('DLC_')[0]
    video_to_group[video_id] = row['group']
  
  direction_data = {}
  for key, value in direction_paths.items():
    csv_data = pd.read_csv(os.path.join(base_path, value))
    csv_data = filter_cols(csv_data, ['Frame', 'Video', 'ROI', 'Directing_BOOL'])
    direction_data[key] = csv_data
    
  rearing_data = {}
  print('Reading rearing data...')
  read_csvs(rearing_paths, rearing_data, ['frame', 'rearing'])

  roi_data = {}
  print('Reading ROI data...')
  read_csvs(roi_paths, roi_data)

  for key in roi_data.keys():
    keep_cols = []
    for file_key in roi_data[key].keys():
      if len(keep_cols) == 0:
        keep_cols = [col for col in roi_data[key][file_key].columns if 'mid_mid in zone' in col]
        keep_cols.append('frame')
        rename_cols = {col: col.split(' ')[0] for col in keep_cols}
        print(f'Keeping cols ROI {key}: {list(rename_cols.values())}')
      roi_data[key][file_key] = filter_cols(roi_data[key][file_key], keep_cols)
      roi_data[key][file_key] = roi_data[key][file_key].rename(columns=rename_cols)


  summary_data = pd.DataFrame()
  

  main_cols = ['session', 'video', 'group', 'frame']
  for session in ['t', 's1', 's2']:
    session_videos = list(roi_data[session].keys())
    rearing_videos = list(rearing_data[session].keys())
    
    assert session_videos.sort() == rearing_videos.sort(), f'Video mismatch for {session}: {session_videos} != {rearing_videos}'
    
    for video in session_videos:
      rearing_df = rearing_data[session][video].copy()
      roi_df = roi_data[session][video].copy()
      
      assert len(rearing_df) == len(roi_df), f"Frame count mismatch for {session}/{video}: rearing={len(rearing_df)}, roi={len(roi_df)}"
      
      merged_df = pd.merge(rearing_df, roi_df, on='frame', how='inner')
      
      merged_df['session'] = session
      merged_df['video'] = video
      
      if video in video_to_group:
        merged_df['group'] = video_to_group[video]
      else:
        print(f'Warning: No group found for video {video}, setting to unknown')
        merged_df['group'] = 'unknown'
      
      cols = main_cols + [col for col in merged_df.columns if col not in main_cols]
      merged_df = merged_df[cols]
      
      summary_data = pd.concat([summary_data, merged_df], ignore_index=True)
      
      print(f'    Added {len(merged_df)} rows for {session}/{video}')
  
  all_rois = set()
  for session in ['t', 's1', 's2']:
    if session in direction_data:
      all_rois.update(direction_data[session]['ROI'].unique())
  
  print(f'All ROIs: {sorted(all_rois)}')
  
  for roi in all_rois:
    summary_data[f'dir_{roi}'] = 0
  
  for session in ['t', 's1', 's2']:
    assert session in direction_data, f'Session {session} not found in direction data'
      
    print(f'Processing direction data for session: {session}')
    dir_df = direction_data[session]
    
    session_videos = dir_df['Video'].unique()
    print(f'  Found {len(session_videos)} videos in direction data: {session_videos}')
    
    for video in session_videos:
      video_dir_data = dir_df[dir_df['Video'] == video].copy()
      
      video_rois = video_dir_data['ROI'].unique()
      
      for roi in video_rois:
        roi_data = video_dir_data[video_dir_data['ROI'] == roi]
        
        directing_frames = roi_data[roi_data['Directing_BOOL'] == 1]['Frame'].values
        
        if len(directing_frames) > 0:
          mask = (summary_data['session'] == session) & \
                 (summary_data['video'] == video) & \
                 (summary_data['frame'].isin(directing_frames))
          
          summary_data.loc[mask, f'dir_{roi}'] = 1
  
  print(f'\nSummary data created with {len(summary_data)} total rows')
  print(f'Final columns: {list(summary_data.columns)}')
  
  
  cols_to_normalize = [col for col in summary_data.columns if col not in main_cols]
  
  for col in cols_to_normalize:
    summary_data[col] = summary_data[col].fillna(0)
    summary_data[col] = pd.to_numeric(summary_data[col], errors='coerce').fillna(0)
    summary_data[col] = (summary_data[col] != 0).astype(int)

    unique_vals = summary_data[col].unique()
    assert set(unique_vals).issubset({0, 1}), f"Column {col} is not binary: {unique_vals}"
  
  csv_output_path = os.path.join(base_path, 'src/data/summary_data.csv')
  summary_data.to_csv(csv_output_path, index=False)
  print(f'Summary data exported to CSV: {csv_output_path}')


def read_csvs(paths, data, keep_cols=None):
  for key, path in paths.items():
    data[key] = {}
    full_path = os.path.join(base_path, path)
    
    if os.path.exists(full_path):
      files = os.listdir(full_path)
      for i, filename in enumerate(files):
        if filename.endswith('.csv'):
          file_key = filename[:-4]
          file_path = os.path.join(full_path, filename)
          
          csv_data = pd.read_csv(file_path)
          if "Rearing" in csv_data.columns:
            csv_data = csv_data.rename(columns={"Rearing": "rearing"})
          if "Unnamed: 0" in csv_data.columns:
            csv_data = csv_data.rename(columns={"Unnamed: 0": "frame"})
          
          if keep_cols is not None:
            csv_data = filter_cols(csv_data, keep_cols)

          data[key][file_key] = csv_data
          print(f'Loaded data {i+1}/{len(files)}: {key}/{file_key}')

def filter_cols(csv_data, keep_cols):
  for col in keep_cols:
    assert col in csv_data.columns, f'Column {col} not found in {csv_data.columns}'
  csv_data = csv_data[keep_cols]
  return csv_data

if __name__ == '__main__':
  main()