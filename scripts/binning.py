import pandas as pd
import numpy as np

fps = 30

def bin_frames_to_seconds(input_file, output_file, frames_per_bin=fps):
    df = pd.read_csv(input_file)
    grouped = df.groupby(['session', 'video'])

    skip_cols = ['session', 'video', 'group', 'frame']

    binned_data = []
    for (session, video), group_df in grouped:
        group_df = group_df.copy()
        
        group_df['second'] = (group_df['frame'] // (frames_per_bin))
        
        binary_columns = [col for col in group_df.columns if col not in skip_cols]
        
        binned_group = group_df.groupby('second', as_index=False)[binary_columns].sum() / fps

        binned_group['session'] = session
        binned_group['video'] = video
        binned_group['group'] = group_df['group'].iloc[0]
        # binned_group['second'] = binned_group['second'] // fps
        
        column_order = ['session', 'video', 'group', 'second'] + [col for col in binary_columns if col != 'second']
        binned_group = binned_group[column_order]
        
        binned_data.append(binned_group)
    
    result_df = pd.concat(binned_data, ignore_index=True)
    
    result_df.to_csv(output_file, index=False)

    return result_df

if __name__ == "__main__":
    # frames_per_bin = 150
    # frames_per_bin = 30
    frames_per_bin = 1800
    input_file = "/home/marlon/edu/mestrado/simba_analysis/data/summary_data.csv"
    output_file = f"/home/marlon/edu/mestrado/simba_analysis/data/processed/time_bins.csv"
    
    result_df = bin_frames_to_seconds(input_file, output_file, frames_per_bin=frames_per_bin)
    print(result_df.head())
