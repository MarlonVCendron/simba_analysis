import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import fig_path, session_types, groups

colors = {
    'saline': '#145faa',
    'muscimol': '#828287'
}

base_fig_path = os.path.join(fig_path, 'rearing_zone')

def plot_rearing_episode(video_name, rearing_col):
    plt.figure(figsize=(12, 2))
    plt.plot(rearing_col)
    plt.ylabel('Rearing')
    plt.xlabel('Frame')
    plt.ylim(-0.1, 1.1)
    
    fig_dir = os.path.join(base_fig_path, 'rearing_episodes')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f'{video_name}.png'))
    plt.close()

def plot_mean_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='session', y='mean_duration', hue='group', 
                order=session_types, hue_order=groups,
                palette=[colors['saline'], colors['muscimol']])
    plt.xlabel('Session')
    plt.ylabel('Mean Duration (s)')
    plt.title('Mean Duration of Rearing Episodes')
    plt.legend(title='Group')
    
    plt.savefig(os.path.join(base_fig_path, 'mean_duration.png'))
    plt.close()

def plot_sum_duration(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='session', y='sum_duration', hue='group',
                order=session_types, hue_order=groups,
                palette=[colors['saline'], colors['muscimol']])
    plt.xlabel('Session')
    plt.ylabel('Sum Duration (s)')
    plt.title('Total Duration of Rearing Episodes')
    plt.legend(title='Group')
    
    plt.savefig(os.path.join(base_fig_path, 'sum_duration.png'))
    plt.close()

def plot_total_episodes(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='session', y='total_episodes', hue='group',
                order=session_types, hue_order=groups,
                palette=[colors['saline'], colors['muscimol']])
    plt.xlabel('Session')
    plt.ylabel('Total Episodes')
    plt.title('Total Number of Rearing Episodes')
    plt.legend(title='Group')
    plt.savefig(os.path.join(base_fig_path, 'total_episodes.png'))
    plt.close()
    
def plot_area_during_rearing_counts(data):
    plot_data = []
    for _, row in data.iterrows():
        area_counts = row['area_during_rearing_counts']
        for area, count in area_counts.items():
            plot_data.append({'session': row['session'], 'group': row['group'], 'area': area, 'count': count})
    
    df_plot = pd.DataFrame(plot_data)
    # df_plot = df_plot.groupby(['session', 'group', 'area'], as_index=False)['count'].sum()
    
    fig, axes = plt.subplots(1, len(session_types), figsize=(6*len(session_types), 6), sharey=True)
    if len(session_types) == 1:
        axes = [axes]
    
    for ax, session in zip(axes, session_types):
        session_data = df_plot[df_plot['session'] == session]
        if len(session_data) > 0:
            areas = sorted(session_data['area'].unique())
            sns.barplot(data=session_data, x='area', y='count', hue='group',
                       order=areas, hue_order=groups,
                       palette=[colors['saline'], colors['muscimol']], ax=ax)
            ax.set_title(f'Session {session.upper()}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Group')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Most Prominent Area During Rearing Counts', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_during_rearing_counts.png'))
    plt.close()

def plot_area_during_rearing_mean_duration(data):
    plot_data = []
    for _, row in data.iterrows():
        area_mean_durations = row['area_during_rearing_mean_duration']
        for area, mean_duration in area_mean_durations.items():
            plot_data.append({'session': row['session'], 'group': row['group'], 'area': area, 'mean_duration': mean_duration})
    
    df_plot = pd.DataFrame(plot_data)
    # df_plot = df_plot.groupby(['session', 'group', 'area'], as_index=False)['mean_duration'].mean()
    
    fig, axes = plt.subplots(1, len(session_types), figsize=(6*len(session_types), 6), sharey=True)
    if len(session_types) == 1:
        axes = [axes]
    
    for ax, session in zip(axes, session_types):
        session_data = df_plot[df_plot['session'] == session]
        if len(session_data) > 0:
            areas = sorted(session_data['area'].unique())
            sns.barplot(data=session_data, x='area', y='mean_duration', hue='group',
                       order=areas, hue_order=groups,
                       palette=[colors['saline'], colors['muscimol']], ax=ax)
            ax.set_title(f'Session {session.upper()}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Mean Duration (s)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Group')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Most Prominent Area During Rearing Mean Duration', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_during_rearing_mean_duration.png'))
    plt.close()

def plot_area_after_rearing_durations(data):
    plot_data = []
    for _, row in data.iterrows():
        area_durations = row['area_after_rearing_durations']
        for area in area_durations.columns:
            durations = area_durations[area].values
            for duration in durations:
                plot_data.append({'session': row['session'], 'group': row['group'], 'area': area, 'duration': duration})
    
    df_plot = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(1, len(session_types), figsize=(6*len(session_types), 6), sharey=True)
    if len(session_types) == 1:
        axes = [axes]
    
    for ax, session in zip(axes, session_types):
        session_data = df_plot[df_plot['session'] == session]
        if len(session_data) > 0:
            areas = sorted(session_data['area'].unique())
            sns.barplot(data=session_data, x='area', y='duration', hue='group',
                       order=areas, hue_order=groups,
                       palette=[colors['saline'], colors['muscimol']], ax=ax)
            ax.set_title(f'Session {session.upper()}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Duration (s)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Group')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Area After Rearing Duration', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_after_rearing_durations.png'))
    plt.close()

