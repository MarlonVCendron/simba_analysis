import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scripts.utils import fig_path, session_types, groups

colors = {
    'saline': '#145faa',
    'salina': '#145faa',
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
    fig.suptitle('Area During Rearing Counts', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_during_rearing_counts.png'))
    plt.close()

def plot_area_during_rearing_mean_duration(data, normalized=False):
    normalized_suffix = "_normalized" if normalized else ""
    plot_data = []
    for _, row in data.iterrows():
        area_mean_durations = row[f'area_during_rearing_mean_duration{normalized_suffix}']
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
    fig.suptitle(f'Area During Rearing Duration {"Normalized" if normalized else ""}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, f'area_during_rearing_mean_duration{normalized_suffix}.png'))
    plt.close()

def plot_area_after_rearing_durations(data, normalized=False):
    normalized_suffix = "_normalized" if normalized else ""
    plot_data = []
    for _, row in data.iterrows():
        area_durations = row[f'area_after_rearing_durations{normalized_suffix}']
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
    fig.suptitle(f'Area After Rearing Duration {"Normalized" if normalized else ""}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, f'area_after_rearing_durations{normalized_suffix}.png'))
    plt.close()

    
def plot_direction_during_rearing(data):
    plot_data = []
    for _, row in data.iterrows():
        direction_durations = row['direction_during_rearing']
        for direction in direction_durations.columns:
            durations = direction_durations[direction].values
            for duration in durations:
                plot_data.append({'session': row['session'], 'group': row['group'], 'direction': direction, 'duration': duration})
    
    df_plot = pd.DataFrame(plot_data)
    
    fig, axes = plt.subplots(1, len(session_types), figsize=(6*len(session_types), 6), sharey=True)
    if len(session_types) == 1:
        axes = [axes]
    
    for ax, session in zip(axes, session_types):
        session_data = df_plot[df_plot['session'] == session]
        if len(session_data) > 0:
            directions = sorted(session_data['direction'].unique())
            sns.barplot(data=session_data, x='direction', y='duration', hue='group',
                       order=directions, hue_order=groups,
                       palette=[colors['saline'], colors['muscimol']], ax=ax)
            ax.set_title(f'Session {session.upper()}')
            ax.set_xlabel('Direction')
            ax.set_ylabel('Duration (s)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Group')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f'Direction During Rearing Duration', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, f'direction_during_rearing.png'))
    plt.close()

def plot_area_correlation_matrix(data):
    fig, axes = plt.subplots(len(session_types), len(groups), figsize=(8*len(groups), 6*len(session_types)))
    
    session_matrices = {}
    for i, session in enumerate(session_types):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            session_group_data = data[(data['session'] == session) & (data['group'] == group)]
            
            aggregated_matrix = None
            for _, row in session_group_data.iterrows():
                matrix = row['area_correlation_matrix']
                if aggregated_matrix is None:
                    aggregated_matrix = matrix.copy()
                else:
                    aggregated_matrix = aggregated_matrix.add(matrix, fill_value=0)
            
            aggregated_matrix = aggregated_matrix.div(aggregated_matrix.sum(axis=1), axis=0).fillna(0)
            if session not in session_matrices:
                session_matrices[session] = {}
            session_matrices[session][group] = aggregated_matrix
            
            annot_data = pd.DataFrame(index=aggregated_matrix.index, columns=aggregated_matrix.columns,
                                     data=[[f'{val:.3f}' for val in row] for row in aggregated_matrix.values])
            sns.heatmap(aggregated_matrix, annot=annot_data, fmt='', cmap='RdBu_r', ax=ax, 
                       cbar_kws={'label': 'Proportion'}, annot_kws={'fontsize': 8}, 
                       vmin=0, vmax=1, xticklabels=True, yticklabels=True)
            ax.set_title(f'Session {session.upper()} - {group.capitalize()}')
            ax.set_xlabel('Area After Rearing')
            ax.set_ylabel('Area During Rearing')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
    
    for i, session in enumerate(session_types):
        matrix1 = session_matrices[session][groups[0]].values.flatten()
        matrix2 = session_matrices[session][groups[1]].values.flatten()
        corr = np.corrcoef(matrix1, matrix2)[0, 1]
        ax = axes[i, 0]
        ax.text(0.02, 0.98, f'$\\rho$={corr:.3f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Area Correlation Matrix', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_correlation_matrix.png'))
    plt.close()

def plot_area_correlation_graph(data):
    fig, axes = plt.subplots(len(session_types), len(groups), figsize=(8*len(groups), 6*len(session_types)))
    
    session_matrices = {}
    for i, session in enumerate(session_types):
        for j, group in enumerate(groups):
            ax = axes[i, j]
            session_group_data = data[(data['session'] == session) & (data['group'] == group)]
            
            aggregated_matrix = None
            for _, row in session_group_data.iterrows():
                matrix = row['area_correlation_matrix']
                if aggregated_matrix is None:
                    aggregated_matrix = matrix.copy()
                else:
                    aggregated_matrix = aggregated_matrix.add(matrix, fill_value=0)
            
            aggregated_matrix = aggregated_matrix.div(aggregated_matrix.sum(axis=1), axis=0).fillna(0)
            if session not in session_matrices:
                session_matrices[session] = {}
            session_matrices[session][group] = aggregated_matrix
            
            G = nx.DiGraph()
            for from_area in aggregated_matrix.index:
                for to_area in aggregated_matrix.columns:
                    weight = aggregated_matrix.loc[from_area, to_area]
                    if weight > 0:
                        G.add_edge(from_area, to_area, weight=weight)
            
            if len(G.nodes()) > 0:
                pos = nx.circular_layout(G)
                
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                edge_widths = [w * 15 + 1.0 for w in weights]
                
                node_colors = colors[group]
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                      node_size=2000, alpha=0.85, linewidths=2, 
                                      edgecolors='white')
                
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold',
                                      font_color='white')
                
                if weights:
                    cmap = plt.cm.RdBu_r
                    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
                    edge_colors = [cmap(norm(w)) for w in weights]
                else:
                    edge_colors = 'gray'
                
                nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                                      edge_color=edge_colors, alpha=0.7, arrows=True, 
                                      arrowsize=25, arrowstyle='->', connectionstyle='arc3,rad=0.1')
                
                ax.set_facecolor('#f5f5f5')
            
            ax.set_title(f'Session {session.upper()} - {group.capitalize()}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.axis('off')
    
    for i, session in enumerate(session_types):
        matrix1 = session_matrices[session][groups[0]].values.flatten()
        matrix2 = session_matrices[session][groups[1]].values.flatten()
        corr = np.corrcoef(matrix1, matrix2)[0, 1]
        ax = axes[i, 0]
        ax.text(0.02, 0.98, f'$\\rho$={corr:.3f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('Area Correlation Graph', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(base_fig_path, 'area_correlation_graph.png'))
    plt.close()
