import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, chi2
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from itertools import combinations

from utils import fig_path, get_rearing, session_types, get_area_and_direction_columns

def _plot_group_comparison(df, columns, session_type, column_type, xlabel, title_prefix, filename_prefix):
    muscimol_counts = []
    salina_counts = []
    p_values = []
    
    for column in columns:
        column_df = df[df[column] == 1]

        ct = pd.crosstab(column_df['group'], column_df['rearing'])
        chi2, p, dof, expected = chi2_contingency(ct)
        significant = p < 0.05
        print(f'{significant * "*"} {session_type} {column} (p={p:.3f} chi2={chi2:.3f})')
        
        rearing_counts = get_rearing(column_df).groupby('group').size()
        muscimol_count = rearing_counts.get('muscimol', 0)
        salina_count = rearing_counts.get('salina', 0)
        
        muscimol_counts.append(muscimol_count)
        salina_counts.append(salina_count)
        p_values.append(p)
    
    x = np.arange(len(columns))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, muscimol_counts, width, label='muscimol', color='red')
    bars2 = plt.bar(x + width/2, salina_counts, width, label='salina', color='blue')
    
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            max_height = max(muscimol_counts[i], salina_counts[i])
            plt.text(i, max_height + max_height*0.05, f'* p={p_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel(xlabel)
    plt.ylabel('Total')
    plt.title(f'{title_prefix} {session_type}')
    plt.xticks(x, columns, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(fig_path, 'areas'), exist_ok=True)
    plt.savefig(os.path.join(fig_path, f'areas/{filename_prefix}_{session_type}.png'))

def areas_by_group(df, session_type):
    df = df.copy()
    area_columns, direction_columns = get_area_and_direction_columns(df, session_type)
    _plot_group_comparison(df, area_columns, session_type, 'area', 'Áreas', 'Rearing por área e grupo', 'areas_by_group')

def direction_by_group(df, session_type):
    df = df.copy()
    area_columns, direction_columns = get_area_and_direction_columns(df, session_type)
    _plot_group_comparison(df, direction_columns, session_type, 'direction', 'Direções', 'Rearing por direção e grupo', 'direction_by_group')

def mean_rears_per_area(df, session_type):
    df = df.copy()
    area_columns, _ = get_area_and_direction_columns(df, session_type)
    
    mean_rears = []
    areas = []
    p_values = []
    
    for area in area_columns:
        area_data = df[df[area] == 1]
        if len(area_data) > 0:
            mean_rear = area_data.groupby('group')['rearing'].mean()
            mean_rears.append(mean_rear)
            areas.append(area)
            
            ct = pd.crosstab(area_data['group'], area_data['rearing'])
            chi2, p, dof, expected = chi2_contingency(ct)
            
            p_values.append(p)
    
    if mean_rears:
        mean_df = pd.concat(mean_rears, axis=1, keys=areas)
        
        plt.figure(figsize=(12, 6))
        ax = mean_df.T.plot(kind='bar', color=['red', 'blue'])
        plt.xlabel('Áreas')
        plt.ylabel('Média de Rearing')
        plt.title(f'Média de Rearing por Área - {session_type}')
        plt.legend(['muscimol', 'salina'])
        plt.xticks(rotation=45, ha='right')
        
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                max_height = max(mean_df.iloc[0, i], mean_df.iloc[1, i])
                plt.text(i, max_height + max_height*0.1, f'* p={p_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.join(fig_path, 'areas'), exist_ok=True)
        plt.savefig(os.path.join(fig_path, f'areas/mean_rears_per_area_{session_type}.png'))

def rearing_direction_relationship(df, session_type):
    df = df.copy()
    area_columns, direction_columns = get_area_and_direction_columns(df, session_type)
    
    correlation_matrix = np.zeros((len(area_columns), len(direction_columns)))
    
    
    rearing_df = get_rearing(df)
    for i, rearing_area in enumerate(area_columns):
        for j, direction_area in enumerate(direction_columns):
            corr = rearing_df[rearing_area].corr(rearing_df[direction_area])
            correlation_matrix[i, j] = corr if not np.isnan(corr) else 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=direction_columns, yticklabels=area_columns,
                cbar_kws={'label': 'Correlation'})
    plt.title(f'Correlação área-direção - {session_type}')
    plt.xlabel('Direcionamento')
    plt.ylabel('Áreas')
    
    plt.tight_layout()
    os.makedirs(os.path.join(fig_path, 'areas'), exist_ok=True)
    plt.savefig(os.path.join(fig_path, f'areas/rearing_direction_relationship_{session_type}.png'))
    plt.close()

