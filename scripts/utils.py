import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, chi2
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from itertools import combinations


base_path = '/home/marlon/edu/mestrado/simba_analysis'
fig_path = os.path.join(base_path, 'figures')

summary_data_path = os.path.join(base_path, 'data/summary_data.csv')

session_types = ['s1', 's2', 't']
groups = ['salina', 'muscimol']
    

def load_data():
    df = pd.read_csv(summary_data_path)
    return df

def get_rearing(df):
    return df[df['rearing'] == 1].copy()

def get_area_and_direction_columns(session_type):
    if session_type == 's1':
        area_columns = ['OBJ_1', 'OBJ_2', 'OBJ_3', 'OBJ_4', 'NO_OBJ_1', 'NO_OBJ_2', 'NO_OBJ_3', 'NO_OBJ_4']
    elif session_type == 's2':
        area_columns = ['NOVEL_1', 'NOVEL_2', 'FORMER_1', 'FORMER_2', 'SAME_1', 'SAME_2', 'NEVER_1', 'NEVER_2']
    elif session_type == 't':
        area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER_1', 'NEVER_2', 'NEVER_3']
    
    direction_columns = ['dir_' + col for col in area_columns]
    
    return area_columns, direction_columns


def group_areas_and_directions(_df, session_type):
    df = _df.copy()

    if session_type == 's1':
        df['OBJ'] = merge_bool_columns(df, ['OBJ_1', 'OBJ_2', 'OBJ_3', 'OBJ_4'])
        df['NO_OBJ'] = merge_bool_columns(df, ['NO_OBJ_1', 'NO_OBJ_2', 'NO_OBJ_3', 'NO_OBJ_4'])
        df['dir_OBJ'] = merge_bool_columns(df, ['dir_OBJ_1', 'dir_OBJ_2', 'dir_OBJ_3', 'dir_OBJ_4'])
        df['dir_NO_OBJ'] = merge_bool_columns(df, ['dir_NO_OBJ_1', 'dir_NO_OBJ_2', 'dir_NO_OBJ_3', 'dir_NO_OBJ_4'])
        area_columns = ['OBJ', 'NO_OBJ']
    elif session_type == 's2':
        df['NOVEL'] = merge_bool_columns(df, ['NOVEL_1', 'NOVEL_2'])
        df['FORMER'] = merge_bool_columns(df, ['FORMER_1', 'FORMER_2'])
        df['SAME'] = merge_bool_columns(df, ['SAME_1', 'SAME_2'])
        df['NEVER'] = merge_bool_columns(df, ['NEVER_1', 'NEVER_2'])
        df['dir_NOVEL'] = merge_bool_columns(df, ['dir_NOVEL_1', 'dir_NOVEL_2'])
        df['dir_FORMER'] = merge_bool_columns(df, ['dir_FORMER_1', 'dir_FORMER_2'])
        df['dir_SAME'] = merge_bool_columns(df, ['dir_SAME_1', 'dir_SAME_2'])
        df['dir_NEVER'] = merge_bool_columns(df, ['dir_NEVER_1', 'dir_NEVER_2'])
        area_columns = ['NOVEL', 'FORMER', 'SAME', 'NEVER']
    elif session_type == 't':
        # df['NEVER'] = merge_bool_columns(df, ['NEVER_1', 'NEVER_2', 'NEVER_3'])
        # df['dir_NEVER'] = merge_bool_columns(df, ['dir_NEVER_1', 'dir_NEVER_2', 'dir_NEVER_3'])
        # area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER']
        area_columns = ['A1', 'A2', 'B1', 'B2', 'FORMER', 'NEVER_1', 'NEVER_2', 'NEVER_3']
    
    direction_columns = ['dir_' + col for col in area_columns]
    return df, area_columns, direction_columns

def merge_bool_columns(df, columns):
    return (df[columns].sum(axis=1) > 0).astype(int)


def add_significance_lines(ax, labels, values, alpha=0.05):
    for i, j in combinations(range(len(labels)), 2):
        if len(values[i]) > 0 and len(values[j]) > 0:
            stat, p_val = ttest_ind(values[i], values[j])
            if p_val < alpha:
                y_max = max(np.max(values[i]), np.max(values[j]))
                y_line = y_max * 1.1
                ax.plot([i, j], [y_line, y_line], 'k-', linewidth=1)
                ax.plot([i, i], [y_line, y_line*0.95], 'k-', linewidth=1)
                ax.plot([j, j], [y_line, y_line*0.95], 'k-', linewidth=1)
                ax.text((i+j)/2, y_line*1.05, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=8)