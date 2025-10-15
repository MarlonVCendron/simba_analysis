import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, chi2
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from itertools import combinations

from utils import fig_path, get_rearing, session_types

def rearing_by_group(df, session_type):
    ct = pd.crosstab(df['group'], df['rearing'])
    chi2, p, dof, expected = chi2_contingency(ct)

    rearing_counts = get_rearing(df).groupby('group').size()
    muscimol_count = rearing_counts.get('muscimol', 0)
    salina_count = rearing_counts.get('salina', 0)
    
    plt.figure(figsize=(6, 4))
    plt.bar(['muscimol', 'salina'], [muscimol_count, salina_count])
    plt.title(f'Rearing por grupo {session_type} (p={p:.3f} chi2={chi2:.3f})')
    plt.ylabel('Total')
    plt.savefig(os.path.join(fig_path, f'rearing_counts/rearing_by_group_{session_type}.png'))

def rearing_by_session(df):
    ct = pd.crosstab(df['session'], df['rearing'])
    chi2, p, dof, expected = chi2_contingency(ct)

    rearing_counts = get_rearing(df).groupby('session').size()
    t_count = rearing_counts.get('t', 0)
    s1_count = rearing_counts.get('s1', 0)
    s2_count = rearing_counts.get('s2', 0)

    plt.figure(figsize=(6, 4))
    plt.bar(['t', 's1', 's2'], [t_count, s1_count, s2_count])
    plt.title(f'Rearing por sessão (p={p:.3f} chi2={chi2:.3f})')
    plt.ylabel('Total')
    plt.savefig(os.path.join(fig_path, 'rearing_counts/rearing_by_session.png'))

def rearing_by_video(df, session_type):
    rearing_df = get_rearing(df)
    muscimol_df = rearing_df[rearing_df['group'] == 'muscimol']
    salina_df = rearing_df[rearing_df['group'] == 'salina']
    
    muscimol_video_counts = muscimol_df.groupby('video').size().sort_values(ascending=False)
    salina_video_counts = salina_df.groupby('video').size().sort_values(ascending=False)
    
    muscimol_videos = muscimol_video_counts.index.tolist()
    salina_videos = salina_video_counts.index.tolist()
    
    muscimol_counts = [muscimol_video_counts[v] for v in muscimol_videos]
    salina_counts = [salina_video_counts[v] for v in salina_videos]
    
    muscimol_stat, muscimol_p = shapiro(muscimol_counts)
    salina_stat, salina_p = shapiro(salina_counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    ax1.bar(muscimol_videos, muscimol_counts)
    ax1.set_title(f'Vídeos Muscimol (shapiro p={muscimol_p:.3f})')
    ax1.set_ylabel('Total')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(salina_videos, salina_counts)
    ax2.set_title(f'Vídeos Salina (shapiro p={salina_p:.3f})')
    ax2.set_ylabel('Total')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'rearing_counts/rearing_by_video_{session_type}.png'))

    