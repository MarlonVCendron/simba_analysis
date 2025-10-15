import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, chi2
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from itertools import combinations

from utils import load_data, get_area_and_direction_columns, fig_path, session_types, get_rearing
from rearing_counts import rearing_by_session, rearing_by_group, rearing_by_video
from areas import areas_by_group, direction_by_group

def analyze_session(df, session_type):
    session_df = df[df['session'] == session_type].copy()

    rearing_by_group(session_df, session_type)
    rearing_by_video(session_df, session_type)

    areas_by_group(session_df, session_type)
    direction_by_group(session_df, session_type)


def main():
    df = load_data()
    
    rearing_by_session(df)

    for session_type in session_types:
        analyze_session(df, session_type)
    
if __name__ == '__main__':
    main()