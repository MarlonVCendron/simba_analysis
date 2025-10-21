import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os

from utils import fig_path, group_areas_and_directions, session_types

def latent_class(_df, session_type):
    df = _df.copy()
    data, area_columns, direction_columns = group_areas_and_directions(df, session_type)
    
    data['group'] = (data['group'] == 'muscimol').astype(int)
    features = data[['group'] + area_columns + direction_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = GaussianMixture(n_components=3)
    clusters = model.fit_predict(features_scaled)
    
    data['cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    cluster_means = data.groupby('cluster')[area_columns + direction_columns].mean()
    sns.heatmap(cluster_means.T, annot=True, cmap='viridis')
    plt.title('Cluster Means')
    
    plt.subplot(1, 2, 2)
    cluster_rearing = data.groupby('cluster')['rearing'].mean()
    cluster_rearing.plot(kind='bar')
    plt.title('Rearing Rate by Cluster')
    plt.ylabel('Rearing Rate')
    
    plt.tight_layout()
    os.makedirs(os.path.join(fig_path, 'latent'), exist_ok=True)
    plt.savefig(os.path.join(fig_path, 'latent', f'latent_{session_type}.png'), dpi=300, bbox_inches='tight')
    
    print(f"Cluster sizes: {np.bincount(clusters)}")
    print(f"Rearing rates by cluster: {cluster_rearing.to_dict()}")
