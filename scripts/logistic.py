import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils import fig_path, group_areas_and_directions, session_types

def logistic(_df, session_type):
    df = _df.copy()
    data, area_columns, direction_columns = group_areas_and_directions(df, session_type)
    
    X = data[['group'] + area_columns + direction_columns].copy()
    X['group'] = (X['group'] == 'muscimol').astype(int)
    y = data['rearing']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nFeature Coefficients:")
    feature_names = X.columns
    coefficients = model.coef_[0]
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    coefficients_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    coefficients_df = coefficients_df.sort_values('coefficient', key=abs, ascending=False)
    sns.barplot(data=coefficients_df, x='coefficient', y='feature')
    plt.title('Coeficientes')
    plt.xlabel('Valor do coeficiente')
    
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    
    plt.subplot(2, 2, 3)
    plt.hist(y_prob[y_test == 0], alpha=0.7, label='No Rearing', bins=20)
    plt.hist(y_prob[y_test == 1], alpha=0.7, label='Rearing', bins=20)
    plt.xlabel('Probabilidade prevista')
    plt.ylabel('Frequência')
    plt.title('Probabilidades previstas')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    rearing_by_group = data.groupby('group')['rearing'].mean()
    rearing_by_group.plot(kind='bar')
    plt.title('Taxa de rearing por grupo')
    plt.ylabel('Taxa de rearing')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{fig_path}/logistic_{session_type}.png', dpi=300, bbox_inches='tight')
    
