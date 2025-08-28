import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Modelos
from sklearn.linear_model import LogisticRegression

# Métricas
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)

def train_LogisticRegression(X_train, X_test, y_train, y_test,
                            preprocessor=None, thresholds: List[float] = [0.5],
                            param_grid: Dict = None, cv_folds: int = 5,
                            max_iter: int = 1000, random_state: int = 42, **kwargs
):
    """
    Entrena Logistic Regression con múltiples thresholds.
    
    Retorna:
    - results_df: DataFrame con métricas por threshold
    - best_df: DataFrame con las mejores configuraciones por métrica
    """

    if param_grid is None:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs']
        }

    # Pipeline
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessing", preprocessor))
    steps.append(("classifier", LogisticRegression(max_iter=max_iter, random_state=random_state)))
    pipeline = Pipeline(steps)

    # Cross-validation con búsqueda de hiperparámetros
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Predicciones de probabilidades
    y_proba = grid_search.predict_proba(X_test)[:, 1]
    best_params = grid_search.best_params_

    value_C = best_params.get('classifier__C')

    if value_C:
        best_params_str = f"C: {value_C}"
    
    
    # Si no hay parámetros, usar string por defecto
    if not best_params_str:
        best_params_str = "Default"

    metrics_data = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        metrics_data.append({
            'Threshold': round(thr, 2),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'Best Params': best_params_str,
            'TN': cm[0][0] if len(cm) > 0 else 0,
            'TP': cm[1][1] if len(cm) > 1 else 0,
            'FP': cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0,
            'FN': cm[1][0] if len(cm) > 1 else 0
        })

    # DataFrame con todos los resultados
    results_df = pd.DataFrame(metrics_data)

    # Selección de las mejores configuraciones por métrica
    best_rows = []
    for metric in ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']:
        idx_max = results_df[metric].idxmax()
        best_rows.append({
            'Métrica': metric,
            'Mejor Threshold': results_df.loc[idx_max, 'Threshold'],
            'Valor': results_df.loc[idx_max, metric],
            'Parámetros': results_df.loc[idx_max, 'Best Params']
        })
    best_df = pd.DataFrame(best_rows)

    return results_df, best_df