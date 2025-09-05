from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


# ------------- Diccionario con la parametrizacieon por cada algoritmo ----------------    
models_config = {
        'KNN': {
            'function': KNeighborsClassifier,
            'param_grid': {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean']
            }   
        } ,
        'LogisticRegression': {
            'function': LogisticRegression,
            'init_params': {'max_iter': 1000}, 
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'DecisionTree': {
            'function': DecisionTreeClassifier,
            'param_grid': {
                'classifier__max_depth': [3, 5, 7, 10, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        },
        'RandomForest': {
            'function': RandomForestClassifier, 
            'param_grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [3, 5, 7, 10, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        } ,
        'XGBoost': {
            'function': XGBClassifier,
            'init_params': {   # parámetros por defecto especiales de XGBoost
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            'param_grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5, 6],
                'classifier__subsample': [0.8, 1.0]
            }
        },
    'SVM': {
        'function': SVC,
        'init_params': {   # Necesario porque SVC no devuelve proba por defecto
            'probability': True
        },
        'param_grid': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    },
    'MLP': {
        'function': MLPClassifier,
        'init_params': {
            'max_iter': 500
        },
        'param_grid': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        }
    }
}


def run_benchmark(common_params, preprocessor, balance=False, method="smote", sampling_strategy=1.0,
    n_samples_per_class=None):
    """
    Ejecuta benchmark de modelos definidos en models_config.

    Parámetros
    ----------
    common_params : dict
        Diccionario con X_train, X_test, y_train, y_test, cv_folds, random_state.
    preprocessor : ColumnTransformer (u otro transformador de sklearn)
        Preprocesamiento definido previamente.
    balance : bool, default=False
        Si se aplica o no remuestreo balanceado.
    method : str, default="smote"
        Método de remuestreo: "smote" o "smote_tomek".
    sampling_strategy : float | int | dict, default=1.0
        - float → proporción (ej. 0.5, 1.0).
        - int → número fijo de observaciones por clase (ej. 10000).
        - dict → número exacto de observaciones por clase {clase: n_muestras}.

    Retorna
    -------
    results_df : pd.DataFrame
    best_config_df : pd.DataFrame
    roc_curves : dict
    """

    metrics_rows = []
    best_configs_list = []
    cv_results_dfs = []   
    roc_curves = {}       

    # Definir el sampler si aplica balance
    sampler = None
    sampler_steps = []
    if balance:
        y_train = common_params['y_train']

        ### INICIO DEL BLOQUE MODIFICADO ###
        
        # CASO 1: sampling_strategy es un entero (int).
        # Se activa el pipeline de sobremuestreo + submuestreo.
        if isinstance(sampling_strategy, int):
            n_samples_per_class = sampling_strategy

            # --- Validación (Paso Opcional pero Recomendado) ---
            # Verificamos que el tamaño deseado no sea mayor que la clase mayoritaria.
            majority_count = y_train.value_counts().max()
            if n_samples_per_class > majority_count:
                raise ValueError(
                    f"El tamaño de muestra especificado por clase ({n_samples_per_class}) no puede ser mayor "
                    f"que el número de muestras en la clase mayoritaria original ({majority_count})."
                )

            # Paso 1: Definir el sobremuestreador para igualar las clases (50/50).
            # Usamos 'auto' para que la clase minoritaria se iguale a la mayoritaria.
            over_sampler = None
            if method == "smote":
                over_sampler = SMOTE(sampling_strategy='auto', random_state=common_params['random_state'])
            elif method == "smote_tomek":
                over_sampler = SMOTETomek(sampling_strategy='auto', random_state=common_params['random_state'])
            else:
                raise ValueError("Método de balanceo no reconocido. Use 'smote' o 'smote_tomek'.")
            
            # Paso 2: Definir la estrategia final con el tamaño deseado por clase.
            clases = np.unique(y_train)
            final_strategy_dict = {clase: n_samples_per_class for clase in clases}

            # Paso 3: Definir el submuestreador que aplicará la estrategia final.
            under_sampler = RandomUnderSampler(sampling_strategy=final_strategy_dict, random_state=common_params['random_state'])

            # Añadimos los dos samplers como pasos individuales a nuestra lista
            sampler_steps.append(('over_sampling_step', over_sampler))
            sampler_steps.append(('under_sampling_step', under_sampler))

        # CASO 2: sampling_strategy es float, dict, o str.
        # Se mantiene el comportamiento original.
        else: # Caso para float, dict, str
            sampler = None
            if method == "smote":
                sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=common_params['random_state'])
            elif method == "smote_tomek":
                sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=common_params['random_state'])
            else:
                raise ValueError("Método de balanceo no reconocido. Use 'smote' o 'smote_tomek'.")
            
            # Añadimos el único sampler como un paso individual a nuestra lista
            sampler_steps.append(('sampler', sampler))

    for model_name, config in models_config.items():
        print(f"\nEntrenando modelo: {model_name}")

        init_params = config.get('init_params', {})

        # Usar pipeline con sampler si aplica
        if sampler:
            pipeline = ImbPipeline([
                ("preprocessing", preprocessor),
                ("sampler", sampler),
                ("classifier", config['function'](**init_params))
            ])
        else:
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("classifier", config['function'](**init_params))
            ])

        # 2. Cross-validation
        skf = StratifiedKFold(
            n_splits=common_params['cv_folds'],
            shuffle=True,
            random_state=common_params['random_state']
        )
        grid_search = GridSearchCV(
            pipeline,
            config['param_grid'],
            cv=skf,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        inicio = time.perf_counter()
        grid_search.fit(common_params['X_train'], common_params['y_train'])
        fin = time.perf_counter()
        train_time = fin - inicio

        # Guardar resultados completos de GridSearchCV
        cv_results_df = pd.DataFrame(grid_search.cv_results_)
        cv_results_df["Model"] = model_name
        cv_results_dfs.append(cv_results_df)

        # 3. Predicciones
        y_pred = grid_search.predict(common_params['X_test'])
        y_proba = grid_search.predict_proba(common_params['X_test'])[:, 1]

        # 4. Resultados principales
        cm = confusion_matrix(common_params['y_test'], y_pred)
        metrics_row = {
            'Model': model_name,
            'Precision': precision_score(common_params['y_test'], y_pred, zero_division=0),
            'Recall': recall_score(common_params['y_test'], y_pred, zero_division=0),
            'F1-Score': f1_score(common_params['y_test'], y_pred, zero_division=0),
            'Accuracy': accuracy_score(common_params['y_test'], y_pred),
            'ROC-AUC': roc_auc_score(common_params['y_test'], y_proba),
            'Best Params': {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()},
            'TN': int(cm[0, 0]),
            'TP': int(cm[1, 1]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0])
        }
        metrics_rows.append(metrics_row)

        # 5. Guardar configuraciones de mejores métricas
        for metric in ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC']:
            best_configs_list.append({
                'Modelo': model_name,
                'Métrica': metric,
                'Valor': metrics_row[metric],
                'Parámetros': metrics_row['Best Params']
            })

        # 6. Guardar curva ROC
        fpr, tpr, _ = roc_curve(common_params['y_test'], y_proba)
        roc_auc = auc(fpr, tpr)
        roc_curves[model_name] = (fpr, tpr, roc_auc)

        print(f"Entrenamiento completado. Tiempo entrenamiento {model_name}: {train_time:.4f} segundos")

    col_order = [
        'Model', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC',
        'Best Params', 'TN', 'TP', 'FP', 'FN'
    ]
    results_df = pd.DataFrame(metrics_rows)[col_order]
    best_config_df = pd.DataFrame(best_configs_list)[['Modelo', 'Métrica', 'Valor', 'Parámetros']]

    return results_df, best_config_df, roc_curves
