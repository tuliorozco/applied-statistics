from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def config(random_state: int = 42):
    
    models_config = {
            'KNN': {
                'function': KNeighborsClassifier,
                'init_params': {'random_state': random_state}, 
                'param_grid': {
                'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean']
                }   
            } ,
            'LogisticRegression': {
                'function': LogisticRegression,
                'init_params': {'max_iter': 1000, 'random_state': random_state}, 
                'param_grid': {
                    'classifier__C': [0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            },
            'DecisionTree': {
                'function': DecisionTreeClassifier,
                'init_params': {'random_state': random_state}, 
                'param_grid': {
                    'classifier__max_depth': [3, 5, 7, 10, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'RandomForest': {
                'function': RandomForestClassifier, 
                'init_params': {'random_state': random_state}, 
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
                    'use_label_encoder': False,
                    'random_state': random_state
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
                'probability': True,
                'random_state': random_state
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
                'max_iter': 500,
                'random_state': random_state
            },
            'param_grid': {
                'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__solver': ['adam', 'sgd'],
                'classifier__alpha': [0.0001, 0.001, 0.01]
            }
        }
    }
    return models_config