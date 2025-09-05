import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Tuple, Optional, Any

# ------ Importaciones para SMOTE y SMOTE-Tomek ---------
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from .architecture import build_switch_transformer_tabular
from .evaluate  import evaluate
from .losses  import binary_focal_loss
from .metrics  import F1ScoreMetric
from .train import train

# ==============================================================================
# FUNCIÓN DE VALIDACIÓN CRUZADA (VERSIÓN 2.0 - CON CONTROL DE REMUESTREO)
# ==============================================================================
def run_cross_validation(
    df_full_train: pd.DataFrame,
    target_col: str,
    strategy: tf.distribute.Strategy,
    model_build_params: Dict,
    training_params: Dict,
    n_splits: int = 5,
    apply_smote: bool = True,
    smote_sampler: str = 'SMOTETomek',
    smote_strategy: Any = 'auto',
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Ejecuta una validación cruzada estratificada con control sobre el remuestreo.

    Args:
        ... (parámetros anteriores) ...
        apply_smote (bool): Si es True, aplica el remuestreo. Por defecto es True.
        smote_sampler (str): El tipo de remuestreador a usar ('SMOTE' o 'SMOTETomek').
        smote_strategy (Any): La estrategia para el remuestreo ('auto', float, o dict).
    
    Returns:
        Un DataFrame de Pandas con los resultados agregados de todos los folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df_full_train.drop(columns=[target_col])
    y = df_full_train[target_col]
    all_fold_results = []

    best_model = None
    best_f1_score = -np.inf


    print(f"Iniciando validación cruzada con {n_splits} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print("-" * 50)
        print(f"--- FOLD {fold + 1}/{n_splits} ---")
        print("-" * 50)

        train_fold_df = df_full_train.iloc[train_idx]
        val_fold_df = df_full_train.iloc[val_idx]
        
        # Asignamos el DF de entrenamiento a una nueva variable
        train_fold_to_use = train_fold_df

        # --- NUEVO: Bloque de remuestreo condicional ---
        if apply_smote:
            print(f"Fold {fold+1}: Aplicando {smote_sampler} con estrategia: {smote_strategy}...")
            
            if smote_sampler == 'SMOTETomek':
                sampler = SMOTETomek(sampling_strategy=smote_strategy, random_state=42, n_jobs=-1)
            elif smote_sampler == 'SMOTE':
                sampler = SMOTE(sampling_strategy=smote_strategy, random_state=42, n_jobs=-1)
            else:
                raise ValueError("El parámetro smote_sampler debe ser 'SMOTE' o 'SMOTETomek'")

            X_train_fold = train_fold_df.drop(columns=[target_col])
            y_train_fold = train_fold_df[target_col]
            X_res, y_res = sampler.fit_resample(X_train_fold, y_train_fold)
            
            # El DF que se usará para entrenar es el remuestreado
            train_fold_to_use = pd.concat([X_res, y_res], axis=1)
            print(f"Remuestreo completado. Nuevo tamaño: {len(train_fold_to_use)}")
        else:
            print(f"Fold {fold+1}: No se aplicará remuestreo.")

        # Limpieza de grafo/recursos entre folds para evitar fugas y fragmentación
        tf.keras.backend.clear_session()
        import gc; gc.collect()
        print(f"Fold {fold+1}: Construyendo y compilando un nuevo modelo...")
        

        with strategy.scope():
            model = build_switch_transformer_tabular(**model_build_params)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=training_params['lr']),
                loss=binary_focal_loss(),
                metrics=[
                    F1ScoreMetric(name="f1_score"), 
                    keras.metrics.AUC(name="auc"),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                    keras.metrics.BinaryAccuracy(name="accuracy")
                    ],
                jit_compile=False
            )
        
        # Se usa la variable `train_fold_to_use` que contiene los datos correctos
        model, history = train(
            model=model,
            df_train=train_fold_to_use,
            df_val=val_fold_df,
            batch_size=training_params['batch_size'],
            epochs=training_params['epochs'],
            **training_params['col_groups']
        )


        print(f"Fold {fold+1}: Evaluando el modelo...")
        results_df, _ = evaluate(
            model=model,
            df_test=val_fold_df,
            thresholds=thresholds,
            **training_params['col_groups']
        )
        results_df['fold'] = fold + 1
        all_fold_results.append(results_df)

        # --- CAMBIO: Lógica para identificar y guardar el mejor modelo ---
        fold_max_f1 = results_df['F1-Score'].max()
        # print(f"Fold {fold+1} F1-Score: {current_f1:.4f}")
        
        if fold_max_f1 > best_f1_score:
            best_f1_score = fold_max_f1
            best_model = model

    # Consolidar y promediar los resultados (sin cambios)
    final_results_df = pd.concat(all_fold_results, ignore_index=True)
    summary = final_results_df.drop(columns=['fold', 'Threshold']).mean()
    summary_std = final_results_df.drop(columns=['fold', 'Threshold']).std()
    summary_df = pd.DataFrame({'Mean': summary, 'Std Dev': summary_std})

    print("\n" + "="*50)
    print("Resumen de la Validación Cruzada")
    print("="*50)
    print(summary_df)

    return best_model, final_results_df