import os
# Establece el nivel de registro de logs de TensorFlow para ocultar los mensajes de INFO.
# 0 = Muestra todos los mensajes (default)
# 1 = Filtra los mensajes INFO
# 2 = Filtra los mensajes INFO y WARNING
# 3 = Filtra los mensajes INFO, WARNING, y ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from typing import List, Dict, Tuple, Optional, Any
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from .callbacks import EpochLogger
from .losses import binary_focal_loss
from .metrics import F1ScoreMetric


def train(model: keras.Model,
          df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          cont_cols: List[str],
          bin_cols: List[str],
          cat_cols: List[str],
          target_col: str,
          batch_size: int = 256,
          epochs: int = 30,
          lr: float = 1e-4,
          focal_loss: bool = True,
          callbacks: Optional[List[Callback]] = None,
          mixed_precision: bool = True): # CAMBIO: True por defecto para máximo rendimiento
    """
    Entrena el modelo usando un pipeline de tf.data optimizado y vectorizado.
    
    CAMBIOS CLAVE:
    - Se eliminó el `data_generator` y el parámetro `smote_config`. El remuestreo
      debe realizarse ANTES de llamar a esta función.
    - Se utiliza `tf.data.Dataset.from_tensor_slices` para una ingesta de datos
      de alto rendimiento, eliminando el cuello de botella de `iterrows`.
    - Se recomienda ejecutar la construcción y compilación del modelo dentro de un
      `strategy.scope()` para entrenamiento distribuido.
    """

    # --- CAMBIO: Mixed Precision activado por defecto ---
    # Permite un rendimiento significativamente mayor en GPUs compatibles (Tensor Cores)
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --- CAMBIO: Conversión vectorizada de DataFrame a diccionario de tensores ---
    # Este helper convierte eficientemente un DataFrame a un formato que TensorFlow puede ingerir.
    def df_to_input_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs

    # 1. Convertir los DataFrames a diccionarios de arrays NumPy (una sola vez)
    print("Convirtiendo DataFrames a tensores para el entrenamiento...")
    train_inputs_dict = df_to_input_dict(df_train)
    train_labels = df_train[target_col].values.astype("float32")
    
    val_inputs_dict = df_to_input_dict(df_val)
    val_labels = df_val[target_col].values.astype("float32")

    # 2. Crear los Datasets de TensorFlow de forma eficiente
    print("Creando datasets de TensorFlow...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs_dict, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs_dict, val_labels))

    # 3. Aplicar caching, shuffle y batching (sin cambios, ya era correcto)
    # El `cache()` ahora guardará los tensores preprocesados, siendo mucho más rápido.
    train_ds = train_ds.cache().shuffle(buffer_size=len(df_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 4. Compilar y entrenar el modelo (sin cambios en la lógica)
    # NOTA: La compilación debe ocurrir dentro de `strategy.scope()` en tu script principal.
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = binary_focal_loss() if focal_loss else keras.losses.BinaryCrossentropy(from_logits=False)
    f1_metric = F1ScoreMetric()

    # Se asume que el modelo ya está compilado antes de llamar a train().
    # Si no, se compila aquí.
    if model.optimizer is None:
        print("Compilando el modelo...")
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                f1_metric,
                keras.metrics.BinaryAccuracy(name="accuracy")
            ]#,
            #jit_compile=True
        )

    # --- callbacks ---
    ckpt_path = "model.weights.best.weights.h5"

    # Keras añade el prefijo "val_" al nombre de la métrica.
    MONITOR_METRIC_NAME = "f1_score"


    # 1. Callback para guardar el mejor modelo basado en F1-score
    model_checkpoint = EpochLogger(
        filepath=ckpt_path,
        monitor=MONITOR_METRIC_NAME,
        mode="max"
    )

    # 2. Callback para detener el entrenamiento si el F1-score no mejora
    early_stopping = EarlyStopping(
        monitor=MONITOR_METRIC_NAME,        # Métrica a monitorear
        patience=7,                         # Épocas a esperar antes de detener si no hay mejora
        mode="max",                         # 'max' porque un F1-score más alto es mejor
        restore_best_weights=True,          # Vuelve a los pesos del mejor F1-score al finalizar
        verbose=1
    )

    # 3. Callback para reducir la tasa de aprendizaje si el F1-score se estanca
    reduce_lr = ReduceLROnPlateau(
        monitor=MONITOR_METRIC_NAME,        # Métrica a monitorear
        factor=0.5,                         # Factor por el cual se reduce la tasa de aprendizaje (new_lr = lr * factor)
        patience=3,                         # Épocas a esperar antes de reducir la tasa de aprendizaje
        mode="max",                         # 'max' porque un F1-score más alto es mejor
        min_lr=1e-6,                        # Tasa de aprendizaje mínima
        verbose=1
    )

    # Agrupar todos los callbacks en una sola lista
    callbacks_list = [
        model_checkpoint,
        early_stopping,
        reduce_lr
    ]
    if callbacks is None:
         callbacks = callbacks_list
        

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=2
    )

    # --- cleanup datasets/tensores para liberar memoria entre folds ---
    try:
        del train_ds, val_ds, train_inputs_dict, val_inputs_dict, train_labels, val_labels
    except Exception:
        pass
    import gc as _gc; _gc.collect()

    return model, history
