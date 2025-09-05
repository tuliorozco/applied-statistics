import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional, Any

# ============================
# Calibración Conforme Condicional a la Clase
# ============================
def calibrate_conformal_conditional(model: keras.Model,
                                    df_cal: pd.DataFrame,
                                    cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
                                    target_col: str,
                                    alpha: float = 0.1,
                                    batch_size: int = 1024) -> Dict[str, float]:
    """
    Realiza calibración conforme condicional a la clase.
    Devuelve un diccionario con qhat para la clase 0 y la clase 1.
    - qhat_1 se usa para definir el umbral de la clase positiva: p >= 1 - qhat_1
    - qhat_0 se usa para definir el umbral de la clase negativa: p <= qhat_0
    """
    # Helper para convertir df a dataset para predicción
    def df_to_pred_dataset(df):
        inputs = {}
        if cont_cols: inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols: inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols: inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)

    ds_cal = df_to_pred_dataset(df_cal)
    probs = model.predict(ds_cal, verbose=0).reshape(-1)
    y = df_cal[target_col].astype(int).values

    # Calibración para la clase 1 (positivos)
    y1_indices = np.where(y == 1)[0]
    scores_1 = 1.0 - probs[y1_indices]
    n1 = len(scores_1)
    rank_1 = int(np.ceil((n1 + 1) * (1 - alpha)))
    qhat_1 = float(np.sort(scores_1)[min(rank_1 - 1, n1 - 1)])

    # Calibración para la clase 0 (negativos)
    y0_indices = np.where(y == 0)[0]
    scores_0 = probs[y0_indices] # Score es p, ya que 1-(1-p) = p
    n0 = len(scores_0)
    rank_0 = int(np.ceil((n0 + 1) * (1 - alpha)))
    qhat_0 = float(np.sort(scores_0)[min(rank_0 - 1, n0 - 1)])
    
    return {"qhat_0": qhat_0, "qhat_1": qhat_1}

# ============================
# CAMBIO: Nueva función para evaluar conjuntos de predicción conformes
# ============================
def evaluate_conformal_sets(model: keras.Model,
                            df_test: pd.DataFrame,
                            cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
                            target_col: str,
                            qhats: Dict[str, float],
                            batch_size: int = 1024) -> Dict[str, float]:
    """
    Evalúa el rendimiento de los conjuntos de predicción conformes.
    Calcula la cobertura empírica y el tamaño promedio del conjunto.
    """
    qhat_0, qhat_1 = qhats["qhat_0"], qhats["qhat_1"]
    
    def df_to_pred_dataset(df): # Helper anidado
        inputs = {}
        if cont_cols: inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols: inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols: inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)

    ds_test = df_to_pred_dataset(df_test)
    probs = model.predict(ds_test, verbose=0).reshape(-1)
    y_true = df_test[target_col].astype(int).values

    # Construir conjuntos de predicción
    prediction_sets = []
    for p in probs:
        s = []
        if p <= qhat_0:
            s.append(0)
        if p >= (1 - qhat_1):
            s.append(1)
        prediction_sets.append(s)

    # Calcular métricas
    correct_predictions = 0
    total_set_size = 0
    for i, pred_set in enumerate(prediction_sets):
        if y_true[i] in pred_set:
            correct_predictions += 1
        total_set_size += len(pred_set)

    coverage = correct_predictions / len(y_true)
    avg_set_size = total_set_size / len(y_true)
    
    return {"coverage": coverage, "average_set_size": avg_set_size}