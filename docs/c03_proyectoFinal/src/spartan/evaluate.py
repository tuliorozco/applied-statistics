import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from typing import List, Dict, Tuple, Optional, Any
# ============================
# Evaluate -> metrics DataFrame
# ============================
def evaluate(model: keras.Model,
             df_test: pd.DataFrame,
             cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
             target_col: str,
             threshold: float = 0.5,
             batch_size: int = 1024,
             thresholds: Optional[List[float]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Evalúa el modelo en df_test. Si thresholds list se evalúa para cada threshold y retorna df con filas por threshold.
    Devuelve (results_df, last_conf_matrix_dict)
    """
    # --- INICIO DE LA CORRECCIÓN ---

    # 1. Replicar la misma lógica de preparación de datos que en `train_`
    def df_to_input_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs

    # Convertir el DataFrame de test al formato de diccionario correcto
    test_inputs_dict = df_to_input_dict(df_test)
    
    # Crear el dataset de TensorFlow a partir del diccionario
    # No incluimos las etiquetas aquí porque `model.predict` solo necesita las características
    ds_test = tf.data.Dataset.from_tensor_slices(test_inputs_dict).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # `model.predict` devuelve las probabilidades directamente porque la última capa ya tiene activación "sigmoid"
    probs = model.predict(ds_test, verbose=0).reshape(-1)
    
    # --- FIN DE LA CORRECCIÓN ---

    y_true = df_test[target_col].astype(int).values

    # --- PUNTOS ROC + AUC (independientes de los thresholds de clasificación) ---
    fpr, tpr, thr = roc_curve(y_true, probs)      # arrays del mismo largo
    auc_val = roc_auc_score(y_true, probs)        # escalar
    roc_dict = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc": float(auc_val)
    }

    # roc_curves = {}

    thr_list = thresholds if thresholds is not None else [threshold]
    rows = []
    last_cm = {}
    for thr in thr_list:
        y_pred = (probs >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, probs)
        # fpr, tpr, thr = roc_curve(y_true, probs)
        # roc_curves = {thr: (fpr, tpr, float(roc))}

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "Threshold": round(thr, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "Accuracy": round(acc, 3),
            "ROC-AUC": round(roc, 3),
            "TN": int(tn), "TP": int(tp), "FP": int(fp), "FN": int(fn)
        })
        last_cm = {"tn":int(tn), "tp":int(tp), "fp":int(fp), "fn":int(fn)}

    results_df = pd.DataFrame(rows)

  
    return results_df, roc_dict, last_cm
