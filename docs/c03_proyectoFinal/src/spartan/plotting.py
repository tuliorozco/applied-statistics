import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_roc_curve(model: keras.Model,
                   df_test: pd.DataFrame,
                   cont_cols: List[str],
                   bin_cols: List[str],
                   cat_cols: List[str],
                   target_col: str,
                   batch_size: int = 1024):
    """
    Calcula y grafica la curva ROC-AUC para un modelo y un conjunto de datos dados.

    Args:
        model (keras.Model): El modelo entrenado a evaluar.
        df_test (pd.DataFrame): El DataFrame con los datos de prueba o validación.
        cont_cols (List[str]): Lista de columnas continuas.
        bin_cols (List[str]): Lista de columnas binarias.
        cat_cols (List[str]): Lista de columnas categóricas.
        target_col (str): El nombre de la columna objetivo.
        batch_size (int): Tamaño del lote para la predicción.
    """
    print("Generando predicciones para la curva ROC...")

    # --- Lógica de preparación de datos (reutilizada de evaluate.py) ---
    def df_to_input_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs

    test_inputs_dict = df_to_input_dict(df_test)
    ds_test = tf.data.Dataset.from_tensor_slices(test_inputs_dict).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Obtener probabilidades y etiquetas verdaderas
    y_probs = model.predict(ds_test, verbose=0).reshape(-1)
    y_true = df_test[target_col].astype(int).values

    # --- Cálculo de la curva ROC ---
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"Área bajo la curva (AUC): {roc_auc:.4f}")

    # --- Creación de la gráfica ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()