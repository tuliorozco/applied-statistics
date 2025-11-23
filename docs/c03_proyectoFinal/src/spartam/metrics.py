# metrics.py
"""
Módulo para cálculo y gestión de métricas de evaluación.
Centraliza todas las métricas utilizadas en evaluación clásica y Conformal Prediction.
"""

from typing import Dict, Optional, Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MetricsCalculator:
    """
    Calculadora centralizada de métricas para evaluación de modelos.
    Soporta métricas clásicas y de Conformal Prediction.
    """

    @staticmethod
    def calculate_classical_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calcula métricas clásicas de clasificación.

        Parameters
        ----------
        y_true : np.ndarray
            Etiquetas verdaderas de shape (N,).
        y_prob : np.ndarray
            Probabilidades de la clase positiva de shape (N,).
        y_pred : np.ndarray, optional
            Predicciones puntuales. Si es None, se obtienen aplicando
            un umbral de 0.5 sobre `y_prob`.

        Returns
        -------
        Dict[str, float]
            Diccionario con métricas clásicas:
            - "accuracy"
            - "f1"
            - "roc_auc"
        """
        if y_pred is None:
            y_pred = (y_prob >= 0.5).astype(int)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    @staticmethod
    def calculate_conformal_metrics(
        prediction_sets: torch.Tensor,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calcula métricas específicas de Conformal Prediction.

        Parameters
        ----------
        prediction_sets : torch.Tensor
            Tensor booleano de shape (N, C) donde prediction_sets[i, c] indica
            si la clase c pertenece al conjunto de predicción S(x_i).
        y_true : np.ndarray
            Etiquetas verdaderas de shape (N,).

        Returns
        -------
        Dict[str, Any]
            Diccionario con métricas de CP:
            - "coverage" : float
            - "avg_set_size" : float
            - "coverage_per_class" : Dict[str, float]
        """
        if prediction_sets.ndim != 2:
            raise ValueError(
                f"prediction_sets debe ser 2D (N, C); se recibió "
                f"shape={tuple(prediction_sets.shape)}"
            )

        y_true_t = torch.from_numpy(y_true).long()
        idx = torch.arange(len(y_true_t))

        # Cobertura global: y_true ∈ S(x)
        coverage = float(prediction_sets[idx, y_true_t].float().mean().item())

        # Tamaño promedio del conjunto S(x)
        avg_set_size = float(prediction_sets.sum(dim=1).float().mean().item())

        # Cobertura por clase
        num_classes = prediction_sets.size(1)
        coverage_per_class: Dict[str, float] = {}

        for c in range(num_classes):
            mask_c = y_true_t == c
            if mask_c.any():
                cov_c = float(prediction_sets[mask_c, c].float().mean().item())
            else:
                cov_c = float("nan")
            coverage_per_class[f"class_{c}"] = cov_c

        return {
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "coverage_per_class": coverage_per_class,
        }

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        prediction_sets: torch.Tensor,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calcula todas las métricas (clásicas + CP) en una sola llamada.

        Parameters
        ----------
        y_true : np.ndarray
            Etiquetas verdaderas de shape (N,).
        y_prob : np.ndarray
            Probabilidades de la clase positiva de shape (N,).
        prediction_sets : torch.Tensor
            Tensor booleano de shape (N, C) con los conjuntos de predicción
            conformales.
        y_pred : np.ndarray, optional
            Predicciones puntuales. Si es None, se obtienen a partir de `y_prob`.

        Returns
        -------
        Dict[str, Any]
            Diccionario consolidado con métricas clásicas y de CP.
            Contiene al menos:
            - "accuracy"
            - "f1"
            - "roc_auc"
            - "coverage"
            - "avg_set_size"
            - "coverage_per_class"
        """
        classical_metrics = MetricsCalculator.calculate_classical_metrics(
            y_true, y_prob, y_pred
        )
        conformal_metrics = MetricsCalculator.calculate_conformal_metrics(
            prediction_sets, y_true
        )

        # Unimos los dos diccionarios en uno solo
        return {**classical_metrics, **conformal_metrics}
