from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


class BaseConformalPredictor(ABC):
    """
    Clase base abstracta para esquemas de Conformal Prediction.

    Define la interfaz mínima que deben implementar todos los predictores
    conformales de la librería, independientemente de si están orientados
    a clasificación o regresión.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Nivel de significancia (1 - cobertura objetivo). Por ejemplo,
            alpha = 0.1 corresponde a una cobertura objetivo del 90 %.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha debe estar en (0, 1); se recibió {alpha}")
        self.alpha: float = float(alpha)
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        """
        Indica si el predictor ha sido calibrado (fase de calibración
        ejecutada correctamente).
        """
        return self._fitted

    @abstractmethod
    def calibrate(self, *args, **kwargs) -> None:
        """
        Ajusta los cuantiles de no-conformidad a partir de un conjunto
        de calibración.

        La firma concreta de este método puede variar según el tipo de
        predictor (clasificación o regresión). En todos los casos, se
        espera que deje el objeto listo para realizar predicciones
        conformales mediante `predict_with_confidence`.
        """

    @abstractmethod
    def predict_with_confidence(self, *args, **kwargs):
        """
        Genera predicciones conformales (conjuntos, intervalos, etc.)
        para nuevas observaciones.

        La forma exacta del retorno depende de la subclase. Ejemplos:
        - Clasificación:  (prediction_sets, confidence_scores)
        - Regresión (CQR): (lower, upper)
        """
class SplitConformalPredictor(BaseConformalPredictor):
    """
    Split-Conformal global para clasificación.

    Este predictor utiliza un único cuantil global sobre los scores de
    no-conformidad, a diferencia de la variante label-conditional
    (Mondrian) que estima un cuantil por clase.

    Construcción:
    - Score de no-conformidad: A(x, y) = 1 - p_y(x).
    - Cuantil global:         q_hat = Quantile_{1 - alpha}( A(x, y) ).
    - Conjunto de predicción: S(x) = { c : 1 - p_c(x) <= q_hat }.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Nivel de significancia (1 - cobertura objetivo). Por ejemplo,
            alpha = 0.1 corresponde a una cobertura objetivo del 90 %.
        """
        super().__init__(alpha=alpha)
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None
        self.n_classes: Optional[int] = None

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Ajusta el cuantil global de no-conformidad a partir de logits y
        etiquetas del conjunto de calibración.

        La lógica es:
        - Se calcula la distribución de probabilidad por clase con softmax.
        - Para cada observación (x_i, y_i), se define el score de
          no-conformidad A(x_i, y_i) = 1 - p_{y_i}(x_i), donde p_{y_i}(x_i)
          es la probabilidad asignada por el modelo a la clase verdadera.
        - Se toma el cuantíl empírico 1 - alpha sobre todos los scores
          para obtener un único umbral q_hat.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (N, C) con los logits del modelo para el
            conjunto de calibración.
        labels : torch.Tensor
            Tensor 1D de shape (N,) con las etiquetas verdaderas en
            {0, 1, ..., C-1}.

        Raises
        ------
        ValueError
            Si las dimensiones de `logits` o `labels` no son consistentes.
        RuntimeError
            Si no se recibieron muestras de calibración.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (N, C); se recibió shape={tuple(logits.shape)}"
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"El número de logits ({logits.size(0)}) y labels "
                f"({labels.size(0)}) no coincide."
            )

        labels = labels.to(torch.long)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)  # (N, C)

        n_samples, n_classes = probs.shape
        self.n_classes = n_classes

        # Probabilidad de la clase verdadera
        idx = torch.arange(n_samples, device=labels.device)
        probs_true = probs[idx, labels]                   # (N,)
        scores = (1.0 - probs_true).cpu().numpy().astype(np.float32)  # (N,)

        if scores.size == 0:
            raise RuntimeError(
                "No se recibieron muestras de calibración para SplitConformalPredictor."
            )

        q_hat = float(
            np.quantile(
                scores,
                1.0 - self.alpha,
                interpolation="higher",  # compatibilidad NumPy
            )
        )

        self.calibration_scores = scores
        self.quantile = q_hat
        self._fitted = True

    def predict_with_confidence(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera conjuntos de predicción split-conformal (global) para
        nuevos logits.

        El mismo umbral global q_hat se aplica a todas las clases.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (B, C) con los logits del modelo.

        Returns
        -------
        prediction_sets : torch.BoolTensor
            Tensor de shape (B, C) tal que prediction_sets[b, c] es True si
            la clase c pertenece al conjunto S(x_b) = { c : 1 - p_c(x_b) <= q_hat }.
        confidence_scores : torch.Tensor
            Tensor 1D de shape (B,) con un valor de confianza global
            asociado al nivel de cobertura (1 - q_hat).
        """
        if not self.fitted or self.quantile is None:
            raise RuntimeError(
                "SplitConformalPredictor no ha sido calibrado. "
                "Llama primero a `calibrate(logits, labels)`."
            )

        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (B, C); se recibió shape={tuple(logits.shape)}"
            )

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)  # (B, C)

            if self.n_classes is not None and probs.size(1) != self.n_classes:
                raise ValueError(
                    f"Número de clases en logits ({probs.size(1)}) no coincide "
                    f"con n_classes calibrado ({self.n_classes})."
                )

            # Score de no-conformidad por clase: A(x,c) = 1 - p_c(x)
            scores = 1.0 - probs  # (B, C)

            # Umbral global q_hat (broadcast sobre (B, C))
            threshold = torch.as_tensor(
                self.quantile,
                device=logits.device,
                dtype=probs.dtype,
            )

            prediction_sets = scores <= threshold  # (B, C), bool

            # Confianza global basada en q_hat
            confidence_value = 1.0 - float(self.quantile)
            confidence_scores = torch.full(
                (logits.size(0),),
                confidence_value,
                dtype=probs.dtype,
                device=logits.device,
            )

        return prediction_sets, confidence_scores


class LabelConditionalConformalPredictor(BaseConformalPredictor):
    """
    Conformal Prediction label-conditional (Mondrian) para clasificación.

    Este predictor asume que el modelo base produce logits de dimensión
    (N, C) y que la variable respuesta toma valores discretos
    {0, 1, ..., C-1}.

    La construcción es:
    - Score de no-conformidad: A(x, y) = 1 - p_y(x).
    - Cuantil por clase:      q_hat_c = Quantile_{1 - alpha}( A(x, c) | y = c ).
    - Conjunto de predicción: S(x) = { c : A(x, c) <= q_hat_c }.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__(alpha=alpha)
        # Global + por clase
        self.calibration_scores: Optional[np.ndarray] = None          # global
        self.class_calibration_scores: Optional[List[np.ndarray]] = None
        self.class_quantiles: Optional[np.ndarray] = None             # q_hat_c
        self.quantile: Optional[float] = None                         # global q_hat
        self.n_classes: Optional[int] = None

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Ajusta los cuantiles de no-conformidad de forma label-conditional
        (Mondrian) a partir de los logits y etiquetas del conjunto de
        calibración.

        La lógica es:
        - Se calcula la distribución de probabilidad por clase con softmax.
        - Para cada observación (x_i, y_i), se define el score de
          no-conformidad A(x_i, y_i) = 1 - p_{y_i}(x_i), donde p_{y_i}(x_i)
          es la probabilidad asignada por el modelo a la clase verdadera.
        - Para cada clase c, se toma el cuantíl empírico 1 - alpha de los
          scores A(x, c) correspondientes a ejemplos con y = c, obteniendo
          un q_hat_c (cuantíl por clase).
        - También se calcula un cuantíl global sobre todos los scores,
          usado como referencia y como fallback para clases sin muestras.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (N, C) con los logits del modelo para el
            conjunto de calibración.
        labels : torch.Tensor
            Tensor 1D de shape (N,) con las etiquetas verdaderas en
            {0, 1, ..., C-1}.

        Raises
        ------
        ValueError
            Si las dimensiones de `logits` o `labels` no son consistentes.
        RuntimeError
            Si no se dispone de muestras de calibración para ninguna clase.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (N, C); se recibió shape={tuple(logits.shape)}"
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"El número de logits ({logits.size(0)}) y labels "
                f"({labels.size(0)}) no coincide."
            )

        labels = labels.to(torch.long)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)  # (N, C)

        n_samples, n_classes = probs.shape
        self.n_classes = n_classes

        class_scores_list: List[np.ndarray] = []
        class_quantiles = np.full((n_classes,), np.nan, dtype=np.float32)
        all_scores: List[float] = []

        for c in range(n_classes):
            mask = labels == c
            if mask.sum().item() == 0:
                # Sin muestras de esta clase en calibración
                class_scores_list.append(np.empty((0,), dtype=np.float32))
                continue

            probs_c = probs[mask, c]
            scores_c = (1.0 - probs_c).cpu().numpy().astype(np.float32)
            class_scores_list.append(scores_c)
            all_scores.extend(scores_c.tolist())

            q_c = np.quantile(
                scores_c,
                1.0 - self.alpha,
                interpolation="higher",  # compatibilidad con versiones previas de NumPy
            )
            class_quantiles[c] = float(q_c)

        if len(all_scores) == 0:
            raise RuntimeError(
                "No se recibieron muestras de calibración para ninguna clase."
            )

        all_scores_arr = np.asarray(all_scores, dtype=np.float32)
        global_quantile = float(
            np.quantile(
                all_scores_arr,
                1.0 - self.alpha,
                interpolation="higher",
            )
        )

        nan_mask = np.isnan(class_quantiles)
        if np.any(nan_mask):
            # Fallback para clases sin datos: usar cuantíl global
            class_quantiles[nan_mask] = global_quantile

        self.class_calibration_scores = class_scores_list
        self.class_quantiles = class_quantiles
        self.calibration_scores = all_scores_arr
        self.quantile = global_quantile
        self._fitted = True

    def predict_with_confidence(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera conjuntos de predicción label-conditional para nuevos logits.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (B, C) con los logits del modelo.

        Returns
        -------
        prediction_sets : torch.BoolTensor
            Tensor de shape (B, C) tal que prediction_sets[b, c] es True si
            la clase c pertenece al conjunto S(x_b).
        confidence_scores : torch.Tensor
            Tensor 1D de shape (B,) con un valor de confianza global
            asociado al nivel de cobertura (basado en el cuantíl global).
        """
        if not self.fitted or self.class_quantiles is None:
            raise RuntimeError(
                "LabelConditionalConformalPredictor no ha sido calibrado. "
                "Llama primero a `calibrate(logits, labels)`."
            )

        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (B, C); se recibió shape={tuple(logits.shape)}"
            )

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)  # (B, C)

            if probs.size(1) != self.class_quantiles.shape[0]:
                raise ValueError(
                    f"Número de clases en logits ({probs.size(1)}) no coincide "
                    f"con class_quantiles ({self.class_quantiles.shape[0]})."
                )

            # Score de no-conformidad por clase: A(x,c) = 1 - p_c(x)
            scores = 1.0 - probs  # (B, C)

            # Umbrales por clase (broadcast a lo largo de B)
            thresholds = torch.from_numpy(self.class_quantiles).to(
                device=logits.device,
                dtype=probs.dtype,
            )  # (C,)

            # S(x) = { c : A(x,c) <= q_hat_c }
            prediction_sets = scores <= thresholds  # (B, C), bool

            # Confianza global basada en el cuantíl global (mantiene la API)
            if self.quantile is None:
                raise RuntimeError(
                    "quantile global no está definido; ver calibrate()."
                )

            confidence_value = 1.0 - float(self.quantile)
            confidence_scores = torch.full(
                (logits.size(0),),
                confidence_value,
                dtype=probs.dtype,
                device=logits.device,
            )

        return prediction_sets, confidence_scores


class CQRConformalPredictor(BaseConformalPredictor):
    """
    Conformalized Quantile Regression (CQR) para regresión.

    Este predictor asume que el modelo base produce, para cada x, dos
    predicciones de cuantiles (lower, upper) aproximando los cuantiles
    (alpha/2, 1 - alpha/2) de Y | X, antes de la conformalización.

    Referencia: Romano, Sesia, Candes (2019).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__(alpha=alpha)
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None

    def calibrate(
        self,
        y_pred_lower: torch.Tensor,
        y_pred_upper: torch.Tensor,
        y_true: torch.Tensor,
    ) -> None:
        """
        Calibra CQR a partir de intervalos preliminares y observaciones reales.

        Parameters
        ----------
        y_pred_lower, y_pred_upper : torch.Tensor
            Tensores 1D de shape (N,) con los límites inferior y superior del
            intervalo de predicción aprendido por el modelo base.
        y_true : torch.Tensor
            Tensor 1D de shape (N,) con los valores observados de la variable
            respuesta.

        Notas
        -----
        Se sigue la construcción propuesta por Romano et al. (2019):
            s_i = max( y_pred_lower_i - y_true_i,
                       y_true_i - y_pred_upper_i )
            q_hat = cuantíl_{1 - alpha}( s_i ).
        """
        if not (
            y_pred_lower.shape == y_pred_upper.shape == y_true.shape
        ):
            raise ValueError(
                "y_pred_lower, y_pred_upper y y_true deben tener la misma shape."
            )

        y_pred_lower_np = y_pred_lower.detach().cpu().numpy().astype(np.float32)
        y_pred_upper_np = y_pred_upper.detach().cpu().numpy().astype(np.float32)
        y_true_np = y_true.detach().cpu().numpy().astype(np.float32)

        lower_error = y_pred_lower_np - y_true_np
        upper_error = y_true_np - y_pred_upper_np
        scores = np.maximum(lower_error, upper_error)  # (N,)
        scores = np.maximum(scores, 0.0)               # estabilidad numérica

        q_hat = float(
            np.quantile(scores, 1.0 - self.alpha, interpolation="higher")
        )

        self.calibration_scores = scores
        self.quantile = q_hat
        self._fitted = True

    def predict_with_confidence(
        self,
        y_pred_lower: torch.Tensor,
        y_pred_upper: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ajusta los intervalos de predicción usando el cuantíl calibrado.

        Parameters
        ----------
        y_pred_lower, y_pred_upper : torch.Tensor
            Tensores 1D con los límites inferior y superior del intervalo
            de predicción del modelo base (sin conformalizar).

        Returns
        -------
        lower : torch.Tensor
            Límite inferior ajustado: y_pred_lower - q_hat.
        upper : torch.Tensor
            Límite superior ajustado: y_pred_upper + q_hat.
        """
        if not self.fitted or self.quantile is None:
            raise RuntimeError(
                "CQRConformalPredictor no ha sido calibrado. "
                "Llama primero a `calibrate(y_pred_lower, y_pred_upper, y_true)`."
            )

        q = float(self.quantile)
        adjust = torch.as_tensor(q, device=y_pred_lower.device, dtype=y_pred_lower.dtype)

        lower = y_pred_lower - adjust
        upper = y_pred_upper + adjust
        return lower, upper


class AdaptiveConformalPredictor(LabelConditionalConformalPredictor):
    """
    Variante adaptativa de Conformal Prediction para flujos de datos
    no estacionarios (Adaptive Conformal Inference sencilla).

    Implementa una versión tipo ventana deslizante sobre los scores de
    calibración por clase. Permite recalibrar los cuantiles de forma
    online para adaptarse a drift o cambios de distribución.
    """

    def __init__(self, alpha: float = 0.1, window_size: int = 1000) -> None:
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Nivel de significancia.
        window_size : int, default=1000
            Tamaño de la ventana deslizante usada para mantener los scores
            recientes de calibración por clase.
        """
        super().__init__(alpha=alpha)
        if window_size <= 0:
            raise ValueError("window_size debe ser un entero positivo.")
        self.window_size: int = int(window_size)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Actualiza de forma online los cuantiles de no-conformidad.

        Esta función puede llamarse periódicamente sobre nuevos datos
        (x_t, y_t). Internamente mantiene una ventana de tamaño finito
        sobre los scores de calibración por clase y recalcula los
        cuantiles label-conditional.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (N, C) con logits de nuevas observaciones.
        labels : torch.Tensor
            Tensor 1D de shape (N,) con las etiquetas verdaderas.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (N, C); se recibió shape={tuple(logits.shape)}"
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"El número de logits ({logits.size(0)}) y labels "
                f"({labels.size(0)}) no coincide."
            )

        labels = labels.to(torch.long)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)  # (N, C)

        n_samples, n_classes = probs.shape
        if self.n_classes is None:
            self.n_classes = n_classes
        elif self.n_classes != n_classes:
            raise ValueError(
                f"n_classes previo ({self.n_classes}) difiere del nuevo ({n_classes})."
            )

        # Inicializar buffers si es necesario
        if self.class_calibration_scores is None:
            self.class_calibration_scores = [
                np.empty((0,), dtype=np.float32) for _ in range(n_classes)
            ]

        # Añadir scores nuevos y truncar ventana
        for c in range(n_classes):
            mask = labels == c
            if mask.sum().item() == 0:
                continue
            probs_c = probs[mask, c]
            scores_c_new = (1.0 - probs_c).cpu().numpy().astype(np.float32)

            prev = self.class_calibration_scores[c]
            concat = np.concatenate([prev, scores_c_new])
            if concat.shape[0] > self.window_size:
                concat = concat[-self.window_size :]

            self.class_calibration_scores[c] = concat

        # Recalcular cuantiles con la ventana actual
        class_quantiles = np.full((n_classes,), np.nan, dtype=np.float32)
        all_scores: List[float] = []

        for c, scores_c in enumerate(self.class_calibration_scores):
            if scores_c.size == 0:
                continue
            all_scores.extend(scores_c.tolist())
            q_c = np.quantile(
                scores_c,
                1.0 - self.alpha,
                interpolation="higher",
            )
            class_quantiles[c] = float(q_c)

        if len(all_scores) == 0:
            raise RuntimeError(
                "No hay scores disponibles para recalibrar en AdaptiveConformalPredictor."
            )

        all_scores_arr = np.asarray(all_scores, dtype=np.float32)
        global_quantile = float(
            np.quantile(
                all_scores_arr,
                1.0 - self.alpha,
                interpolation="higher",
            )
        )

        nan_mask = np.isnan(class_quantiles)
        if np.any(nan_mask):
            class_quantiles[nan_mask] = global_quantile

        self.class_quantiles = class_quantiles
        self.calibration_scores = all_scores_arr
        self.quantile = global_quantile
        self._fitted = True

