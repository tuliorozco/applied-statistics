# losses.py
"""
Módulo de funciones de pérdida para entrenamiento de modelos.

Incluye una implementación de Focal Loss pensada para problemas de
clasificación con fuerte desbalanceo, y una función factory para
construir la loss a partir de `training_params`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementación de Focal Loss para clasificación.

    Esta versión soporta tanto problemas binarios como multi-clase.
    Para el caso binario se asume típicamente que el modelo produce
    logits de shape (N, 2), donde la clase positiva es el índice 1.

    Referencia:
    Lin et al. (2017). Focal Loss for Dense Object Detection.

    L(y, p) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[Union[float, Sequence[float]]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        alpha : float | Sequence[float] | None, default=None
            Factor de ponderación para cada clase:
            - None  -> sin ponderación específica.
            - float -> binario: alpha para clase 1 (positiva), 1-alpha para clase 0.
            - secuencia -> vector de pesos por clase (len = num_classes).
        gamma : float, default=2.0
            Parámetro de enfoque. Valores altos enfatizan ejemplos difíciles.
        reduction : {"none", "mean", "sum"}, default="mean"
            Tipo de reducción sobre la pérdida por ejemplo.
        """
        super().__init__()

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction debe ser 'none', 'mean' o 'sum'; se recibió {reduction!r}")

        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # Caso binario: [peso_negativo, peso_positivo]
            self.alpha = torch.tensor([1.0 - float(alpha), float(alpha)], dtype=torch.float32)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha_tensor.ndim != 1:
                raise ValueError("alpha debe ser un escalar o un vector 1D de pesos por clase.")
            self.alpha = alpha_tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la Focal Loss para un batch.

        Parameters
        ----------
        logits : torch.Tensor
            Tensor de shape (N, C) con los logits del modelo.
        targets : torch.Tensor
            Tensor 1D de shape (N,) con las etiquetas verdaderas.

        Returns
        -------
        torch.Tensor
            Pérdida agregada según `reduction` o vector por ejemplo si
            `reduction="none"`.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits debe ser 2D (N, C); se recibió shape={tuple(logits.shape)}"
            )

        if targets.ndim != 1 or targets.size(0) != logits.size(0):
            raise ValueError(
                "targets debe ser 1D de longitud N y consistente con logits."
            )

        num_classes = logits.size(1)
        targets = targets.to(torch.long)

        log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
        probs = log_probs.exp()                    # (N, C)

        idx = torch.arange(logits.size(0), device=logits.device)
        log_p_t = log_probs[idx, targets]   # (N,)
        p_t = probs[idx, targets]           # (N,)

        # alpha_t
        if self.alpha is None:
            alpha_t = torch.ones_like(p_t)
        else:
            alpha = self.alpha.to(logits.device)
            if alpha.numel() == 2 and num_classes == 2:
                alpha_t = alpha[targets]
            else:
                if alpha.numel() != num_classes:
                    raise ValueError(
                        f"alpha tiene longitud {alpha.numel()} pero num_classes={num_classes}."
                    )
                alpha_t = alpha[targets]

        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = -alpha_t * focal_factor * log_p_t  # (N,)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


def build_loss(
    training_params: Dict,
    num_classes: int,
) -> nn.Module:
    """
    Construye la función de pérdida a partir de `training_params`.

    Permite seleccionar entre Cross-Entropy estándar y Focal Loss,
    manteniendo la abstracción `nn.Module` para el resto del código.

    Parameters
    ----------
    training_params : Dict
        Parámetros de entrenamiento. Claves relevantes:
        - "loss_type": str, opcional:
            * "cross_entropy" (por defecto)
            * "focal"
        - "focal_gamma": float, opcional, default=2.0
        - "focal_alpha": float | Sequence[float] | None, opcional.
        - "loss_reduction": str, opcional. "mean" (default), "sum", "none".
    num_classes : int
        Número de clases de la tarea.

    Returns
    -------
    nn.Module
        Instancia de pérdida lista para usar.
    """
    loss_type = training_params.get("loss_type", "cross_entropy").lower()
    reduction = training_params.get("loss_reduction", "mean")

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction)

    if loss_type == "focal":
        gamma = float(training_params.get("focal_gamma", 2.0))
        alpha = training_params.get("focal_alpha", None)

        if isinstance(alpha, (float, int)) and num_classes != 2:
            raise ValueError(
                "Se proporcionó focal_alpha como escalar, pero num_classes != 2. "
                "Para multi-clase use una secuencia de pesos por clase."
            )

        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    raise ValueError(
        f"loss_type desconocido: {loss_type!r}. "
        "Valores soportados: 'cross_entropy', 'focal'."
    )
