# moe_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .featurizer import TabularFeaturizer
from .switch_transformer import SwitchTransformerTabular

from .conformal_predictor import (
    BaseConformalPredictor,
    SplitConformalPredictor,
    LabelConditionalConformalPredictor,
)

from .metrics import MetricsCalculator
from .device_mixin import DeviceMixin 


class TabularMoEPredictor(DeviceMixin, nn.Module):
    """
    Predictor tabular con Mixture-of-Experts + (Switch)Transformer para clasificación binaria.
    Esta versión usa un featurizador unificado (featurizer.py) que reemplaza preprocessor.py y
    feature_embedder.py.

    Flujo:
      - featurizer.fit(...) fuera del modelo (una sola vez con datos de entrenamiento)
      - forward(features_dict) llama internamente al featurizer para obtener [B, d_model]
      - salida -> logits [B, 2]

    Incluye:
      - factory `from_params(...)` que acepta `model_params` (tu dict)
      - calibración y evaluación con Conformal Prediction (calibrate / evaluate_with_cp)
      - helper `predict_with_confidence` para inferencia con CP
    """

    def __init__(
        self,
        featurizer: TabularFeaturizer,
        d_model: int,
        num_experts: int,
        k: int,
        d_ff: int,
        num_layers: int = 2,     # se exponen por compatibilidad
        num_heads: int = 4,      # idem
        dropout: float = 0.1,
        num_classes: int = 2,
        alpha_cp: float = 0.1,
        conformal_predictor: Optional[BaseConformalPredictor] = None,
    ):
        
        """
            Parameters
            ----------
            featurizer : TabularFeaturizer
                Featurizador tabular ya configurado (embeddings / one-hot).
            d_model : int
                Dimensión de la representación latente.
            num_experts : int
                Número de expertos del Mixture-of-Experts.
            k : int
                Número de expertos activados por ejemplo (top-k routing).
            d_ff : int
                Dimensión interna de los expertos.
            num_layers, num_heads, dropout, num_classes
                Hiperparámetros del backbone y del clasificador.
            alpha_cp : float, default=0.1
                Nivel de significancia usado por defecto para Conformal Prediction.
            conformal_predictor : BaseConformalPredictor, optional
                Instancia concreta de predictor conformal a inyectar. Si es None
                se usa LabelConditionalConformalPredictor(alpha_cp).
            """
        # Llamada a ambos padres (DeviceMixin primero para mantener MRO correcto)
        DeviceMixin.__init__(self)
        nn.Module.__init__(self)

        
        self.featurizer = featurizer
        self.d_model = d_model
        self.num_classes = num_classes

        # Núcleo MoE/Transformer
        # Compatibilidad con posibles firmas distintas de SwitchTransformerTabular
        try:
            self.moe_layer = SwitchTransformerTabular(
                input_dim=d_model,
                output_dim=d_model,
                num_experts=num_experts,
                experts_per_token=k,
                expert_hidden_dims=[d_ff],
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        except TypeError:
            # Firma antigua / mínima
            self.moe_layer = SwitchTransformerTabular(
                input_dim=d_model,
                num_experts=num_experts,
                experts_per_token=k,
                expert_hidden_dims=[d_ff],
                output_dim=d_model,
            )

        hidden = max(d_ff, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # Conformal Prediction (inyectable)
        self.conformal_predictor: BaseConformalPredictor = (
            conformal_predictor
            if conformal_predictor is not None
            # else LabelConditionalConformalPredictor(alpha=alpha_cp)
            else SplitConformalPredictor(alpha=alpha_cp)
        )
        self.alpha_cp: float = alpha_cp

    # ------------------------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------------------------ #
    @classmethod
    def from_params(
        cls,
        model_params: Dict,
        featurizer_mode: str = "index",
        conformal_predictor: Optional[BaseConformalPredictor] = None,
    ) -> "TabularMoEPredictor":
        """
        Construye featurizer + predictor a partir de `model_params`:

        Espera las claves:
          - cont_cols, bin_cols, cat_cols, cat_vocab_sizes
          - d_model, num_heads, d_ff, num_experts, num_layers, dropout, k
          - opcionalmente: num_classes, alpha_cp
        """
        # Construir featurizer (reemplaza preprocessor+feature_embedder)
        featurizer = TabularFeaturizer.from_params(model_params, mode=featurizer_mode)

        return cls(
            featurizer=featurizer,
            d_model=model_params["d_model"],
            num_experts=model_params["num_experts"],
            k=model_params["k"],
            d_ff=model_params["d_ff"],
            num_layers=model_params.get("num_layers", 2),
            num_heads=model_params.get("num_heads", 4),
            dropout=model_params.get("dropout", 0.1),
            num_classes=model_params.get("num_classes", 2),
            alpha_cp=model_params.get("alpha_cp", 0.1),
            conformal_predictor=conformal_predictor,
        )

    # ------------------------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------------------------ #
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        features: dict[str, Tensor 1D] con columnas raw (tensores por columna).
        El featurizer aplica transformaciones + proyecciones y devuelve [B, d_model].
        """
        z = self.featurizer(features)          # [B, d_model]
        z = z.unsqueeze(1)                     # [B, 1, d_model] (token agregado)
        z, routing_metrics = self.moe_layer(z) # [B, 1, d_model], metrics
        z = z.squeeze(1)                       # [B, d_model]
        logits = self.classifier(z)            # [B, num_classes]
        return logits, routing_metrics

    # ------------------------------------------------------------------------------------ #
    # Conformal Prediction: calibración, evaluación, inferencia
    # ------------------------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate(self, cal_loader) -> None:
        """
        Realiza la fase de calibración de Conformal Prediction a partir de un
        conjunto de datos de calibración.

        Esta función recorre el `cal_loader`, obtiene los logits del modelo
        base (MoE) para todas las observaciones de calibración, y delega en
        `self.conformal_predictor` el cálculo de los scores de no-conformidad
        y de los cuantiles por clase (label-conditional / Mondrian).

        Parámetros
        ----------
        cal_loader : torch.utils.data.DataLoader
            DataLoader que suministra pares (features, labels) para el conjunto
            de calibración. Se asume que:
            - `features` es un dict de tensores de entrada por grupo/columna,
              ya en el formato esperado por el featurizador/tabular MoE.
            - `labels` es un tensor 1D con las etiquetas verdaderas
              (0/1 en el caso binario, o 0..C-1 en multiclase).

        Notas
        -----
        - Este método no retorna ningún valor; su efecto es actualizar el
          estado interno de `self.conformal_predictor` (scores y cuantiles).
        - Debe llamarse **después** de entrenar el modelo y **antes** de
          invocar `evaluate_with_cp` o cualquier otro método que utilice
          Conformal Prediction para construir conjuntos de predicción.
        """

        self.eval()
        logits_all, labels_all = [], []

        with torch.no_grad():
            for features, labels in cal_loader:
                # Enviar al dispositivo
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                # Forward del modelo base (MoE)
                logits, _ = self.forward(features)

                # Acumular en CPU para la fase de calibración
                logits_all.append(logits.cpu())
                labels_all.append(labels.cpu())

        logits_cat = torch.cat(logits_all, dim=0)
        labels_cat = torch.cat(labels_all, dim=0).long()

        # Delegar la calibración a la clase ConformalPredictor (label-conditional)
        self.conformal_predictor.calibrate(logits_cat, labels_cat)

    @torch.no_grad()
    def evaluate_with_cp(self, dataloader) -> dict[str, float]:
        """
        Evalúa el modelo utilizando Conformal Prediction sobre un DataLoader.

        Esta función realiza una evaluación conjunta del modelo base (MoE)
        y de la capa de Conformal Prediction label-conditional (Mondrian),
        calculando tanto métricas puntuales clásicas como métricas propias
        de conjuntos de predicción.

        Parámetros
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader que suministra pares (features, labels), donde:
            - features es un dict de tensores de entrada por grupo/columna.
            - labels es un tensor 1D con las etiquetas verdaderas (0/1 en el
                caso binario, o 0..C-1 en multiclase).

        Returns
        -------
        dict
            Diccionario con las métricas agregadas sobre todo el DataLoader,
            con las siguientes claves:

            - "accuracy" : float
                Exactitud clásica del clasificador puntual, usando un umbral
                de 0.5 sobre la probabilidad de la clase positiva (índice 1)
                en el caso binario.
            - "f1" : float
                F1-score del clasificador puntual.
            - "roc_auc" : float
                Área bajo la curva ROC, calculada a partir de las probabilidades
                estimadas para la clase positiva (columna 1 de softmax) en el
                caso binario.
            - "coverage" : float
                Cobertura empírica de los conjuntos de predicción conformales.
                Corresponde a la proporción de observaciones para las cuales
                la etiqueta verdadera y está incluida en el conjunto S(x).
            - "avg_set_size" : float
                Tamaño medio de los conjuntos de predicción S(x), es decir, el
                número promedio de clases incluidas por observación.
            - "coverage_per_class" : dict[str, float]
                Cobertura por clase, en formato:
                    {
                        "class_0": cobertura_clase_0,
                        "class_1": cobertura_clase_1,
                        ...
                    }
                donde cada valor es la proporción de ejemplos de esa clase
                cuya etiqueta verdadera está incluida en el conjunto S(x).
                Si alguna clase no aparece en el conjunto de evaluación, su
                cobertura se reporta como NaN.

        Notas
        -----
        - La calibración de Conformal Prediction (método `self.calibrate`)
            debe haberse ejecutado previamente sobre un conjunto de calibración
            independiente, de lo contrario se lanzará un error al invocar
            `self.conformal_predictor.predict_with_confidence`.

        - La lógica de Conformal Prediction es label-conditional (Mondrian):
            en la fase de calibración se aprende un cuantíl de no-conformidad
            por clase, y en la fase de predicción se construye el conjunto
            S(x) = { c : A(x,c) <= q_hat_c }, donde A(x,c) = 1 - p_c(x).
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        self.eval()

        all_probs = []
        all_labels = []
        all_pred_sets = []

        with torch.no_grad():
            for features, labels in dataloader:
                # Enviar a dispositivo
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                # Forward del modelo base (MoE)
                logits, _ = self.forward(features)
                probs = F.softmax(logits, dim=-1)

                # Conjuntos de predicción conformales
                pred_sets, _ = self.conformal_predictor.predict_with_confidence(logits)

                # Acumular resultados en CPU. # prob de clase positiva (índice 1)
                # all_probs.append(probs[:, 1].cpu())      
                # all_labels.append(labels.cpu())
                # all_pred_sets.append(pred_sets.cpu())

                # Probabilidad de la clase positiva (asumiendo binario, clase 1)
                all_probs.append(probs[:, 1].detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_pred_sets.append(pred_sets.detach().cpu())

        # Concatenar todo
        # y_prob = torch.cat(all_probs).numpy()          # shape (N,)
        # y_true = torch.cat(all_labels).numpy()         # shape (N,)
        # prediction_sets = torch.cat(all_pred_sets)           # shape (N, C), bool

        y_prob = torch.cat(all_probs, dim=0).numpy()
        y_true = torch.cat(all_labels, dim=0).numpy()
        prediction_sets = torch.cat(all_pred_sets, dim=0)

        metrics = MetricsCalculator.calculate_all_metrics(
            y_true=y_true,
            y_prob=y_prob,
            prediction_sets=prediction_sets,
        )
        return metrics


    @torch.no_grad()
    def predict_with_confidence(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inferencia con CP para un batch de features.
        Retorna predicciones puntuales, probabilidades, sets conformales y métricas de ruteo.
        """
        self.eval()
        logits, routing_metrics = self.forward(features)
        pred_sets, confidence_scores = self.conformal_predictor.predict_with_confidence(logits)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return {
            "predictions": preds,
            "probabilities": probs,
            "prediction_sets": pred_sets,
            "confidence_scores": confidence_scores,
            "routing_metrics": routing_metrics,
        }