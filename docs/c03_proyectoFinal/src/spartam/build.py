# yourpkg/build.py
from typing import Dict
from .featurizer import TabularFeaturizer
from .moe_predictor import TabularMoEPredictor

def build_featurizer(model_params: Dict, mode: str = "index") -> TabularFeaturizer:
    """
    Crea el featurizador desde model_params.
    mode: "index" (Embeddings) o "onehot" (Linear)
    """
    return TabularFeaturizer.from_params(model_params, mode=mode)

def build_model(model_params: Dict, featurizer: TabularFeaturizer) -> TabularMoEPredictor:
    """
    Construye el modelo usando un featurizador YA creado (y que luego será fit-eado).
    """
    return TabularMoEPredictor(
        featurizer=featurizer,
        d_model=model_params["d_model"],
        num_experts=model_params["num_experts"],
        k=model_params["k"],
        d_ff=model_params["d_ff"],
        num_layers=model_params.get("num_layers", 2),
        num_heads=model_params.get("num_heads", 4),
        dropout=model_params.get("dropout", 0.1),
        num_classes=2,
    )
