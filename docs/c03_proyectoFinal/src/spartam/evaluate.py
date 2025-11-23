# evaluate.py
from typing import Dict, Tuple
import torch
import pandas as pd

from .build import build_featurizer, build_model
from .featurizer_ops import fit_featurizer_from_df
from .data_utils import tensors_from_dataframe
from .distributed_trainer import make_loader

def calibrate_and_evaluate(
    best_state: Dict,
    model_params: Dict,
    training_params: Dict,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device | None = None,
    featurizer_mode: str = "index",
    featurizer_fit_df: pd.DataFrame | None = None,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Carga best_state en un modelo nuevo, calibra con cal_df y evalúa con CP en cal/test.
    - featurizer_fit_df: DataFrame para fit del featurizador (por defecto: cal_df)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if featurizer_fit_df is None:
        featurizer_fit_df = cal_df

    # Featurizador + fit (mappings/escala coherentes con calibración)
    featurizer = build_featurizer(model_params, mode=featurizer_mode)
    featurizer = fit_featurizer_from_df(featurizer, featurizer_fit_df, training_params["col_groups"])

    # Modelo + cargar estado ganador
    model = build_model(model_params, featurizer).to(device)
    model.load_state_dict(best_state)

    # Loaders
    X_cal, y_cal = tensors_from_dataframe(cal_df, training_params["col_groups"])
    X_test, y_test = tensors_from_dataframe(test_df, training_params["col_groups"])
    cal_loader  = make_loader(X_cal, y_cal, batch_size=training_params["batch_size"], shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=training_params["batch_size"], shuffle=False)

    # Calibración CP
    model.calibrate(cal_loader)

    # Evaluación CP
    cal_metrics  = model.evaluate_with_cp(cal_loader)
    test_metrics = model.evaluate_with_cp(test_loader)

    return model, {"cal_metrics": cal_metrics, "test_metrics": test_metrics}
