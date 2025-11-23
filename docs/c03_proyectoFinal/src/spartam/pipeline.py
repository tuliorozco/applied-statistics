# yourpkg/pipeline.py
import torch
import pandas as pd
from typing import Dict

from .cv import cross_validate_get_best
from .evaluate import calibrate_and_evaluate

def run_cv_calibration_evaluation(
    df_trainval: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params: Dict,
    training_params: Dict,
    k_folds: int = 5,
    featurizer_mode: str = "index",          # "index" (Embeddings) o "onehot"
    featurizer_fit_scope: str = "global",    # "global" o "in_fold"
):
    """
    Orquesta:
      1) CV K-fold y selección del mejor estado (por ROC-AUC)
      2) Calibración CP en cal_df
      3) Evaluación CP en cal_df y test_df
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_state, cv_summary, history = cross_validate_get_best(
        df=df_trainval,
        model_params=model_params,
        training_params=training_params,
        k_folds=k_folds,
        device=device,
        featurizer_mode=featurizer_mode,
        featurizer_fit_scope=featurizer_fit_scope,
    )

    best_model, eval_report = calibrate_and_evaluate(
        best_state=best_state,
        model_params=model_params,
        training_params=training_params,
        cal_df=cal_df,
        test_df=test_df,
        device=device,
        featurizer_mode=featurizer_mode,
        featurizer_fit_df=cal_df,   # fit del featurizador coherente con calibración
    )

    return best_model, cv_summary, history, eval_report
