# yourpkg/featurizer_ops.py
from typing import Dict
import copy
import torch
import pandas as pd

from .featurizer import TabularFeaturizer
from .data_utils import tensors_from_dataframe

def fit_featurizer_from_df(
    featurizer: TabularFeaturizer,
    df: pd.DataFrame,
    col_groups: Dict
) -> TabularFeaturizer:
    """
    Ajusta mappings y escaladores del featurizador a partir de un DataFrame.
    """
    X_dict, _ = tensors_from_dataframe(df, col_groups)
    featurizer.fit(X_dict)
    return featurizer

def clone_featurizer(fitted_featurizer: TabularFeaturizer) -> TabularFeaturizer:
    """
    Clona un featurizador YA fit-eado (mappings, escaladores y capas).
    Útil para usar la misma estructura en varios folds sin re-fit global.
    """
    return copy.deepcopy(fitted_featurizer)
