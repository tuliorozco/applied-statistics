# yourpkg/data_utils.py
from typing import Dict, Tuple
import torch
import pandas as pd

def tensors_from_dataframe(
    df: pd.DataFrame,
    col_groups: Dict
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Convierte un DataFrame en (features_dict, labels).
    - features_dict: dict[str, torch.Tensor 1D]
    - labels: torch.Tensor long 1D
    Espera col_groups = {'cont_cols': [...], 'bin_cols': [...], 'cat_cols': [...], 'target_col': '...'}
    """
    X_dict = {}
    target_col = col_groups["target_col"]

    for group_name, cols in col_groups.items():
        if group_name == "target_col":
            continue
        for c in cols:
            # Convierte a tensor 1D (asegúrate de que las categóricas estén ya codificadas a ints si venían como strings)
            X_dict[c] = torch.tensor(df[c].values)

    y = torch.tensor(df[target_col].values).long()
    return X_dict, y