# yourpkg/cv.py
from typing import Dict, Tuple, List
import torch
from sklearn.model_selection import StratifiedKFold

from .build import build_featurizer, build_model
from .featurizer_ops import fit_featurizer_from_df, clone_featurizer
from .data_utils import tensors_from_dataframe
from .distributed_trainer import DistributedTrainer, make_loader

def cross_validate_get_best(
    df,                      # pandas.DataFrame
    model_params: Dict,
    training_params: Dict,
    k_folds: int = 5,
    device: torch.device | None = None,
    featurizer_mode: str = "index",
    featurizer_fit_scope: str = "global",
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    CV estratificada K-fold.
    Retorna:
      - best_state (state_dict del modelo ganador)
      - best_summary (p.ej. {'cv_best_roc_auc': ...})
      - fold_history (lista de métricas por fold)

    featurizer_fit_scope:
      - "global": fit del featurizador con TODO df (establece mappings/escala).
      - "in_fold": fit del featurizador SOLO con X_train de cada fold (más estricto contra leakage).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensores crudos (aún sin transformar por featurizer)
    X_dict, y = tensors_from_dataframe(df, training_params["col_groups"])

    # Featurizador (fit global opcional)
    base_featurizer = build_featurizer(model_params, mode=featurizer_mode)
    if featurizer_fit_scope == "global":
        # Ajusta con todo el df (mappings/escala estables)
        base_featurizer = fit_featurizer_from_df(base_featurizer, df, training_params["col_groups"])

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_metric = -1.0
    best_state = None
    fold_history: List[Dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(torch.arange(len(y)), y)):
        # Split dict de tensores
        X_tr = {k: v[tr_idx] for k, v in X_dict.items()}
        y_tr = y[tr_idx]
        X_va = {k: v[va_idx] for k, v in X_dict.items()}
        y_va = y[va_idx]

        # Build featurizer/model por fold
        if featurizer_fit_scope == "global":
            featurizer = clone_featurizer(base_featurizer)
        else:
            # fit dentro del fold solo con train
            featurizer = build_featurizer(model_params, mode=featurizer_mode)
            featurizer.fit(X_tr)

        model = build_model(model_params, featurizer).to(device)
        trainer = DistributedTrainer(
            model,
            device,
            learning_rate=training_params["lr"],
            training_params=training_params,
        )

        train_loader = make_loader(X_tr, y_tr, batch_size=training_params["batch_size"], shuffle=True)
        val_loader   = make_loader(X_va, y_va, batch_size=training_params["batch_size"], shuffle=False)

        # Entrenamiento
        for _ in range(training_params["epochs"]):
            trainer.train_epoch(train_loader)

        # Validación (ranking por ROC-AUC)
        metrics_val = trainer.evaluate_epoch(val_loader)
        fold_history.append({"fold": fold, "val_metrics": metrics_val})

        if metrics_val["roc_auc"] > best_metric:
            best_metric = metrics_val["roc_auc"]
            best_state = model.state_dict()

    return best_state, {"cv_best_roc_auc": best_metric}, fold_history
