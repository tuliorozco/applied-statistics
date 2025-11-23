import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .losses import build_loss


class DictTensorDataset(Dataset):
    """Dataset que envuelve un dict de tensores de columnas + labels"""
    def __init__(self, features_dict: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.features = features_dict
        self.labels = labels
        self.keys = list(features_dict.keys())
        n = len(labels)
        for k, v in features_dict.items():
            assert len(v) == n, f"Longitud inconsistente en '{k}'"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return ({k: v[idx] for k, v in self.features.items()}, self.labels[idx])

def make_loader(features_dict, labels, batch_size, shuffle=True):
    ds = DictTensorDataset(features_dict, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class DistributedTrainer:
    """Manejador de entrenamiento (DDP o single-GPU si no hay init)"""

    def __init__(
            self, 
            model: nn.Module, 
            device: torch.device,
            world_size: int = 1, 
            learning_rate: float = 1e-4,
            training_params: Optional[Dict] = None):
        self._use_ddp = dist.is_available() and dist.is_initialized()
        self.device = device

        if self._use_ddp:
            self.model = DDP(model.to(device), device_ids=[device.index] if device.index is not None else None)
        else:
            self.model = model.to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        if training_params is not None:
            num_classes = getattr(model, "num_classes", None)
            if num_classes is None:
                raise AttributeError(
                    "El modelo no expone el atributo `num_classes`, "
                    "requerido para construir la función de pérdida."
                )
            self.criterion = build_loss(training_params, num_classes=num_classes)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.auxiliary_loss_weight = 0.01

    def compute_auxiliary_loss(self, routing_metrics: Dict) -> torch.Tensor:
        expert_usage = routing_metrics.get('expert_usage')
        if expert_usage is None:
            return torch.tensor(0.0, device=self.device)
        total_tokens = expert_usage.sum()
        if total_tokens > 0:
            expert_usage_ratio = expert_usage / total_tokens
            target_ratio = torch.ones_like(expert_usage) / len(expert_usage)
            auxiliary_loss = F.kl_div(
                expert_usage_ratio.log(), target_ratio, reduction='batchmean'
            )
            return auxiliary_loss
        return torch.tensor(0.0, device=self.device)

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits, routing_metrics = self.model(features)
            classification_loss = self.criterion(logits, labels)
            auxiliary = self.compute_auxiliary_loss(routing_metrics)
            loss = classification_loss + self.auxiliary_loss_weight * auxiliary
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    @torch.no_grad()
    def evaluate_epoch(self, dataloader) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        self.model.eval()
        probs, ys, preds = [], [], []
        total_loss = 0.0
        for features, labels in dataloader:
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = labels.to(self.device)
            logits, routing_metrics = self.model(features)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            p = F.softmax(logits, dim=-1)[:, 1].detach().cpu()
            y = labels.detach().cpu()
            pred = (p >= 0.5).long()
            probs.append(p)
            ys.append(y)
            preds.append(pred)
        import torch as T
        y_true = T.cat(ys).numpy()
        y_prob = T.cat(probs).numpy()
        y_pred = T.cat(preds).numpy()
        return {
            "loss": total_loss / max(1, len(dataloader)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

