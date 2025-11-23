# featurizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class TabularFeaturizer(nn.Module):
    """
    Featurizador tabular unificado (preprocesamiento + embeddings).

    Flujo:
      - from_params(...) -> instancia
      - fit(X_dict)      -> aprende mappings (categorías) y z-score (cont/disc)
      - forward(X_dict)  -> transforma y proyecta a [B, d_model]

    Modos para categóricas:
      - mode="onehot":   categóricas -> one-hot [B, C] -> Linear(C, d_model)
      - mode="index":    categóricas -> índices [B]    -> Embedding(C+1, d_model), 0 = UNK
    """

    def __init__(
        self,
        cat_cols: List[str],
        cont_cols: List[str],
        bin_cols: Optional[List[str]] = None,
        disc_cols: Optional[List[str]] = None,
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
        d_model: int = 128,
        mode: str = "onehot",  # "onehot" | "index"
    ):
        super().__init__()
        assert mode in ("onehot", "index"), "mode debe ser 'onehot' o 'index'"

        self.cat_cols = cat_cols or []
        self.cont_cols = cont_cols or []
        self.bin_cols = bin_cols or []
        self.disc_cols = disc_cols or []
        self.cat_vocab_sizes = cat_vocab_sizes or {}
        self.d_model = d_model
        self.mode = mode

        # --- Estado aprendido en fit() ---
        # mapeo categórico valor->índice (índice empieza en 1 si mode=='index', 0 reservado a UNK)
        self.category_mappings: Dict[str, Dict[int, int]] = {}
        # dimensiones por feature (C para categóricas, 1 para cont/disc/bin)
        self.feature_dims: Dict[str, int] = {}
        # escaladores (mean/std) para cont/disc
        self.scalers: Dict[str, Dict[str, float]] = {}

        # --- Capas creadas después de fit() ---
        self.layers = nn.ModuleDict()
        self._built = False

    @classmethod
    def from_params(cls, model_params: Dict, mode: str = "onehot") -> "TabularFeaturizer":
        """Construye la clase a partir de model_params."""
        return cls(
            cat_cols=model_params.get("cat_cols", []),
            cont_cols=model_params.get("cont_cols", []),
            bin_cols=model_params.get("bin_cols", []),
            disc_cols=model_params.get("disc_cols", []),
            cat_vocab_sizes=model_params.get("cat_vocab_sizes", {}),
            d_model=model_params.get("d_model", 128),
            mode=mode,
        )

    # -------------------- Fase de ajuste (fit) --------------------

    def fit(self, X: Dict[str, torch.Tensor]) -> "TabularFeaturizer":
        """
        Aprende:
        - mappings de categorías (valor->índice)
        - medias/STD para cont/disc (z-score)
        """
        # Categóricas
        for col in self.cat_cols:
            if col not in X:
                continue

            uniques = torch.unique(X[col])

            if self.mode == "index":
                # índice 0 reservado para UNK
                mapping = {v.item(): i + 1 for i, v in enumerate(uniques)}
                inferred = len(mapping) + 1  # +1 por UNK → índices posibles: 0..len(uniques)
            else:
                # onehot: índices naturales para construir one-hot
                mapping = {v.item(): i for i, v in enumerate(uniques)}
                inferred = len(mapping)

            self.category_mappings[col] = mapping

            declared = self.cat_vocab_sizes.get(col, None)

            if declared is None:
                # Si no hay cat_vocab_sizes para esta col, usamos lo inferido
                self.feature_dims[col] = inferred
            else:
                # Por seguridad: NUNCA crear embedding más pequeña que los índices posibles
                if declared < inferred:
                    # Puedes loggear un warning si quieres:
                    # print(f"[WARN] cat_vocab_sizes['{col}']={declared} < inferred={inferred}, usando inferred.")
                    self.feature_dims[col] = inferred
                else:
                    self.feature_dims[col] = declared

        # Continuas y discretas
        for col in self.cont_cols + self.disc_cols:
            if col not in X:
                continue

            x = X[col]
            # Asegurar dtype flotante antes de mean/std
            if not torch.is_floating_point(x):
                x = x.float()

            mu = torch.mean(x).item()
            sd = torch.std(x).item()
            self.scalers[col] = {"mean": mu, "std": (sd if sd > 0 else 1.0)}
            self.feature_dims[col] = 1

        # Binarias
        for col in self.bin_cols:
            if col in X:
                self.feature_dims[col] = 1

        # Construir capas de proyección/embedding
        self._build_layers()
        self._built = True
        return self


    def _build_layers(self):
        """Crea capas en función de feature_dims y mode."""
        self.layers = nn.ModuleDict()  # reset por si se re-fit-ea

        # Proyecciones lineales para cont/disc/bin
        for col in self.cont_cols + self.disc_cols + self.bin_cols:
            if col in self.feature_dims:
                self.layers[col] = nn.Linear(1, self.d_model)

        # Categóricas
        for col in self.cat_cols:
            if col not in self.feature_dims:
                continue
            C = self.feature_dims[col]
            if self.mode == "onehot":
                # one-hot [B, C] -> Linear(C, d_model)
                self.layers[col] = nn.Linear(C, self.d_model)
            else:
                # índices [B] -> Embedding(C, d_model) (C incluye UNK=0)
                self.layers[col] = nn.Embedding(C, self.d_model)

    # -------------------- Transformación --------------------

    def transform(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Aplica transformaciones:
        - categóricas -> one-hot [B,C] (mode='onehot') o índices [B] (mode='index')
        - cont/disc   -> z-score [B,1]
        - bin         -> float [B,1]
        """
        assert self._built, "Debes llamar a fit() antes de transform/forward"
        out: Dict[str, torch.Tensor] = {}

        # Categóricas
        for col in self.cat_cols:
            if col not in X:
                continue
            mapping = self.category_mappings.get(col, {})
            idx = torch.tensor(
                [mapping.get(v.item(), 0) for v in X[col]],
                dtype=torch.long,
                device=X[col].device,
            )
            C = self.feature_dims[col]
            if self.mode == "onehot":
                one_hot = F.one_hot(idx, num_classes=C).float()
                out[col] = one_hot
            else:
                # índices [B]
                out[col] = idx

        # Continuas y discretas -> z-score
        for col in self.cont_cols + self.disc_cols:
            if col not in X:
                continue
            x = X[col].float()  # forzamos float32 siempre
            sc = self.scalers[col]
            z = (x - sc["mean"]) / (sc["std"] + 1e-8)
            out[col] = z.unsqueeze(-1)  # [B,1] float32

        # Binarias -> float [B,1]
        for col in self.bin_cols:
            if col in X:
                out[col] = X[col].float().unsqueeze(-1)

        return out



    # -------------------- Forward (transform + embed) --------------------

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Devuelve un embedding tabular de tamaño [B, d_model].
        Aplica transformaciones y suma las proyecciones por feature.
        """
        processed = self.transform(X)
        embs = []

        for name, tensor in processed.items():
            if name not in self.layers:
                continue
            layer = self.layers[name]

            if isinstance(layer, nn.Embedding):
                # Índices [B] -> [B, d_model]
                emb = layer(tensor.long())
            else:
                # Linear: acepta [B,1] (cont/disc/bin) o [B,C] (one-hot)
                emb = layer(tensor.float())

            embs.append(emb)

        if not embs:
            raise ValueError("No se generaron embeddings. Revisa columnas y fit().")

        # embs: lista de [B, d_model] → [num_features, B, d_model]
        embs_stack = torch.stack(embs, dim=0)
        # Sumar sobre features → [B, d_model]
        return embs_stack.sum(dim=0)

