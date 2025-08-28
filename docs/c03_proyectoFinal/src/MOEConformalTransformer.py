import tensorflow as tf
from tensorflow import keras
from keras import layers, ops
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)
from typing import List, Dict, Tuple, Optional, Any

class EpochLogger(Callback):
    """
    Custom callback to format output like:
    Epoch 1: val_auc improved from -inf to 0.67143, saving model to model.weights.best.keras
    """
    def __init__(self, filepath, monitor="val_auc", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.best = -np.inf if mode == "max" else np.inf
        self.filepath = filepath
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if (self.mode == "max" and current > self.best) or (self.mode == "min" and current < self.best):
            prev_best = self.best
            self.best = current
            print(f"\nEpoch {epoch+1}: {self.monitor} improved from {prev_best:.5f} to {current:.5f}, "
                  f"saving model to {self.filepath}")
            self.model.save_weights(self.filepath)
        else:
            print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f}")

# ============================
# UTIL: crear tf.data desde DataFrame
# ============================
def df_to_tf_dataset(X: pd.DataFrame, y: pd.Series,
                     batch_size: int = 256,
                     shuffle: bool = True,
                     repeat: bool = False) -> tf.data.Dataset:
    """
    Convierte X,y (pandas) a tf.data.Dataset -> (dict_of_inputs, label)
    Se asume que para columnas categóricas convertiste a str si usas StringLookup.
    """
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y.values.astype("float32")))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for binary classification.
    """
    def loss(y_true, y_pred):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = - (y_true * ops.log(y_pred) + (1 - y_true) * ops.log(1 - y_pred))
        
        # CAMBIO: ops.pow -> ops.power
        weight = alpha * ops.power(1 - y_pred, gamma) * y_true + \
                 (1 - alpha) * ops.power(y_pred, gamma) * (1 - y_true)
        
        return ops.mean(weight * cross_entropy)
    return loss

# ============================
# MoE (top-1) layer - sencillo y robusto
# ============================
class SimpleMoE(layers.Layer):
    """
    Mixture of Experts simplificado - top-1 gating.
    Entrada: [B, T, D]
    Salida:  [B, T, D]
    """
    def __init__(self, num_experts: int, d_ff: int, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.d_ff = d_ff
        self.dropout = dropout
        # gating
        self.gate = layers.Dense(num_experts)
        # experts (stacked Dense)
        self.expert_fwd = [keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(None)  # we'll set output dim at build time (hack below)
        ]) for _ in range(num_experts)]
        # We'll re-create experts in build to set final Dense output dim properly

    def build(self, input_shape):
        # input_shape: (B, T, D)
        d = int(input_shape[-1])
        # recreate experts with correct output dims
        self.experts = []
        for i in range(self.num_experts):
            seq = keras.Sequential([
                layers.Dense(self.d_ff, activation='gelu'),
                layers.Dropout(self.dropout),
                layers.Dense(d)   # back to d_model
            ], name=f"expert_{i}")
            self.experts.append(seq)
        super().build(input_shape)

    def call(self, x, training=None):
        """
        x: [B, T, d_model]
        """
        import tensorflow as tf
        from keras import ops
        
        B, T, d = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        
        # Gating
        logits = self.gate(x)  # [B, T, num_experts]
        top1 = ops.argmax(logits, axis=-1)  # [B, T]
        
        # Flatten para procesamiento
        flat_x = ops.reshape(x, [B * T, d])  # [B*T, d]
        flat_top1 = ops.reshape(top1, [B * T])  # [B*T]
        
        # Inicializar output
        output = ops.zeros_like(flat_x)  # [B*T, d]
        
        # Procesar cada experto usando tf.cond en lugar de if/continue
        for e in range(self.num_experts):
            # Crear máscara para este experto
            mask = ops.equal(flat_top1, e)  # [B*T] boolean
            mask_f = ops.cast(mask, x.dtype)  # [B*T] float
            
            # Usar tf.cond para manejar el caso donde no hay tokens para este experto
            def process_expert():
                inp_e = flat_x * mask_f[:, None]  # [B*T, d]
                out_e = self.experts[e](inp_e, training=training)  # [B*T, d]
                return out_e * mask_f[:, None]  # Solo mantener outputs relevantes
            
            def skip_expert():
                return ops.zeros_like(flat_x)
            
            # Verificar si hay tokens para procesar
            has_tokens = ops.any(mask)
            expert_output = tf.cond(has_tokens, process_expert, skip_expert)
            
            output = output + expert_output
        
        # Reshape back
        output = ops.reshape(output, [B, T, d])
        return output

# ============================
# Transformer block + MoE FFN
# ============================
class TransformerMoEBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.moe = SimpleMoE(num_experts=num_experts, d_ff=d_ff, dropout=dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn = self.mha(x, x, attention_mask=mask, training=training)
        x = self.ln1(x + self.drop1(attn, training=training))
        ffn = self.moe(x, training=training)
        x = self.ln2(x + self.drop2(ffn, training=training))
        return x

# ============================
# Tokenizer tabular (feature -> token)
# ============================
def build_tabular_encoder(
    df_train: pd.DataFrame,
    cont_cols: List[str],
    bin_cols: List[str],
    cat_cols: List[str],
    d_model: int
) -> Tuple[Dict[str, layers.Input], Any]:
    """
    Construye un encoder tabular con embeddings compartidos para variables categóricas,
    proyecciones densas para variables continuas/binarias y un token [CLS].

    Args:
        df_train: DataFrame de entrenamiento (para determinar vocab sizes de categóricas).
        cont_cols: Columnas numéricas continuas.
        bin_cols: Columnas binarias (0/1).
        cat_cols: Columnas categóricas.
        d_model: Dimensión de embedding común.

    Returns:
        inputs: diccionario de entradas Keras
        X: Tensor resultante de shape [batch, T+1, d_model]
    """

    inputs = {}

    # --- 1. Entradas numéricas ---
    cont_in = layers.Input(shape=(len(cont_cols),), name="cont_in")
    if len(cont_cols) > 0:
        cont_proj = layers.Dense(d_model)(cont_in)
        cont_proj = ops.expand_dims(cont_proj, axis=1)  # [B, 1, d_model]
    else:
        cont_proj = None
    inputs["cont_in"] = cont_in

    # --- 2. Entradas binarias ---
    bin_in = layers.Input(shape=(len(bin_cols),), name="bin_in")
    if len(bin_cols) > 0:
        bin_proj = layers.Dense(d_model)(bin_in)
        bin_proj = ops.expand_dims(bin_proj, axis=1)  # [B, 1, d_model]
    else:
        bin_proj = None
    inputs["bin_in"] = bin_in

    # --- 3. Embedding compartido para categóricas ---
    cat_in = layers.Input(shape=(len(cat_cols),), dtype="int32", name="cat_in")
    inputs["cat_in"] = cat_in

    if len(cat_cols) > 0:
        # Determinar vocabulario máximo entre todas las categóricas
        max_vocab_size = 0
        for c in cat_cols:
            vocab = df_train[c].nunique()
            max_vocab_size = max(max_vocab_size, vocab)

        shared_embedding = layers.Embedding(
            input_dim=max_vocab_size + 1,  # +1 para OOV/padding
            output_dim=d_model,
            name="shared_cat_embedding"
        )

        cat_emb = shared_embedding(cat_in)  # [B, T_cat, d_model]
    else:
        cat_emb = None

    # --- 4. Concatenar todas las fuentes ---
    all_parts = []
    for part in [cont_proj, bin_proj, cat_emb]:
        if part is not None:
            all_parts.append(part)

    if len(all_parts) > 1:
        X = ops.concatenate(all_parts, axis=1)  # [B, T, d_model]
    else:
        X = all_parts[0]  # solo uno presente

    # --- 5. Añadir token [CLS] ---
    def add_cls_lambda(x):
        import tensorflow as tf
        batch_size = tf.shape(x)[0]
        cls_token = tf.zeros([batch_size, 1, d_model])
        return tf.concat([cls_token, x], axis=1)

    X = layers.Lambda(add_cls_lambda)(X)

    return inputs, X


def self_add_cls_token(X, d_model):
    """
    Añade un token [CLS] al inicio de la secuencia.
    X: tensor de forma [B, T, d_model] (KerasTensor)
    Retorna: tensor de forma [B, 1, d_model]
    """
    from keras import ops
    
    # Crear token [CLS] base [1, 1, d_model]
    cls_base = ops.zeros((1, 1, d_model))
    
    # Obtener batch size dinámico usando ops de Keras
    batch_size = ops.shape(X)[0]
    
    # Repetir para cada muestra en el batch
    cls_token = ops.repeat(cls_base, batch_size, axis=0)
    
    return cls_token

# ============================
# Build full model
# ============================
def build_moe_transformer_tabular(df_train: pd.DataFrame,
                                  cont_cols: List[str],
                                  bin_cols: List[str],
                                  cat_cols: List[str],
                                  d_model: int = 64,
                                  num_heads: int = 4,
                                  d_ff: int = 256,
                                  num_experts: int = 4,
                                  num_layers: int = 2,
                                  dropout: float = 0.1):
    inputs, X = build_tabular_encoder(df_train, cont_cols, bin_cols, cat_cols, d_model=d_model)
    h = layers.Dropout(dropout)(X)
    for i in range(num_layers):
        h = TransformerMoEBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_experts=num_experts, dropout=dropout)(h)
    cls = layers.Lambda(lambda t: t[:, 0, :])(h)
    cls = layers.LayerNormalization(epsilon=1e-6)(cls)
    cls = layers.Dropout(dropout)(cls)
    logits = layers.Dense(1, name='logits')(cls)   # logits (sigmoid later)
    model = keras.Model(inputs=inputs, outputs=logits, name='MoE_Transformer_Tabular')
    return model

# ============================
# TRAIN function
# ============================
def train(model: keras.Model,
          df_train: pd.DataFrame, df_val: pd.DataFrame,
          cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
          target_col: str,
          batch_size: int = 256,
          epochs: int = 30,
          class_weight: Optional[Dict[int, float]] = None,
          lr: float = 3e-4,
          callbacks: Optional[List[keras.callbacks.Callback]] = None,
          mixed_precision: bool = False,
          focal_loss=False):
    """
    Entrena el modelo con tf.data. Devuelve model y history.
    class_weight: {0: w0, 1: w1} para manejar desbalanceo.
    """
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --- helper para convertir df -> inputs del modelo ---
    def df_to_input_dict(df, cont_cols, bin_cols, cat_cols, target_col):
        cont = ops.convert_to_tensor(df[cont_cols].astype("float32").values, dtype="float32") if cont_cols else ops.zeros((len(df), 0))
        binv = ops.convert_to_tensor(df[bin_cols].astype("float32").values, dtype="float32") if bin_cols else ops.zeros((len(df), 0))

        if cat_cols:
            # convertir cada col categórica a códigos enteros
            cat_encoded = []
            for c in cat_cols:
                cat_encoded.append(pd.Categorical(df[c]).codes)
            cat_matrix = np.stack(cat_encoded, axis=1)  # [N, num_cat_cols]
            cat = ops.convert_to_tensor(cat_matrix.astype("int32"), dtype="int32")
        else:
            cat = ops.zeros((len(df), 0))

        y = ops.convert_to_tensor(df[target_col].astype("int32").values, dtype="int32")

        return {
            "cont_in": cont,
            "bin_in": binv,
            "cat_in": cat
        }, y

    # --- datasets ---
    train_inputs, train_labels = df_to_input_dict(df_train, cont_cols, bin_cols, cat_cols, target_col)
    val_inputs, val_labels = df_to_input_dict(df_val, cont_cols, bin_cols, cat_cols, target_col)

    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(df_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- optimizador ---
    opt = keras.optimizers.Adam(learning_rate=lr)

    if focal_loss:
        loss_fn = binary_focal_loss(gamma=2., alpha=0.25)
    else:
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.BinaryAccuracy(name="accuracy")
        ],
    )


    # --- callbacks ---
    ckpt_path = "model.weights.best.weights.h5"
    if callbacks is None:
        callbacks = [
            EpochLogger(filepath=ckpt_path, monitor="val_auc", mode="max"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min",restore_best_weights=True)
        ]

    # --- entrenamiento ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    return model, history

# ============================
# Conformal calibration (split-conformal)
# ============================
def calibrate_conformal(model: keras.Model,
                        df_cal: pd.DataFrame,
                        cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
                        target_col: str,
                        alpha: float = 0.1,
                        batch_size: int = 1024) -> Tuple[float, float]:
    """
    Devuelve qhat y threshold p >= threshold
    qhat = cuantil de no-conformidad = 1 - p_true
    """
    ds_cal = df_to_tf_dataset(df_cal[cont_cols+bin_cols+cat_cols], df_cal[target_col], batch_size=batch_size, shuffle=False)
    logits = model.predict(ds_cal, verbose=0)
    probs = tf.sigmoid(logits).numpy().reshape(-1)
    y = df_cal[target_col].astype(int).values
    p_true = np.where(y == 1, probs, 1 - probs)
    scores = 1.0 - p_true
    n = len(scores)
    # quantile with higher method like split-conformal
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    qhat = float(np.sort(scores)[max(0, rank-1)])
    threshold = 1.0 - qhat
    return qhat, threshold

# ============================
# Evaluate -> metrics DataFrame
# ============================
def evaluate(model: keras.Model,
             df_test: pd.DataFrame,
             cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
             target_col: str,
             threshold: float = 0.5,
             batch_size: int = 1024,
             thresholds: Optional[List[float]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Evalúa el modelo en df_test. Si thresholds list se evalúa para cada threshold y retorna df con filas por threshold.
    Devuelve (results_df, last_conf_matrix_dict)
    """
    ds_test = df_to_tf_dataset(df_test[cont_cols+bin_cols+cat_cols], df_test[target_col], batch_size=batch_size, shuffle=False)
    logits = model.predict(ds_test, verbose=0)
    probs = tf.sigmoid(logits).numpy().reshape(-1)
    y_true = df_test[target_col].astype(int).values

    thr_list = thresholds if thresholds is not None else [threshold]
    rows = []
    last_cm = {}
    for thr in thr_list:
        y_pred = (probs >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, probs)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append({
            "Threshold": round(thr, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "Accuracy": round(acc, 3),
            "ROC-AUC": round(roc, 3),
            "TN": int(tn), "TP": int(tp), "FP": int(fp), "FN": int(fn)
        })
        last_cm = {"tn":int(tn), "tp":int(tp), "fp":int(fp), "fn":int(fn)}

    results_df = pd.DataFrame(rows)
    return results_df, last_cm

# ============================
# UTILS: generar class_weight automáticamente
# ============================
def make_class_weight(y: pd.Series, mu: float = 0.15) -> Dict[int, float]:
    """
    Empiric class weight (sklearn-like) - puedes ajustar.
    """
    value_counts = y.value_counts().to_dict()
    total = len(y)
    # inverse freq
    weights = {cls: total/count for cls, count in value_counts.items()}
    # normalize so that min weight = 1
    min_w = min(weights.values())
    weights_norm = {cls: float(w/min_w) for cls, w in weights.items()}
    return weights_norm