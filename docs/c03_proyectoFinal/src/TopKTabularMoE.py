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

# ------ Importaciones para SMOTE y SMOTE-Tomek ---------
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

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


def resample_with_smote(
    df: pd.DataFrame,
    target_col: str,
    sampling_strategy: dict,
    use_tomek: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Aplica SMOTE o SMOTE+Tomek Links al dataset de entrenamiento.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if use_tomek:
        sm = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    X_res, y_res = sm.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled[target_col] = y_res

    return df_resampled

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
# Capa SwitchLayer con gating Top-k (Mixture of Experts)
# ============================
class SwitchLayer(layers.Layer):
    """
    Capa Mixture of Experts (MoE) con gating Top-k.
    Inspirada en Switch Transformers.
    """
    # CAMBIO: Añadido el parámetro k para el gating Top-k, con valor predeterminado de 2.
    def __init__(self, num_experts, d_ff, dropout=0.1, aux_loss_weight=0.01, k=2, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.d_ff = d_ff
        self.dropout = dropout
        self.aux_loss_weight = aux_loss_weight
        self.k = k # Número de expertos a seleccionar por token

        self.gate = layers.Dense(num_experts, use_bias=False)
        self.experts = []

    def build(self, input_shape):
        d = int(input_shape[-1])
        for i in range(self.num_experts):
            self.experts.append(
                keras.Sequential([
                    layers.Dense(self.d_ff, activation='gelu'),
                    layers.Dropout(self.dropout),
                    layers.Dense(d)
                ], name=f"expert_{i}")
            )
        super().build(input_shape)

    # CAMBIO: El método call ha sido reescrito para implementar la lógica Top-k.
    def call(self, x, training=None):
        B, T, d = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        
        # Gating
        logits = self.gate(x)
        probs = ops.softmax(logits, axis=-1)

        # Seleccionar los k mejores expertos y sus probabilidades
        top_k_probs, top_k_indices = ops.top_k(probs, k=self.k)

        # Normalizar las probabilidades de los k seleccionados para que sumen 1.
        # Esto estabiliza la salida.
        top_k_probs = top_k_probs / tf.reduce_sum(top_k_probs, axis=-1, keepdims=True)

        # Pérdida de Balanceo de Carga
        if training:
            # P_i: Probabilidad promedio del router para cada experto
            P = ops.mean(probs, axis=[0, 1])
            
            # f_i: Fracción de tokens enrutados a cada experto (considerando las k selecciones)
            # Creamos un one-hot para todas las selecciones top-k
            mask_k_one_hot = ops.one_hot(top_k_indices, self.num_experts, dtype='float32')
            # Sumamos sobre la dimensión k para saber qué expertos fueron elegidos para cada token
            experts_chosen_per_token = tf.reduce_sum(mask_k_one_hot, axis=2)
            # Calculamos la fracción de carga promediando sobre el batch y la secuencia
            f = ops.mean(experts_chosen_per_token, axis=[0, 1])
            
            aux_loss = self.aux_loss_weight * self.num_experts * ops.sum(f * P)
            self.add_loss(aux_loss)
            
        # Inicializar la salida final
        final_output = ops.zeros_like(x)

        # Iterar a través de los k expertos seleccionados para el enrutamiento
        for i in range(self.k):
            expert_indices = top_k_indices[..., i]
            gating_weights = top_k_probs[..., i]
            
            # Ponderar la entrada con los pesos del gating
            weighted_input = x * ops.expand_dims(gating_weights, axis=-1)
            
            # Dispatch (enviar cada token a su experto correspondiente)
            dispatched_input = ops.zeros_like(x)
            for e in range(self.num_experts):
                mask = ops.equal(expert_indices, e)
                
                def process():
                    # Seleccionar tokens y esparcirlos en un tensor para el experto
                    indices = tf.where(mask)
                    tokens = tf.gather_nd(weighted_input, indices)
                    return tf.tensor_scatter_nd_update(ops.zeros_like(x), indices, tokens)
                
                def skip():
                    return ops.zeros_like(x)
                
                # Usamos tf.cond para asegurar compatibilidad con el modo grafo
                branch_dispatched = tf.cond(ops.any(mask), true_fn=process, false_fn=skip)
                dispatched_input += branch_dispatched

            # Combinar (ejecutar todos los expertos en paralelo)
            expert_outputs = [self.experts[e](dispatched_input) for e in range(self.num_experts)]
            combined_output = tf.add_n(expert_outputs)
            
            # Sumar la contribución de esta k-ésima selección a la salida final
            final_output += combined_output
            
        return final_output

# ============================
# Transformer block (Pre-LayerNorm) usando SwitchLayer
# ============================
class TransformerSwitchBlock(layers.Layer):
    """Bloque Transformer con una capa Switch (MoE Top-k)"""
    # CAMBIO: Añadido k al constructor
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout=0.1, aux_loss_weight=0.01, k=2, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        # CAMBIO: Pasar k a la capa SwitchLayer
        self.switch = SwitchLayer(num_experts=num_experts, d_ff=d_ff, dropout=dropout, aux_loss_weight=aux_loss_weight, k=k)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        # Arquitectura Pre-LN para estabilidad
        normed_x = self.ln1(x)
        attn = self.mha(query=normed_x, value=normed_x, key=normed_x, attention_mask=mask, training=training)
        x = x + self.drop1(attn, training=training)

        normed_x = self.ln2(x)
        ffn = self.switch(normed_x, training=training)
        x = x + self.drop2(ffn, training=training)
        return x

# ============================
# CAMBIO: Tokenizer con Embeddings Categóricos Individuales
# ============================
def build_tabular_encoder(
    cont_cols: List[str],
    bin_cols: List[str],
    cat_cols: List[str],
    cat_vocab_sizes: Dict[str, int], # Diccionario con tamaños de vocabularios
    d_model: int
) -> Tuple[Dict[str, layers.Input], Any]:
    inputs = {}
    all_projections = []

    # Entradas continuas
    if cont_cols:
        cont_in = layers.Input(shape=(len(cont_cols),), name="cont_in")
        cont_proj = layers.Dense(d_model, name="cont_proj")(cont_in)
        all_projections.append(ops.expand_dims(cont_proj, axis=1))
        inputs["cont_in"] = cont_in

    # Entradas binarias
    if bin_cols:
        bin_in = layers.Input(shape=(len(bin_cols),), name="bin_in")
        bin_proj = layers.Dense(d_model, name="bin_proj")(bin_in)
        all_projections.append(ops.expand_dims(bin_proj, axis=1))
        inputs["bin_in"] = bin_in

    # Entradas categóricas (una por una)
    for col in cat_cols:
        vocab_size = cat_vocab_sizes[col]
        cat_in = layers.Input(shape=(1,), dtype="int32", name=f"cat_in_{col}")
        inputs[f"cat_in_{col}"] = cat_in
        embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=d_model, name=f"embedding_{col}")
        all_projections.append(embedding(cat_in))

    # Concatenar todos los tokens de características
    X = ops.concatenate(all_projections, axis=1)

    # Añadir token [CLS] aprendible
    #cls_token = layers.Embedding(input_dim=1, output_dim=d_model, name="cls_token")

    # --- CORRECCIÓN DEFINITIVA ---
    # 1. Usamos tf.shape(X) para obtener la forma DINÁMICA del tensor X.
    #    Esto devuelve un tensor 1D (ej. [batch_size, seq_len, features]), no una tupla con None.
    # 2. Extraemos el tamaño del lote, que ahora es un TENSOR escalar, no el objeto None.
    #batch_size = tf.shape(X)[0]
    
    # 3. Creamos el tensor de forma [batch_size, 1] usando tf.stack.
    #    Como batch_size ya es un tensor, tf.stack puede manejarlo sin problemas.
    #shape_for_zeros = ops.stack([batch_size, 1])
    #cls_input = tf.zeros(shape_for_zeros, dtype="int32")

    #cls = cls_token(cls_input)
    #X = ops.concatenate([cls, X], axis=1)
    # --- 5. Añadir token [CLS] ---
    def add_cls_lambda(x):
        import tensorflow as tf
        batch_size = tf.shape(x)[0]
        cls_token = tf.zeros([batch_size, 1, d_model])
        return tf.concat([cls_token, x], axis=1)

    X = layers.Lambda(add_cls_lambda)(X)
    
    return inputs, X

# ============================
# Construcción del modelo completo
# ============================
def build_switch_transformer_tabular(
    cont_cols: List[str],
    bin_cols: List[str],
    cat_cols: List[str],
    cat_vocab_sizes: Dict[str, int],
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 256,
    num_experts: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    aux_loss_weight: float = 0.01,
    # CAMBIO: Añadido k a los parámetros de construcción
    k: int = 2
):
    """Construye el modelo completo Transformer-MoE para datos tabulares."""
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Se añade la llamada explícita a TabularEncoder para definir 'inputs' y 'X'.
    # Esta es la línea que faltaba.
    inputs, X = build_tabular_encoder(
        cont_cols=cont_cols,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        cat_vocab_sizes=cat_vocab_sizes,
        d_model=d_model
    ) # Se invoca la capa inmediatamente
    # --- FIN DE LA CORRECCIÓN ---
    
    h = layers.Dropout(dropout)(X)
    for i in range(num_layers):
        # CAMBIO: Pasar el parámetro k al bloque Transformer
        h = TransformerSwitchBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
            num_experts=num_experts, dropout=dropout, 
            aux_loss_weight=aux_loss_weight,
            k=k,
            name=f"transformer_block_{i}"
        )(h)
        
    cls = h[:, 0, :]
    cls = layers.LayerNormalization(epsilon=1e-6)(cls)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(cls)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# ============================
# Función de Entrenamiento (con `df_to_input_dict` y estrategia de muestreo personalizada SMOTE / SMOTE-Tomek Links)
# ============================
def train(model: keras.Model,
          df_train: pd.DataFrame, df_val: pd.DataFrame,
          cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
          target_col: str,
          batch_size: int = 256,
          epochs: int = 30,
          lr: float = 1e-4,
          focal_loss=True,
          callbacks: Optional[List[keras.callbacks.Callback]] = None,
          smote_config: Optional[dict] = None,
          mixed_precision: bool = False):
    
    """
    Entrena el modelo. Si smote_config no es None, aplica SMOTE/SMOTE+Tomek
    al conjunto de entrenamiento antes de entrenar.

    smote_config ejemplo:
    {
        "sampling_strategy": {0: 2000, 1: 2000},
        "use_tomek": True,
        "random_state": 42
    }
    """

    # Mixed Precision: Permite procesar operaciones de float16 
    # significativamente más rápido que las de float32
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --- aplicar balanceo si está configurado ---
    if smote_config is not None:
        df_train = resample_with_smote(
            df=df_train,
            target_col=target_col,
            sampling_strategy=smote_config.get("sampling_strategy", "auto"),
            use_tomek=smote_config.get("use_tomek", False),
            random_state=smote_config.get("random_state", 42)
        )
    
    
    # --- Fin de la Lógica de Remuestreo ---

    def df_to_input_dict(df):
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs
    


    train_inputs = df_to_input_dict(df_train)
    val_inputs = df_to_input_dict(df_val)
    train_labels = df_train[target_col].values.astype("float32")
    val_labels = df_val[target_col].values.astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(df_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = binary_focal_loss() if focal_loss else keras.losses.BinaryCrossentropy(from_logits=False)



    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.BinaryAccuracy(name="accuracy")
        ],
        jit_compile=True        # Compilación del Modelo con XLA (Accelerated Linear Algebra)
    )

    # --- callbacks ---
    ckpt_path = "model.weights.best.weights.h5"
    if callbacks is None:
         callbacks = [
            EpochLogger(filepath=ckpt_path, monitor="val_auc", mode="max"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min",restore_best_weights=True)
        ]

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=2
    )
    return model, history

# ============================
# Calibración Conforme Condicional a la Clase
# ============================
def calibrate_conformal_conditional(model: keras.Model,
                                    df_cal: pd.DataFrame,
                                    cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
                                    target_col: str,
                                    alpha: float = 0.1,
                                    batch_size: int = 1024) -> Dict[str, float]:
    """
    Realiza calibración conforme condicional a la clase.
    Devuelve un diccionario con qhat para la clase 0 y la clase 1.
    - qhat_1 se usa para definir el umbral de la clase positiva: p >= 1 - qhat_1
    - qhat_0 se usa para definir el umbral de la clase negativa: p <= qhat_0
    """
    # Helper para convertir df a dataset para predicción
    def df_to_pred_dataset(df):
        inputs = {}
        if cont_cols: inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols: inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols: inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)

    ds_cal = df_to_pred_dataset(df_cal)
    probs = model.predict(ds_cal, verbose=0).reshape(-1)
    y = df_cal[target_col].astype(int).values

    # Calibración para la clase 1 (positivos)
    y1_indices = np.where(y == 1)[0]
    scores_1 = 1.0 - probs[y1_indices]
    n1 = len(scores_1)
    rank_1 = int(np.ceil((n1 + 1) * (1 - alpha)))
    qhat_1 = float(np.sort(scores_1)[min(rank_1 - 1, n1 - 1)])

    # Calibración para la clase 0 (negativos)
    y0_indices = np.where(y == 0)[0]
    scores_0 = probs[y0_indices] # Score es p, ya que 1-(1-p) = p
    n0 = len(scores_0)
    rank_0 = int(np.ceil((n0 + 1) * (1 - alpha)))
    qhat_0 = float(np.sort(scores_0)[min(rank_0 - 1, n0 - 1)])
    
    return {"qhat_0": qhat_0, "qhat_1": qhat_1}

# ============================
# CAMBIO: Nueva función para evaluar conjuntos de predicción conformes
# ============================
def evaluate_conformal_sets(model: keras.Model,
                            df_test: pd.DataFrame,
                            cont_cols: List[str], bin_cols: List[str], cat_cols: List[str],
                            target_col: str,
                            qhats: Dict[str, float],
                            batch_size: int = 1024) -> Dict[str, float]:
    """
    Evalúa el rendimiento de los conjuntos de predicción conformes.
    Calcula la cobertura empírica y el tamaño promedio del conjunto.
    """
    qhat_0, qhat_1 = qhats["qhat_0"], qhats["qhat_1"]
    
    def df_to_pred_dataset(df): # Helper anidado
        inputs = {}
        if cont_cols: inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols: inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols: inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)

    ds_test = df_to_pred_dataset(df_test)
    probs = model.predict(ds_test, verbose=0).reshape(-1)
    y_true = df_test[target_col].astype(int).values

    # Construir conjuntos de predicción
    prediction_sets = []
    for p in probs:
        s = []
        if p <= qhat_0:
            s.append(0)
        if p >= (1 - qhat_1):
            s.append(1)
        prediction_sets.append(s)

    # Calcular métricas
    correct_predictions = 0
    total_set_size = 0
    for i, pred_set in enumerate(prediction_sets):
        if y_true[i] in pred_set:
            correct_predictions += 1
        total_set_size += len(pred_set)

    coverage = correct_predictions / len(y_true)
    avg_set_size = total_set_size / len(y_true)
    
    return {"coverage": coverage, "average_set_size": avg_set_size}

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