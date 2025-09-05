import os
# Establece el nivel de registro de logs de TensorFlow para ocultar los mensajes de INFO.
# 0 = Muestra todos los mensajes (default)
# 1 = Filtra los mensajes INFO
# 2 = Filtra los mensajes INFO y WARNING
# 3 = Filtra los mensajes INFO, WARNING, y ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, ops
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
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
    Callback personalizado que guarda el mejor modelo y loguea el progreso de la época.
    """
    def __init__(self, filepath, monitor="val_f1_score", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.mode = mode
        
        if mode == "max":
            self.best = -np.inf
            self.monitor_op = np.greater
        else: # mode == "min"
            self.best = np.inf
            self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
            
        # Extraer el valor numérico del tensor si es necesario
        if hasattr(current, 'numpy'):
            current = current.numpy()

        if self.monitor_op(current, self.best):
            prev_best = self.best
            self.best = current
            print(f"\nEpoch {epoch+1}: {self.monitor} improved from {prev_best:.5f} to {current:.5f}, "
                  f"saving model to {self.filepath}")
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            print(f"\nEpoch {epoch+1}: {self.monitor} did not improve from {self.best:.5f}")




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

class F1ScoreMetric(keras.metrics.Metric):
    """
    Métrica F1-Score 'con estado' (stateful) para Keras.
    
    Esta implementación es numéricamente estable y acumula los valores
    a lo largo de los lotes de una época antes de calcular el resultado final.
    """
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Usamos las métricas internas de Keras para manejar el estado
        self.precision = keras.metrics.Precision(thresholds=threshold)
        self.recall = keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # TensorFlow se encarga de castear y_true al tipo de y_pred
        # Actualizamos el estado de nuestras métricas internas
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        # Calculamos el F1-score a partir de los resultados de las métricas internas
        precision_result = self.precision.result()
        recall_result = self.recall.result()
        # Usamos tf.math.divide_no_nan para evitar divisiones por cero
        f1 = tf.math.divide_no_nan(
            2 * precision_result * recall_result,
            precision_result + recall_result
        )
        return f1

    def reset_state(self):
        # Se llama al inicio de cada época
        self.precision.reset_state()
        self.recall.reset_state()

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
    
    def call(self, x, training=False):
        """
        x: Tensor (batch, seq, d_model)
        return: Tensor (batch, seq, d_model)
        """
        import tensorflow as tf
        from keras import ops

        # Validaciones básicas
        tf.debugging.assert_rank(x, 3, message="SwitchLayer espera (batch, seq, d_model)")
        x_dtype = x.dtype

        # ---- Gating ----
        logits = self.gate(x)                 # (b, s, E), dtype ~ x.dtype
        P = ops.softmax(logits, axis=-1)      # (b, s, E)
        P = ops.cast(P, x_dtype)              # para mezclar con salidas de expertos

        # ---- Top-k gating ----
        # tf.math.top_k opera sobre tensores tf.*; convertimos explícitamente
        topk_vals, topk_idx = tf.math.top_k(tf.convert_to_tensor(P), k=self.k)  # (b, s, k)

        # Máscara one-hot por experto (b, s, E), sumando las k selecciones
        # one_hot sobre la última dimensión E, luego reducimos el eje k
        one_hot_masks = tf.one_hot(topk_idx, depth=self.num_experts, dtype=x_dtype)  # (b, s, k, E)
        topk_mask = tf.reduce_sum(one_hot_masks, axis=2)  # (b, s, E), valores 0/1

        # Mantener solo top-k pesos y renormalizar por token
        P_topk = P * topk_mask                             # (b, s, E) – 0 fuera de top-k
        denom = tf.reduce_sum(P_topk, axis=-1, keepdims=True)  # (b, s, 1)
        denom = denom + tf.cast(1e-9, x_dtype)             # estabilidad numérica
        P_norm = P_topk / denom                            # (b, s, E) suma=1 por token

        # ---- Mezcla de expertos ----
        # Evaluamos todos los expertos y ponderamos por P_norm[..., e]
        outputs = []
        for e, expert in enumerate(self.experts):
            y_e = expert(x, training=training)             # (b, s, d_model)
            y_e = ops.cast(y_e, x_dtype)                   # asegurar dtype consistente
            w_e = ops.expand_dims(P_norm[..., e], axis=-1) # (b, s, 1)
            outputs.append(y_e * w_e)

        y = tf.add_n(outputs)                              # (b, s, d_model)

        # ---- Auxiliary Load-Balancing Loss (opcional y en float32) ----
        if training and (self.aux_loss_weight is not None) and (self.aux_loss_weight > 0.0):
            # Fracción de carga por experto (proporción de tokens en top-k)
            f = ops.mean(topk_mask, axis=[0, 1])           # (E,)
            # Promedio de pesos asignados (solo top-k ya renormalizados)
            P_bar = ops.mean(P_norm, axis=[0, 1])          # (E,)

            # Calcular la aux loss en float32 para mayor estabilidad bajo mixed precision
            f32       = tf.cast(f, tf.float32)
            P32       = tf.cast(P_bar, tf.float32)
            aux_w32   = tf.cast(self.aux_loss_weight, tf.float32)
            num_exp32 = tf.cast(self.num_experts, tf.float32)
            aux_loss  = aux_w32 * num_exp32 * tf.reduce_sum(f32 * P32)

            self.add_loss(aux_loss)

        return y


    # CAMBIO: El método call ha sido reescrito para implementar la lógica Top-k.
    def call_base(self, x, training=None):
        B, T, d = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        
        # Gating
        logits = self.gate(x)
        
        # --- INICIO DE LA CORRECCIÓN ---
        # Convertir logits a float32 antes de softmax para estabilidad numérica
        # y para asegurar que todos los cálculos de la pérdida auxiliar usen float32.
        # Esto soluciona el error de tipo de dato con la precisión mixta.
        if logits.dtype != 'float32':
            logits = ops.cast(logits, 'float32')
        # --- FIN DE LA CORRECCIÓN ---

        probs = ops.softmax(logits, axis=-1)

        # Seleccionar los k mejores expertos y sus probabilidades
        top_k_probs, top_k_indices = ops.top_k(probs, k=self.k)

        # Normalizar las probabilidades de los k seleccionados para que sumen 1.
        # Esto estabiliza la salida.
        top_k_probs = top_k_probs / tf.reduce_sum(top_k_probs, axis=-1, keepdims=True)

        # Pérdida de Balanceo de Carga
        if training:
            # P_i: Probabilidad promedio del router para cada experto
            # Como 'probs' ahora es float32, 'P' también lo será.
            P = ops.mean(probs, axis=[0, 1])
            
            # f_i: Fracción de tokens enrutados a cada experto (considerando las k selecciones)
            # Creamos un one-hot para todas las selecciones top-k
            mask_k_one_hot = tf.one_hot(top_k_indices, self.num_experts, dtype='float32')
            # Sumamos sobre la dimensión k para saber qué expertos fueron elegidos para cada token
            experts_chosen_per_token = tf.reduce_sum(mask_k_one_hot, axis=2)
            # Calculamos la fracción de carga promediando sobre el batch y la secuencia
            # Como 'mask_k_one_hot' es float32, 'f' también lo será.
            f = ops.mean(experts_chosen_per_token, axis=[0, 1])
            
            # Ahora, la multiplicación f * P será entre dos tensores float32.
            aux_loss = self.aux_loss_weight * self.num_experts * ops.sum(f * P)
            self.add_loss(aux_loss)
            
        # Inicializar la salida final
        # La salida final debe tener el mismo tipo de dato que la entrada original 'x'.
        final_output = ops.zeros_like(x)

        # Iterar a través de los k expertos seleccionados para el enrutamiento
        for i in range(self.k):
            expert_indices = top_k_indices[..., i]
            # Convertimos los pesos de vuelta al dtype original si es necesario
            gating_weights = tf.cast(top_k_probs[..., i], dtype=x.dtype)
            
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

    # --- 5. Añadir token [CLS] ---
    # def add_cls_lambda(x):
    #    import tensorflow as tf
    #    batch_size = tf.shape(x)[0]
    #    cls_token = tf.zeros([batch_size, 1, d_model])
    #    return tf.concat([cls_token, x], axis=1)

    def add_cls_lambda(x):
        # x: (batch, seq_len, d_model)
        tf.debugging.assert_rank(x, 3, message="Esperaba (batch, seq, d_model)")
        x_dtype = x.dtype
        cls = tf.zeros_like(x[:, :1, :], dtype=x_dtype)
        x = tf.cast(x, x_dtype)
        out = tf.concat([cls, x], axis=1)
        tf.debugging.assert_type(out, x_dtype)
        return out
    
    def compute_cls_output_shape(input_shape):
        # input_shape es una tupla tipo (batch, seq_len, d_model)
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    X = layers.Lambda(add_cls_lambda, output_shape=compute_cls_output_shape)(X)

    # X = layers.Lambda(add_cls_lambda)(X)


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

# ==============================================================================
# FUNCIÓN DE ENTRENAMIENTO OPTIMIZADA (VERSIÓN 2.0)
# ==============================================================================
def train_(model: keras.Model,
          df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          cont_cols: List[str],
          bin_cols: List[str],
          cat_cols: List[str],
          target_col: str,
          batch_size: int = 256,
          epochs: int = 30,
          lr: float = 1e-4,
          focal_loss: bool = True,
          callbacks: Optional[List[keras.callbacks.Callback]] = None,
          mixed_precision: bool = True): # CAMBIO: True por defecto para máximo rendimiento
    """
    Entrena el modelo usando un pipeline de tf.data optimizado y vectorizado.
    
    CAMBIOS CLAVE:
    - Se eliminó el `data_generator` y el parámetro `smote_config`. El remuestreo
      debe realizarse ANTES de llamar a esta función.
    - Se utiliza `tf.data.Dataset.from_tensor_slices` para una ingesta de datos
      de alto rendimiento, eliminando el cuello de botella de `iterrows`.
    - Se recomienda ejecutar la construcción y compilación del modelo dentro de un
      `strategy.scope()` para entrenamiento distribuido.
    """

    # --- CAMBIO: Mixed Precision activado por defecto ---
    # Permite un rendimiento significativamente mayor en GPUs compatibles (Tensor Cores)
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --- CAMBIO: Conversión vectorizada de DataFrame a diccionario de tensores ---
    # Este helper convierte eficientemente un DataFrame a un formato que TensorFlow puede ingerir.
    def df_to_input_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs

    # 1. Convertir los DataFrames a diccionarios de arrays NumPy (una sola vez)
    print("Convirtiendo DataFrames a tensores para el entrenamiento...")
    train_inputs_dict = df_to_input_dict(df_train)
    train_labels = df_train[target_col].values.astype("float32")
    
    val_inputs_dict = df_to_input_dict(df_val)
    val_labels = df_val[target_col].values.astype("float32")

    # 2. Crear los Datasets de TensorFlow de forma eficiente
    print("Creando datasets de TensorFlow...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_inputs_dict, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_inputs_dict, val_labels))

    # 3. Aplicar caching, shuffle y batching (sin cambios, ya era correcto)
    # El `cache()` ahora guardará los tensores preprocesados, siendo mucho más rápido.
    train_ds = train_ds.cache().shuffle(buffer_size=len(df_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 4. Compilar y entrenar el modelo (sin cambios en la lógica)
    # NOTA: La compilación debe ocurrir dentro de `strategy.scope()` en tu script principal.
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = binary_focal_loss() if focal_loss else keras.losses.BinaryCrossentropy(from_logits=False)
    f1_metric = F1ScoreMetric()

    # Se asume que el modelo ya está compilado antes de llamar a train().
    # Si no, se compila aquí.
    if model.optimizer is None:
        print("Compilando el modelo...")
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                f1_metric,
                keras.metrics.BinaryAccuracy(name="accuracy")
            ]#,
            #jit_compile=True
        )

    # --- callbacks ---
    ckpt_path = "model.weights.best.weights.h5"

    # 1. Callback para guardar el mejor modelo basado en F1-score
    model_checkpoint = EpochLogger(
        filepath="model.weights.best.keras",
        monitor=f1_score,
        mode="max"
    )

    # 2. Callback para detener el entrenamiento si el F1-score no mejora
    early_stopping = EarlyStopping(
        monitor=f1_score,       # Métrica a monitorear
        patience=7,             # Épocas a esperar antes de detener si no hay mejora
        mode="max",             # 'max' porque un F1-score más alto es mejor
        restore_best_weights=True, # Vuelve a los pesos del mejor F1-score al finalizar
        verbose=1
    )

    # 3. Callback para reducir la tasa de aprendizaje si el F1-score se estanca
    reduce_lr = ReduceLROnPlateau(
        monitor=f1_score,       # Métrica a monitorear
        factor=0.5,             # Factor por el cual se reduce la tasa de aprendizaje (new_lr = lr * factor)
        patience=3,             # Épocas a esperar antes de reducir la tasa de aprendizaje
        mode="max",             # 'max' porque un F1-score más alto es mejor
        min_lr=1e-6,            # Tasa de aprendizaje mínima
        verbose=1
    )

    # Agrupar todos los callbacks en una sola lista
    callbacks_list = [
        model_checkpoint,
        early_stopping,
        reduce_lr
    ]
    if callbacks is None:
         callbacks = callbacks_list
        

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=2
    )

    # --- cleanup datasets/tensores para liberar memoria entre folds ---
    try:
        del train_ds, val_ds, train_inputs_dict, val_inputs_dict, train_labels, val_labels
    except Exception:
        pass
    import gc as _gc; _gc.collect()

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
    # --- INICIO DE LA CORRECCIÓN ---

    # 1. Replicar la misma lógica de preparación de datos que en `train_`
    def df_to_input_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        inputs = {}
        if cont_cols:
            inputs["cont_in"] = df[cont_cols].values.astype("float32")
        if bin_cols:
            inputs["bin_in"] = df[bin_cols].values.astype("float32")
        for col in cat_cols:
            inputs[f"cat_in_{col}"] = df[[col]].values.astype("int32")
        return inputs

    # Convertir el DataFrame de test al formato de diccionario correcto
    test_inputs_dict = df_to_input_dict(df_test)
    
    # Crear el dataset de TensorFlow a partir del diccionario
    # No incluimos las etiquetas aquí porque `model.predict` solo necesita las características
    ds_test = tf.data.Dataset.from_tensor_slices(test_inputs_dict).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # `model.predict` devuelve las probabilidades directamente porque la última capa ya tiene activación "sigmoid"
    probs = model.predict(ds_test, verbose=0).reshape(-1)
    
    # --- FIN DE LA CORRECCIÓN ---
    
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


# ==============================================================================
# FUNCIÓN DE VALIDACIÓN CRUZADA (VERSIÓN 2.0 - CON CONTROL DE REMUESTREO)
# ==============================================================================
def run_cross_validation(
    df_full_train: pd.DataFrame,
    target_col: str,
    strategy: tf.distribute.Strategy,
    model_build_params: Dict,
    training_params: Dict,
    n_splits: int = 5,
    apply_smote: bool = True,
    smote_sampler: str = 'SMOTETomek',
    smote_strategy: Any = 'auto'
) -> pd.DataFrame:
    """
    Ejecuta una validación cruzada estratificada con control sobre el remuestreo.

    Args:
        ... (parámetros anteriores) ...
        apply_smote (bool): Si es True, aplica el remuestreo. Por defecto es True.
        smote_sampler (str): El tipo de remuestreador a usar ('SMOTE' o 'SMOTETomek').
        smote_strategy (Any): La estrategia para el remuestreo ('auto', float, o dict).
    
    Returns:
        Un DataFrame de Pandas con los resultados agregados de todos los folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df_full_train.drop(columns=[target_col])
    y = df_full_train[target_col]
    all_fold_results = []
    trained_models = []
    best_model = None
    best_f1 = -np.inf

    print(f"Iniciando validación cruzada con {n_splits} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print("-" * 50)
        print(f"--- FOLD {fold + 1}/{n_splits} ---")
        print("-" * 50)

        train_fold_df = df_full_train.iloc[train_idx]
        val_fold_df = df_full_train.iloc[val_idx]
        
        # Asignamos el DF de entrenamiento a una nueva variable
        train_fold_to_use = train_fold_df

        # --- NUEVO: Bloque de remuestreo condicional ---
        if apply_smote:
            print(f"Fold {fold+1}: Aplicando {smote_sampler} con estrategia: {smote_strategy}...")
            
            if smote_sampler == 'SMOTETomek':
                sampler = SMOTETomek(sampling_strategy=smote_strategy, random_state=42, n_jobs=-1)
            elif smote_sampler == 'SMOTE':
                sampler = SMOTE(sampling_strategy=smote_strategy, random_state=42, n_jobs=-1)
            else:
                raise ValueError("El parámetro smote_sampler debe ser 'SMOTE' o 'SMOTETomek'")

            X_train_fold = train_fold_df.drop(columns=[target_col])
            y_train_fold = train_fold_df[target_col]
            X_res, y_res = sampler.fit_resample(X_train_fold, y_train_fold)
            
            # El DF que se usará para entrenar es el remuestreado
            train_fold_to_use = pd.concat([X_res, y_res], axis=1)
            print(f"Remuestreo completado. Nuevo tamaño: {len(train_fold_to_use)}")
        else:
            print(f"Fold {fold+1}: No se aplicará remuestreo.")

        # Limpieza de grafo/recursos entre folds para evitar fugas y fragmentación
        tf.keras.backend.clear_session()
        import gc; gc.collect()
        print(f"Fold {fold+1}: Construyendo y compilando un nuevo modelo...")
        

        with strategy.scope():
            model = build_switch_transformer_tabular(**model_build_params)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=training_params['lr']),
                loss=binary_focal_loss(),
                metrics=[F1ScoreMetric(name="f1_score"), keras.metrics.AUC(name="auc")],
                jit_compile=False
            )
        
        # Se usa la variable `train_fold_to_use` que contiene los datos correctos
        model, history = train_(
            model=model,
            df_train=train_fold_to_use,
            df_val=val_fold_df,
            batch_size=training_params['batch_size'],
            epochs=training_params['epochs'],
            **training_params['col_groups']
        )

        print(f"Fold {fold+1}: Evaluando el modelo...")
        results_df, _ = evaluate(
            model=model,
            df_test=val_fold_df,
            threshold=0.5,
            **training_params['col_groups']
        )
        results_df['fold'] = fold + 1
        all_fold_results.append(results_df)

    # Consolidar y promediar los resultados (sin cambios)
    final_results_df = pd.concat(all_fold_results, ignore_index=True)
    summary = final_results_df.drop(columns=['fold', 'Threshold']).mean()
    summary_std = final_results_df.drop(columns=['fold', 'Threshold']).std()
    summary_df = pd.DataFrame({'Mean': summary, 'Std Dev': summary_std})

    print("\n" + "="*50)
    print("Resumen de la Validación Cruzada")
    print("="*50)
    print(summary_df)

    return final_results_df

