import tensorflow as tf
from tensorflow import keras
from keras import layers, ops
from typing import List, Dict, Tuple, Any

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