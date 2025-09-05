# Clasificación de Riesgo en Pacientes con Diabetes: Comparativa de Modelos Clásicos de Machine Learning Frente a Propuesta de Modelo Híbrido

$$
\begin{align}
\textit{Leyniker Escobar Brache, Carlos Hodwalker Martínez, Tulio Orozco Torres} \\
\textit{Maestría en Estadística Aplicada, Universidad del Norte} \\
\small{\textit{Septiembre 05 de 2025}}
\end{align}
$$


#### Proyecto Integrador de Aprendizaje Automático

## Resumen

La diabetes mellitus es una enfermedad metabólica crónica caracterizada por niveles elevados de glucosa en sangre, ocasionados por una producción insuficiente de insulina o por la resistencia del organismo a su acción. Esta condición altera la capacidad del cuerpo para transformar los alimentos en energía y, si no se controla, puede generar complicaciones graves como enfermedades cardiovasculares, insuficiencia renal, daño ocular y neuropatías.
En este estudio, se emplea un conjunto de datos obtenido de la plataforma Kaggle, que recopila información clínica, demográfica y de hábitos de vida de 100.000 individuos. Dicho dataset ofrece variables relevantes que incluyen:

* Datos demográficos: género, edad, raza y ubicación geográfica.
* Indicadores clínicos y de salud: hipertensión, enfermedades cardíacas, historial de tabaquismo, índice de masa corporal (IMC), hemoglobina glicosilada (HbA1c), glucemia y otros parámetros biomédicos.
* Variable objetivo: diagnóstico de diabetes (positivo o negativo).

El enfoque del trabajo se centra en una comparativa entre algoritmos clásicos de machine learning y una propuesta híbrida que integra `Mixture of Experts` y `Conformal Prediction` y establecer la solución más óptima para clasificar pacientes con y sin diagnóstico de diabetes, buscando predicciones más confiables que apoyen la toma de decisiones médicas.  


## Algoritmos a Evaluar

A continuación, se define el alcance de los algoritmos a evaluar y que son del interés de este artículo.

* `KNeighborsClassifier`
* `LogisticRegression`
* `DecisionTreeClassifier`
* `Random Forest`
* `XGBoost`
* `SVC` (Support Vector Machine/Máquinas de Vectores de Soporte)
* `MLP` (Multilayer Perceptron/Red Neuronal Multicapa)
* `MOE Conformal Transformer` (Mixture of Experts-Conformal Prediction-Transformed Based)

El propósito de este documento es desarrollar y presentar un *benchmark* entre todos los algoritmos citados y compararlos, especialmente, con `MOE Conformal Transformer`. Sin embargo, antes de la implementación o entrenamiento se abordará la fundamentacieon de la propuesta novedosa de este ejercicio.

### Propuesta: `Mixture of Experts-Conformal Prediction-Transformed Based (MOE Conformal Transformer)`

Para entender qué es `MoE-Conformal Transformer`, cómo funciona, sus características, entre otros aspectos así como el impacto que tiene su aplicación reciente en *deep learning* (como en Large Language Models o LLMs), se presenta una vista general de los 3 componentes principales que lo integran:

* `Conformal Prediction` o Predicción Conforme.
* `Transformers`
* `Mixture of Experts` o Mezcla de Expertos.

### ¿Qué es `Conformal Prediction`?

Conformal Prediction (CP) es un marco estadístico novedoso que proporciona una forma robusta y fundamental para cuantificar la incertidumbre en las predicciones de cualquier algoritmo, transformando predicciones puntuales de cualquier modelo en conjuntos de predicción con sólidas propiedades de cobertura. A diferencia de los métodos clásicos que a menudo dependen de supuestos asintóticos o distribucionales (e.g., normalidad de los residuos), CP ofrece un enfoque no paramétrico y agnóstico al modelo. Su validez teórica se fundamenta en un único supuesto relativamente débil: la intercambiabilidad de los datos (Vovk, Gammerman, & Shafer, 2005).

El supuesto de intercambiabilidad estipula que la probabilidad conjunta de una secuencia de variables aleatorias $(Z_{1}, Z_{2},..., Z_{n})$ es invariante bajo cualquier permutación de sus índices, donde $Z_{i} = (Z_{i}, Y_{i})$. Formalmente, para cualquier permutación $\pi$ de {$1,...,n$}, se tiene que:

$$P(Z_{1},..., Z_{n}) = P(Z_{\pi(1)},..., Z_{\pi(n)})$$

Este supuesto es más débil que el de datos independientemente e idénticamente distribuidos (i.i.d.), aunque en la práctica, los datos i.i.d. son el caso de uso más común que satisface la intercambiabilidad.

#### Objetivo Principal de la `Conformal Prediction`: Cobertura Marginal Garantizada

El objetivo central de Conformal Prediction no es la precisión de la predicción puntual, sino la construcción de un conjunto de predicción $C(X_{n+1})$ para una nueva observación $X_{n+1}$ que contenga el verdadero valor $Y_{n+1}$ con una probabilidad predefinida y controlada por el usuario. Específicamente, para un nivel de significancia $\alpha\in(0,1)$ elegido a priori, el conjunto de predicción debe satisfacer la siguiente propiedad de cobertura marginal:

$$
P(Y_{n+1} \in C(X_{n+1})) \geq (1 - \alpha)
$$

Esta garantía es válida para cualquier tamaño de muestra n, para cualquier distribución de datos que cumpla con la intercambiabilidad y para cualquier algoritmo de predicción subyacente. Esta robustez es una de las características más atractivas del marco (Angelopoulos & Bates, 2023).

#### Mecanismo Operativo: Puntuaciones de No Conformidad y Cuantiles Empíricos

El mecanismo de CP se articula en torno a una función de puntuación de no conformidad (Non-Conformity Measure, NCM). Esta función, denotada como $s(x,y)$, asigna una puntuación numérica a cada par $(x,y)$ que cuantifica qué tan "atípico" o "disconforme" es el valor $y$ para las covariables $x$, según un modelo de predicción subyacente. Una elección común en regresión es el residuo absoluto:

$$
s(x,y) = |y - \widehat{\mu}(x)|
$$

donde $\widehat{\mu}(x)$ es la predicción puntual de un modelo entrenado.

Para mitigar el sobreajuste y mantener la validez teórica, el procedimiento más extendido y práctico es el *Split Conformal Prediction* o *Inductive Conformal Prediction* (Lei et al., 2018; Papadopoulos, 2008). El proceso es el siguiente:

$$
    S_{cal} = \{ s_{i} = s(X_{i}, Y_{i}) | (X_{i}, Y_{i}) \in {\mathcal{D}_{cal}\}} \\
$$

1. División de Datos (Data Splitting): El conjunto de datos de entrenamiento $\mathcal{D} = {\{(X_{i}, Y_{i})\}_{i=1}^{n}}$ se divide aleatoriamente en dos subconjuntos disjuntos: un conjunto de entrenamiento propiamente dicho $\mathcal{D}_{train}$ (con $m$ puntos), y un conjunto de calibración $\mathcal{D}_{cal}$ con $n_{cal} = n - m$ puntos.

2. Entrenamiento del Modelo: Se entrena cualquier modelo de predicción (ejemplo RandomForest, Red Neuronal, Gradient Boosting) utilizando únicamente $\mathcal{D}_{train}$.

3. Cálculo de Puntuaciones de No Conformidad: Utilizando el modelo fijo $\widehat{\mu}(x)$ se calculan las puntuaciones de no conformidad para cada punto en el conjunto de calibración $\mathcal{D}_{cal}$:

    $$
        S_{cal} = \{ s_{i} = s(X_{i}, Y_{i}) | (X_{i}, Y_{i}) \in \mathcal{D}_{cal}\}
    $$

    Estos $n_{cal}$ scores forman una distribución empírica de "errores" o "disconformidades" observados en datos no vistos durante el entrenamiento.

4. Determinación del Umbral (Cuantil): Se calcula el umbral $\widehat{q}$ como el cuantil empírico de las puntuaciones de calibración $S_{cal}$ que asegura la cobertura deseada. Específicamente, $\widehat{q}$ se define como el $[(1 - \alpha)(n_{cal} + 1)]$-ésimo valor más pequeño en el conjunto $S_{cal}$. El ajuste $(n_{cal} + 1)$ es una corrección para muestras finitas que garantiza la desigualdad de cobertura (Angelopoulos & Bates, 2023).

5. Construcción del Conjunto de Predicción: Para una nueva instancia $X_{new}$, el conjunto de predicción $C(X_{new})$ se construye invirtiendo la función de puntuación. Se define como el conjunto de todos los posibles valores $y \in \mathcal{Y}$ para los cuales la puntuación de no conformidad no excede el umbral $\widehat{q}$:

    $$
        C(X_{new}) = \{y \in \mathcal{Y} \; | \; s(X_{new}, y) \leq \widehat{q}\}
    $$

    Para el caso de la *Puntuación de No Conformidad* de residuo absoluto, esto resulta en un intervalo de predicción simétrico:

    $$
        C(X_{new}) = [\widehat{\mu}(X_{new}) - \widehat{q}\text{,} \quad \widehat{\mu}(X_{new}) + \widehat{q}]
    $$

#### Propiedades Fundamentales

1. Validez (Garantía de Cobertura): Como se mencionó, la propiedad principal es la cobertura marginal $P(Y_{n+1} \in C(X_{n+1})) \geq (1 - \alpha)$, que se mantiene bajo el único supuesto de intercambiabilidad.
2. Agnosticismo del Modelo: CP es una "envoltura" (wrapper) que puede aplicarse sobre cualquier algoritmo preexistente (k-NN, SVM, árboles de decisión, redes neuronales, regresión ridge o clasificadores probabilísticos) para producir conjuntos de predicción calibrados. La calidad del modelo subyacente no afecta la validez de la cobertura, pero sí su eficiencia.
3. Eficiencia: La eficiencia se refiere al tamaño (e.g., la longitud en regresión, la cardinalidad en clasificación) de los conjuntos de predicción. Un modelo $\widehat{\mu}$ más preciso generará puntuaciones de no conformidad más bajas en $\mathcal{D}_{cal}$, lo que resultará en un umbral $\widehat{\mu}$ más pequeño y, por ende, en conjuntos de predicción más informativos (más pequeños), manteniendo la misma garantía de cobertura (Lei et al., 2018).

### ¿Qué son los `Transformers`?

En el panorama actual del aprendizaje profundo, la arquitectura Transformer ha surgido como un pilar fundamental, redefiniendo el estado del arte en el procesamiento de datos secuenciales y extendiendo su influencia a dominios que trascienden el lenguaje natural. En el ámbito de la estadística, comprender la formalización y las propiedades inherentes de este modelo es crucial, no solo para su aplicación sino también para su potencial extensión y mejora. A continuacieon se desglosan los fundamentos, objetivos, propiedades y de la arquitectura Transformer, y explora su relación sinérgica con el paradigma de *Mixture of Experts* (MoE).

La arquitectura *Transformer*, introducida por Vaswani et al. (2017) en su influyente artículo "*Attention Is All You Need*", representa un cambio de paradigma con respecto a las arquitecturas de redes neuronale recurrentes (RNN) y convolucionales (CNN) que dominaban el procesamiento de secuencias. El objetivo principal del *Transformer* es capturar dependencias a larga distancia en los datos sin la necesidad de un procesamiento secuencial inherente, lo que a su vez permite una paralelización masiva y una eficiencia computacional sin precedentes.

El concepto fundamental que sustenta al *Transformer* es el mecanismo de auto-atención (self-attention). A diferencia de las RNN que procesan la información de manera secuencial, manteniendo un estado oculto que evoluciona en el tiempo, la auto-atención permite al modelo ponderar la importancia de todas las demás palabras en la secuencia al procesar una palabra específica. Esto se logra calculando una representación de cada elemento de la secuencia que es una suma ponderada de todos los demás elementos, donde los pesos son determinados dinámicamente.

#### Objetivos y Propiedades

Desde una perspectiva estadística, los objetivos y propiedades de la arquitectura Transformer son de gran interés:

* Modelado de Dependencias a Larga Distancia: El principal objetivo es superar la dificultad de las RNN para mantener información a través de secuencias largas, un problema conocido como el desvanecimiento del gradiente. La auto-atención, al conectar directamente cada par de elementos de la secuencia, reduce la longitud máxima de la ruta de la señal a O(1), en contraste con O(n) en las RNN, donde n es la longitud de la secuencia.

* Paralelización y Escalabilidad: Al eliminar la recurrencia, los cálculos dentro de cada capa del Transformer pueden ser masivamente paralelos. Esta propiedad ha sido fundamental para entrenar modelos de un tamaño sin precedentes en vastos conjuntos de datos, una tendencia que ha dado lugar a los modelos de lenguaje a gran escala (LLMs).

* Invariancia a la Permutación y la Necesidad de Codificación Posicional: El mecanismo de auto-atención es, por diseño, invariante a la permutación del orden de la secuencia de entrada. Para que el modelo pueda utilizar el orden de la secuencia, es necesario inyectar información sobre la posición de los elementos. Esto se logra mediante la codificación posicional (positional encoding), que son vectores que se suman a las representaciones de entrada (embeddings) para proporcionar al modelo información sobre la posición relativa o absoluta de los tokens.


#### Mecanismos Operativos

La arquitectura Transformer se compone típicamente de un codificador (encoder) y un decodificador (decoder), cada uno de los cuales es una pila de capas idénticas. Cada capa, a su vez, está compuesta por sub-capas: un mecanismo de atención multi-cabeza y una red neuronal de avance (feed-forward) completamente conectada.

Auto-Atención Escalada por Producto Punto (Scaled Dot-Product Self-Attention)

El mecanismo de atención se puede describir como una función que mapea una consulta y un conjunto de pares clave-valor a una salida. La salida se calcula como una suma ponderada de los valores, donde el peso asignado a cada valor es calculado por una función de compatibilidad de la consulta con la clave correspondiente. En el Transformer, se utiliza la atención de producto punto escalada.

Para una secuencia de entrada, se crean tres matrices: la de Consulta (Query, Q), la de Clave (Key, K) y la de Valor (Value, V), que se obtienen al multiplicar la matriz de embeddings de entrada por tres matrices de pesos aprendibles $W^{Q}, W^{K} \text{ y } W^{V}$. La salida de la atención se calcula como:

$$
    Attention(Q, K, V) = softmax \left( \frac{QK^{T}}{\sqrt{d_{k}}}\right)*V
$$

donde $d_{k}$ es la dimensión de los vectores de clave y consulta. La escala por $\sqrt{d_{k}}$ es un factor crucial que previene que los productos punto crezcan demasiado en magnitud, lo que podría llevar a gradientes extremadamente pequeños en la función softmax.

#### Relación de los Transformers con Mixture of Experts (MoE)

A medida que los modelos Transformer han crecido en tamaño, el costo computacional de entrenarlos y servirlos se ha vuelto prohibitivo. El paradigma de Mixture of Experts (MoE) ofrece una solución para escalar la capacidad del modelo sin un aumento proporcional en el costo computacional.

El concepto central de MoE es reemplazar componentes densos del modelo, como las redes de avance (feed-forward), por múltiples "expertos" (redes neuronales más pequeñas) y una red de compuerta (gating network) que aprende a enrutar cada token de la secuencia de entrada a un pequeño subconjunto de estos expertos. De esta manera, para cada entrada, solo se activa una fracción del modelo total, lo que resulta en una activación dispersa.


### ¿Qué es `Mixture of Experts`?

*Mixture of Experts* (MoE) puede ser interpretado no solo como una técnica de *ensemble learning*, sino como una poderosa extensión de los modelos de mezcla finita y los Modelos Lineales Generalizados (GLM). Su concepción fundamental radica en una estrategia de "divide y vencerás" que particiona el espacio de los predictores de manera suave (*soft partitioning*), permitiendo que diferentes modelos expertos se especialicen en distintas regiones de dicho espacio. Esta aproximación ofrece una flexibilidad considerable para modelar procesos complejos y heterogéneos, y ha resurgido como un componente clave en la arquitectura de modelos de lenguaje a gran escala (LLMs) y otros sistemas de aprendizaje profundo.

#### Objetivos y Propiedades Estadísticas

El objetivo principal de un MoE es modelar datos que exhiben heterogeneidad en la relación entre variables predictoras y respuesta. En lugar de ajustar un único modelo global que puede resultar inadecuado, MoE ajusta múltiples modelos más simples, cada uno especializado en una región del espacio de entrada, y los combina de una manera probabilísticamente coherente.

Desde una perspectiva estadística, las propiedades de los estimadores de MoE son de sumo interés. La estimación de los parámetros se realiza comúnmente a través de Máxima Verosimilitud (Maximum Likelihood Estimation, MLE). Dada la estructura de la verosimilitud, que involucra una suma dentro del logaritmo, la optimización directa es a menudo intratable. Por ello, el algoritmo de Expectativa-Maximización (EM) es el mecanismo operativo para la inferencia en estos modelos.

#### Mecanismos Operativos: La Red de Compuertas

La red de compuertas es el componente central que distingue a los MoE. Su función es determinar la contribución de cada experto para una entrada dada. La formulación más común para la red de puertas es el modelo softmax, que es una generalización de la regresión logística para múltiples clases:

$$
    g_{i}(x,v) = \frac{exp(v_{i}^{T}x)}{\sum_{j=1}^{N} exp(v_{j}^{T}x)}
$$


donde $ν_{i}$ es el vector de parámetros asociado al experto $i$. Esta formulación produce una partición suave del espacio de entrada, donde las fronteras de decisión entre las regiones de los expertos son lineales (en el espacio de $x$).


### Visión general de la solución
* **Arquitectura central (Transformer-MoE para tabular)**
    * `SwitchLayer`: capa MoE con gating Top-k (parámetro k) inspirada en Switch Transformers. Ejecuta todos los expertos en paralelo sólo para los tokens seleccionados y agrega pérdida auxiliar de balanceo de carga (aux_loss_weight) usando la probabilidad media del router y la fracción de enrutamiento por experto para evitar el colapso de expertos.
    * `TransformerSwitchBlock`: bloque Pre-LayerNorm + MultiHeadAttention + `SwitchLayer`, con residuals y dropout.
    * Tokenizador tabular (`build_tabular_encoder`):
        * Continuas y binarias → proyección densa a d_model como tokens.
        * Categóricas → embeddings por columna (una entrada por feature con su vocab_size).
        * Inserta un token [CLS] para el pooling final.
    * `build_switch_transformer_tabular`: ensambla el encoder + N bloques Transformer-MoE y una salida sigmoide (clasificación binaria).

* **Entrenamiento y rendimiento**
    * `train.py`: pipeline con tf.data (vectorizado, cache(), prefetch()), mixed precision opcional para GPUs y métricas: AUC, Precision, Recall, F1 (métrica principal) y Accuracy.
    * **Callbacks**:
        * `EpochLogger`: guarda pesos del mejor modelo por F1.
        * `EarlyStopping` y `ReduceLROnPlateau` monitorizando F1 (modo `max`).
    * **Pérdida**: `binary_focal_loss`(gamma, alpha) para manejar desbalanceo.

* **Validación, selección de modelo y remuestreo**
    * `cross_validation.py`: StratifiedKFold con soporte de `tf.distribute.Strategy` (p.ej., `MirroredStrategy`/`MultiWorkerMirroredStrategy`).
    * **SMOTE / SMOTE-Tomek** integrados (via `imblearn`) con `sampling_strategy` flexible (proporción, entero por clase o diccionario).
    * Para cada fold: se entrena, se evalúan múltiples umbrales, se conserva el mejor F1 y se retorna la instancia entrenada del mejor modelo + el `final_results_df` consolidado.

* **Evaluación y visualización**
    * `evaluate.py`: calcula Precision, Recall, F1, Accuracy, ROC-AUC y matriz de confusión a lo largo de una rejilla de thresholds.
    * `plotting.py`: utilitario para curva ROC.

* **Conformal Prediction (binario, condicional por clase)**
    * `conformal.py`: Split Conformal con calibración condicional por clase (qhat_0, qhat_1) usando puntuaciones de no-conformidad:
    positivos: $s = 1 - p$; negativos: $s = p$
    * `evaluate_conformal_sets`: genera conjuntos de predicción $\{0,1\}$ por muestra y reporta coverage y tamaño medio de conjunto.
    * La separación por clase ayuda a mantener cobertura con clases desbalanceadas y umbrales diferenciados.

----

* Puntos fuertes (highlights)

    * **Especializado para tabular**: tokenización nativa de continuas, binarias y categóricas (embeddings por columna) + [CLS] para un readout robusto.
    * **MoE eficiente y estable**: Top-k gating + pérdida de balanceo para distribuir carga entre expertos y mejorar capacidad sin costear un modelo denso equivalente.
    * **Pipeline productivo**: tf.data vectorizado, mixed precision, y callbacks orientados a F1 → entrenamiento estable y más rápido.
    * **Desbalanceo cubierto extremo a extremo**: combinación de Focal Loss, SMOTE/SMOTE-Tomek y selección de umbral por F1.
    * **Selección de mejor modelo real**: se retorna la instancia entrenada con el mejor F1 (no sólo archivos), lista para calibrar Conformal.
    * **Conformal Prediction condicional**: cobertura controlada y trazable (coverage y tamaño de conjunto), útil para decisiones bajo riesgo.
    * **Escalabilidad**: soporte directo para estrategias distribuidas vía tf.distribute.Strategy.
    * **Modularidad y mantenibilidad**: arquitectura, entrenamiento, validación, métricas, pérdidas, callbacks y conformal separados por módulos.

----

* Stack utilizado
    * Lenguaje: Python (>=3.10 recomendado).
    * Framework DL: TensorFlow / Keras (MultiHeadAttention, Layers personalizados, `tf.data`, `mixed_precision`, `tf.distribute`).
    * ML clásico & utilidades: scikit-learn (KFold, métricas), imbalanced-learn (SMOTE, SMOTE-Tomek), pandas, numpy.
    * Visualización: matplotlib (ROC).
    * Arquitectura propia: capas personalizadas (`SwitchLayer`, `TransformerSwitchBlock`), *loss* focal y métrica F1 stateful.

----

* Flujo típico de uso
    * Construcción del modelo con `build_switch_transformer_tabular  (parámetros: d_model, num_heads, d_ff, num_experts, k, aux_loss_weight, num_layers, dropout, cat_vocab_sizes).
    * Entrenamiento/validación cruzada con `run_cross_validation`(...) pasando una Strategy y training_params (col_groups, batch_size, epochs, lr, etc.).
    * Selección de umbral y métricas con `evaluate`(...) y curva ROC con `plotting.py`.
    * Calibración conforme con `calibrate_conformal_conditional`(...) y evaluación con `evaluate_conformal_sets`(...).
    * Estado actual: la arquitectura y evaluación están orientadas a clasificación binaria (salida sigmoide). La extensión a multiclase es directa (p.ej., softmax + sets conformes multicategoría), manteniendo el mismo patrón modular.

