

# Conclusiones

### Un Contraste entre Rendimiento Empírico y Fiabilidad Garantizada

El análisis comparativo exhaustivo entre los modelos de clasificación clásicos (tales como `XGBoost`, `RandomForest` y `MLP`) y el marco de `MoE-Conformal Prediction` (MoE-CP) revela una distinción fundamental en sus objetivos, capacidades y dominios de aplicabilidad. Las conclusiones de este artículo no señalan a un ganador absoluto, sino que delinean dos paradigmas distintos de la predicción estadística, cada uno con fortalezas inherentes para propósitos específicos.

### Maximización del Rendimiento Puntual vs. Garantía de Fiabilidad a Largo Plazo

Los modelos clásicos, con `XGBoost` a la cabeza, demostraron una superioridad empírica en métricas de rendimiento puntual como la *Accuracy* (97.2%) y el *F1-Score* (0.803). Estos algoritmos están diseñados para minimizar una función de pérdida sobre el conjunto de entrenamiento, resultando en modelos altamente optimizados para la distribución de datos específica con la que fueron entrenados. Sin embargo, este alto rendimiento carece de garantías formales sobre su comportamiento en datos futuros no observados.

En contraste, el enfoque `MoE-Conformal Prediction` no tiene como objetivo principal superar la exactitud de los modelos clásicos, sino establecer un contrato de fiabilidad con el usuario. Al alcanzar una cobertura empírica (~89.7%) que se alinea rigurosamente con el nivel de confianza predefinido del 90% $(1 - \alpha)$,  el método MoE-CP valida su principal fortaleza: la capacidad de proporcionar un control de error marginal que es robusto, no paramétrico y libre de supuestos distribucionales. La conclusión es clara: se sacrifica un potencial de rendimiento máximo a cambio de una garantía estadística de fiabilidad, un **compromiso indispensable en aplicaciones de alto riesgo**.

### Cuantificación Implícita vs. Explícita de la Incertidumbre

Una de las conclusiones más relevantes de este estudio es el tratamiento diferencial de la incertidumbre. Los modelos clásicos la manejan de forma implícita a través de *scores* de probabilidad, los cuales, sin una calibración adecuada, no son necesariamente interpretables ni fiables. Estos modelos operan bajo un paradigma de "elección forzada", emitiendo siempre una predicción puntual incluso en casos de alta ambigüedad.

El marco MoE-Conformal Prediction representa un cambio de paradigma hacia la cuantificación explícita y accionable de la incertidumbre. La incertidumbre no es un simple *score*, sino que se materializa en el tamaño y la composición del conjunto de predicción. El análisis demostró la capacidad del sistema para:

* Abstenerse de predecir cuando la confianza no es suficiente, generando conjuntos vacíos $\phi$
* Señalar ambigüedad entre clases plausibles, produciendo conjuntos con múltiples etiquetas (e.g., {0,1}).

Esta capacidad para comunicar de forma honesta cuándo una predicción es fiable y cuándo no lo es, constituye su ventaja más significativa. El análisis de sensibilidad entre calibraciones reveló además que, si bien los umbrales son robustos, el modo de expresar la incertidumbre (abstención vs. ambigüedad) puede variar, un hallazgo sutil de gran importancia para la implementación práctica.

### Eficiencia y Aplicabilidad en Dominios Críticos

A pesar de ofrecer garantías rigurosas, el predictor conformal demostró ser altamente eficiente. Con un tamaño de conjunto promedio cercano a 1, el MoE-CP fue tan decisivo como un clasificador clásico en la gran mayoría de los casos. Esto desmitifica la idea de que la fiabilidad garantizada debe pagarse con predicciones sistemáticamente vagas.

En conclusión, la elección metodológica depende intrínsecamente del dominio de aplicación. Para tareas de bajo riesgo o competiciones académicas donde se persigue la optimización de métricas sobre un conjunto de datos estático, la potencia empírica de modelos como XGBoost es innegable. Sin embargo, para el despliegue en sistemas críticos —diagnóstico médico asistido por ordenador, evaluación de riesgo crediticio, jurisprudencia predictiva o sistemas autónomos— donde las consecuencias de un error silencioso son graves, el marco MoE-Conformal Prediction es metodológicamente superior. No solo proporciona una predicción, sino que la acompaña de un nivel de confianza cuantificable y garantizado, representando un paso crucial hacia una inteligencia artificial más responsable, fiable y transparente.