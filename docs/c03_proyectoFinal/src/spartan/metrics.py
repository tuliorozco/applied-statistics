import tensorflow as tf
from tensorflow import keras
from keras import metrics

class F1ScoreMetric(metrics.Metric):
    """
    Métrica F1-Score 'con estado' (stateful) para Keras.
    
    Esta implementación es numéricamente estable y acumula los valores
    a lo largo de los lotes de una época antes de calcular el resultado final.
    """
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Usamos las métricas internas de Keras para manejar el estado
        self.precision = metrics.Precision(thresholds=threshold)
        self.recall = metrics.Recall(thresholds=threshold)

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