import tensorflow as tf
from keras import ops

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