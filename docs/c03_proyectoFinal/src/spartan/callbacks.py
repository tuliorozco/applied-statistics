import numpy as np
import tensorflow as tf
from tensorflow import keras
# from keras import layers, ops
from keras.callbacks import Callback

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