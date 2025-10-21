from tensorflow.keras.callbacks import Callback
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os


""" class GRLSchedule(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, max_epochs, lambda_min=0.0, lambda_max=1.0,
                 peak_frac=1.0, gamma=10.0):
        super().__init__()
        self.grl_layer  = grl_layer
        self.max_epochs = int(max(1, max_epochs))
        self.lmin       = tf.convert_to_tensor(lambda_min, tf.float32)
        self.lmax       = tf.convert_to_tensor(lambda_max, tf.float32)
        self.peak_frac  = float(min(1.0, max(1e-6, peak_frac)))  # when p hits 1
        self.gamma      = tf.convert_to_tensor(gamma, tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        # progress p in [0,1]; reaches 1.0 at peak_frac * max_epochs
        p_raw = tf.cast(epoch, tf.float32) / tf.cast(self.max_epochs - 1 + 1e-7, tf.float32)
        p     = tf.minimum(1.0, p_raw / self.peak_frac)

        # DANN logistic ramp in [0,1]
        lam01 = 2.0 / (1.0 + tf.exp(-self.gamma * p)) - 1.0

        # map to [lambda_min, lambda_max]
        lam = self.lmin + (self.lmax - self.lmin) * lam01
        self.grl_layer.hp_lambda.assign(lam) """
class GRLSchedule(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, max_epochs, lambda_min=0.0, lambda_max=1.0, peak_frac=1.0):
        super().__init__()
        self.grl_layer  = grl_layer
        self.max_epochs = int(max(1, max_epochs))
        self.lmin       = tf.convert_to_tensor(lambda_min, tf.float32)
        self.lmax       = tf.convert_to_tensor(lambda_max, tf.float32)
        # reach λ_max at fraction of training (clamped to (0,1])
        self.peak_frac  = float(min(1.0, max(1e-6, peak_frac)))

    def on_epoch_begin(self, epoch, logs=None):
        p = tf.cast(epoch, tf.float32) / tf.cast(self.max_epochs - 1 + 1e-7, tf.float32)
        q = tf.minimum(1.0, p / self.peak_frac)  # ramp until peak_frac, then hold
        lam = 0.5 * (1.0 - tf.cos(tf.constant(np.pi, tf.float32) * q))
        lam = self.lmin + (self.lmax - self.lmin) * lam
        self.grl_layer.hp_lambda.assign(lam)



class WarmupThenCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, alpha=0.1):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        # cosine after warmup
        self.cosine = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=max(1, int(total_steps - int(warmup_steps))),
            alpha=alpha
        )

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        # linear warmup
        warm_lr = self.base_lr * (step_f + 1.0) / tf.maximum(1.0, self.warmup_steps)
        # cosine after warmup
        cos_lr  = self.cosine(tf.maximum(0.0, step_f - self.warmup_steps))
        return tf.where(step_f < self.warmup_steps, warm_lr, cos_lr)
def hlast(h, key, default=float('nan')):
    try:
        v = h[key][-1]
        # convert tf.Tensor / np scalar → python float
        if hasattr(v, "numpy"):   # tf.Tensor
            v = v.numpy()
        if hasattr(v, "item"):    # np scalar
            v = v.item()
        return float(v)
    except Exception:
        return default
    

def print_epoch_line_from_history(epoch, h, history):
   print(
    f"Epoch {epoch+1:03d}/{h['AE_EPOCHS']} "
    f"loss={hlast(history,'loss'):.4f} "
    f"dom_acc={hlast(history,'train_dom_acc'):.3f} "
    f"los_acc={hlast(history,'train_los_acc'):.3f} | "
    f"val_loss={hlast(history,'val_loss'):.4f} "
    f"val_dom_acc={hlast(history,'val_dom_acc'):.3f} "
    f"val_los_acc={hlast(history,'val_los_acc'):.3f} | "
    f"test_los_acc={hlast(history,'test_los_acc'):.3f}"
    
)

