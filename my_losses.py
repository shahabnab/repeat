from keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf 
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

epsilon = 1e-5
smooth = 1


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.reshape(y_true, (-1,))
    y_pred_f = K.reshape(y_pred, (-1,))

    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def tp(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    return (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)

def tn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 



# Standard Tversky (positive class = 1)
def tversky(y_true, y_prob, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.cast(y_prob, tf.float32)      # y_prob must be in [0,1]
    y_true = tf.reshape(y_true, [-1])
    y_prob = tf.reshape(y_prob, [-1])

    tp = tf.reduce_sum(y_true * y_prob)
    fn = tf.reduce_sum(y_true * (1.0 - y_prob))
    fp = tf.reduce_sum((1.0 - y_true) * y_prob)

    return (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)

def tversky_loss(y_true, y_prob, alpha=0.7, beta=0.3, smooth=1e-6):
    return 1.0 - tversky(y_true, y_prob, alpha, beta, smooth)

def focal_tversky(y_true, y_prob, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    ti = tversky(y_true, y_prob, alpha, beta, smooth)
    return tf.pow(1.0 - ti, gamma)

def weighted_bce_logits(y_true, logits, w0=3.0, w1=1.0):
    y = tf.cast(y_true, tf.float32)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    w = y*w1 + (1.0 - y)*w0    # LOS (0) gets w0
    return tf.reduce_mean(w * ce)



# --- KL(p_teacher || p_student) for *binary* head using logits (preferred)
def kl_consistency_from_logits(student_logits, teacher_logits, T=1.0, weight=None):
    t = tf.cast(T, tf.float32)
    # build 2-class logits [0, z] to reuse stable softmax/log_softmax
    s2 = tf.concat([tf.zeros_like(student_logits), student_logits], axis=-1) / t
    q2 = tf.concat([tf.zeros_like(teacher_logits), tf.stop_gradient(teacher_logits)], axis=-1) / t
    q  = tf.nn.softmax(q2, axis=-1)               # teacher probs (stop-grad)
    logp = tf.nn.log_softmax(s2, axis=-1)         # student log-probs
    kl_per = tf.reduce_sum(q * (tf.math.log(tf.clip_by_value(q,1e-8,1.0)) - logp), axis=-1)
    if weight is not None:
        kl = tf.reduce_sum(weight * kl_per) / (tf.reduce_sum(weight) + 1e-8)
    else:
        kl = tf.reduce_mean(kl_per)
    return (t*t) * kl   # standard T^2 scaling




def bernoulli_entropy_from_logits(logits, base='e'):
        """Mean Bernoulli entropy from *logits* (scalar per example)."""
        x = tf.convert_to_tensor(logits, tf.float32)
        p = tf.nn.sigmoid(x)
        eps = tf.constant(1e-8, tf.float32)
        H = -(p * tf.math.log(p + eps) + (1.0 - p) * tf.math.log(1.0 - p + eps))
        Hm = tf.reduce_mean(H)
        return Hm / tf.math.log(tf.constant(2.0, tf.float32)) if base == 2 else Hm



def entropy_on_adaption_logits(los_logits_model, X2d, base='e'):
        if X2d.ndim == 2:
            X3d = X2d[..., None]   # shape (N,150,1)
        else:
            X3d = X2d
        logits = los_logits_model.predict(X3d, verbose=0).reshape(-1)

        return float(bernoulli_entropy_from_logits(logits, base=base).numpy())

    
          



def mean_entropy(logits, *, base='e'):
   
    x = tf.convert_to_tensor(logits, tf.float32)
    x = tf.reshape(x, (-1, 1))  # ensure (N,1)

    # Build 2-class logits [0, z] and compute softmax probs
    two_class = tf.concat([tf.zeros_like(x), x], axis=-1)  # (N,2)
    p = tf.nn.softmax(two_class, axis=-1)

    # PyTorch-style entropy: -mean(sum p * log(p + 1e-8))
    eps = tf.constant(1e-8, tf.float32)
    H = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)  # (N,)
    Hm = tf.reduce_mean(H)                                 # scalar

    if base == 2:
        Hm = Hm / tf.math.log(tf.constant(2.0, tf.float32))

    return float(Hm.numpy())


    
def binary_entropy(p, eps=1e-8):
    # Cast safely even if p is already a Tensor
    p = tf.cast(p, tf.float32)
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    ent = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))
    return float(tf.reduce_mean(ent).numpy())



def safe_probs(p, eps=1e-6):
    p = tf.cast(p, tf.float32)
    return tf.clip_by_value(p, eps, 1.0 - eps)

def safe_entropy_from_probs(p, eps=1e-6):
    p = tf.clip_by_value(tf.cast(p, tf.float32), eps, 1.0 - eps)
    return -tf.reduce_sum(p * tf.math.log(p), axis=-1)  # per-sample
def _weighted_mean(per_sample, sw):
    per_sample = tf.reshape(tf.cast(per_sample, tf.float32), (-1,))
    sw = tf.reshape(tf.cast(sw, tf.float32), (-1,))
    return tf.reduce_sum(per_sample * sw) / (tf.reduce_sum(sw) + 1e-8)

 # ======================================================================
    # Helpers (DRY, precise semantics: probs vs logits)
    # ======================================================================
def _ensure_3d(X):
    return X[..., None] if X.ndim == 2 else X

def predict_probs(model_prob_head, X2d, weights=None):
    """Predict P(class=1). If weights given, filter where w!=0."""
    X3d = _ensure_3d(X2d)
    if weights is not None:
        X3d = X3d[weights != 0]
    return model_prob_head.predict(X3d, verbose=0).reshape(-1)

def predict_logits(model_logit_head, X2d, weights=None):
    """Predict logits for class=1. If weights given, filter where w!=0."""
    X3d = _ensure_3d(X2d)
    if weights is not None:
        X3d = X3d[weights != 0]
    return model_logit_head.predict(X3d, verbose=0).reshape(-1)

def f1_on_train12(CIRS, Labels, model_prob_head,h, weights=None):
    X_tr12 = np.concatenate([CIRS["TRAIN1"], CIRS["TRAIN2"]], axis=0)
    y_tr12 = np.concatenate([np.asarray(Labels["TRAIN1"]), np.asarray(Labels["TRAIN2"])]).astype(int)
    if weights is not None:
        w_tr12 = np.concatenate([np.asarray(weights["TRAIN1"]), np.asarray(weights["TRAIN2"])]).astype(float)
        mask = (w_tr12 != 0)
        X_tr12 = X_tr12[mask]
        y_tr12 = y_tr12[mask]

    probs = predict_probs(model_prob_head, X_tr12)
    thr = float(h["METRIC_THRESHOLD"]) 
    preds = (probs >= thr).astype(int)
    return float(f1_score(y_tr12, preds)), float(thr)




def f1_minus_entropy(CIRS, Labels, model_prob_head,h, model_logit_head,
                     entropy_base='e', weights=None):
    
    f1,thr = f1_on_train12(CIRS, Labels, model_prob_head,h, weights=weights)
    ent = entropy_on_adaption_logits(model_logit_head, CIRS["ADAPTION"], base=entropy_base)

    return {"f1_train12": f1, "entropy_adaption": ent,"threshold_used":thr, "score_f1_minus_entropy": f1 - ent}
# ======================================================================
    # Closed-form scores for reporting (keep names clear; probs vs logits)
    # ======================================================================
def _binary_crossentropy_mean(y_true, p, eps=1e-7):
        p = np.clip(p, eps, 1. - eps)
        ce = -(y_true*np.log(p) + (1. - y_true)*np.log(1. - p))
        return float(np.mean(ce))

def _binary_focal_mean(y_true, p, gamma=2.0, alpha=0.25, eps=1e-7):
        

        p = np.clip(p, eps, 1. - eps)
        pt = np.where(y_true == 1, p, 1. - p)
        alpha_t = np.where(y_true == 1, alpha, 1. - alpha)
        loss = -alpha_t * ((1. - pt) ** gamma) * np.log(pt)
        return float(np.mean(loss))
from tensorflow.keras.losses import MeanSquaredError
mse = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
def reconstruction_loss(y_true, y_pred):
        """ raw = time_freq_log_loss(y_true, y_pred)        # (batch,) or (batch,features)
        raw=mse(y_true, y_pred)
   
        return (tf.reduce_mean(raw))/14.0 """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        #return 5.0*tf.reduce_mean(tf.square(y_true - y_pred))
        return tf.reduce_mean(tf.square(y_true - y_pred))



def domain_loss(y_true, y_pred, sw,h):
    ce_dom = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=h["DOM_LABEL_SMOOTH"], reduction='none')
    per_sample = ce_dom(y_true, y_pred)             # (batch,)
          
    return _weighted_mean(per_sample, sw)

def domain_loss_without_cdane(y_true, y_pred,h):
    ce_dom = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=h["DOM_LABEL_SMOOTH"], reduction='none')
    per_sample = ce_dom(y_true, y_pred)             # (batch,)  
    return tf.reduce_mean(per_sample)


def los_loss(y_true, y_pred, sw, BFC_mean):
    sw = tf.reshape(tf.cast(sw, tf.float32), (-1,))
    mask = tf.not_equal(sw, 0.0)
    # If no supervised samples in this batch, return 0.0 (no gradient)
    if not tf.reduce_any(mask):
        return tf.constant(0.0, tf.float32)

    y_flat = tf.boolean_mask(y_true, mask)
    p_flat = tf.boolean_mask(y_pred, mask)
    p_flat = safe_probs(p_flat)
    #return (BFC_mean(y_flat, p_flat))*10.0
    return (BFC_mean(y_flat, p_flat))



