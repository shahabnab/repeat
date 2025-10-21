import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras import backend as K
import math
import tensorflow as tf
from tensorflow import keras


# ── 1) a GRL layer that holds a mutable `hp_lambda` variable ─────────────────
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, name=None, initial_lambda=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        # this is the λ we’ll schedule
        self.hp_lambda = self.add_weight(
            name="lambda",
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(initial_lambda),
            trainable=False,
        )

    def call(self, x):
        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                lam_cast = tf.cast(self.hp_lambda, dy.dtype)
                return -lam_cast * dy
            return x, grad
        return _reverse(x)
    



def build_encoder(seq_len: int,const_coef: float, latent_dim: int) -> models.Model:
            inp = layers.Input((seq_len ,1), name="input_signal")
            


            x = layers.Conv1D(int(32*const_coef) , 3, padding='same', name="c1")(inp)
            x = layers.BatchNormalization(name="bn1")(x)
            x = layers.ReLU(name="r1")(x)

            x = layers.Conv1D(int(64*const_coef), 3, strides=2, padding='same', name="c2")(x)
            x = layers.BatchNormalization(name="bn2")(x)
            x = layers.ReLU(name="r2")(x)

            x = layers.Conv1D(int(128*const_coef), 3, strides=3, padding='same', name="c3")(x)
            x = layers.BatchNormalization(name="bn3")(x)
            x = layers.ReLU(name="r3")(x)

            x = layers.GlobalAveragePooling1D(name="gap")(x)
            latent = layers.Dense(latent_dim, activation="linear", name="latent_vector")(x)
            model= models.Model(inp, latent, name="Encoder")

            return model




def build_decoder(latent_dim: int, seq_len: int, const_coef: float):
    inp = layers.Input((latent_dim,), name="latent_input")

    # Use ceil to mirror encoder’s rounding: 150 -> 38
    seed_len = int(math.ceil(seq_len / 6))

    x = layers.Dense(seed_len * 64, activation="relu", name="dec_dense_expand")(inp)
    x = layers.Reshape((seed_len, 64), name="dec_reshape")(x)

    x = layers.UpSampling1D(2, name="dec_upsample1")(x)                    # 38 -> 76
    x = layers.Conv1D(int(64*const_coef), 3, padding='same', name="dec_conv1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.ReLU(name="dec_r1")(x)

    x = layers.UpSampling1D(3, name="dec_upsample2")(x)                    # 76 -> 152
    x = layers.Conv1D(int(32*const_coef), 3, padding='same', name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.ReLU(name="dec_r2")(x)

    
    x = layers.Conv1D(1, 3, padding='same', activation='sigmoid',
                      name="dec_output_signal")(x)                         # (B, 152, 1)

   

    return models.Model(inp, x, name="reconstruction")

def _to_two_class_probs(sigmoid_p):
    # (B,1) -> (B,2) = [p, 1-p]
    return tf.concat([sigmoid_p, 1.0 - sigmoid_p], axis=-1)

def _outer_f_g(f, g):
    # f: (B,F), g: (B,C) -> (B, F*C) via outer product
    op = tf.einsum('bf,bc->bfc', f, g)
    return tf.reshape(op, (-1, tf.shape(f)[-1] * tf.shape(g)[-1])) 
def add_cdan_domain_head(f, g_prob, n_domains, grl, hidden=128, drop=0.5, name_prefix="cdan"):
    """CDAN head: condition domain disc. on feature f and class-prob g."""
    g2 = layers.Lambda(_to_two_class_probs, name=f"{name_prefix}_to2")(g_prob) if g_prob.shape[-1] == 1 else g_prob
    cond = layers.Lambda(lambda t: _outer_f_g(t[0], t[1]), name=f"{name_prefix}_outer")([f, g2])

    h = grl(cond)
    h = layers.Dense(hidden, activation="relu", name=f"{name_prefix}_h1")(h)
    #h = layers.Dropout(drop, name=f"{name_prefix}_drop")(h)
    dom_logits = layers.Dense(n_domains, activation="softmax", name="domain_logits")(h)
    return dom_logits

def build_DA_AE(seq_len: int, n_domains: int, latent_dim: int, ENC_CONST_COEF: float, DEC_CONST_COEF: float):
    encoder = build_encoder(seq_len, ENC_CONST_COEF, latent_dim)
    latent  = encoder.output

    decoder_model = build_decoder(latent_dim, seq_len, DEC_CONST_COEF)
    recon = decoder_model(latent)

    # Task head
    latent_cls = layers.Dense(64, activation="relu", name="latent_cls")(latent)
    s = layers.Dense(16, activation="relu", name="excite_down")(latent_cls)
    s = layers.Dense(64, activation="sigmoid", name="excite_up")(s)
    gated = layers.Multiply(name="attention_gated")([latent_cls, s])

    los_logits = layers.Dense(1, name="los_logit")(gated)                 # raw logits
    los_prob   = layers.Activation("sigmoid", name="los_prob")(los_logits)  # probability

    # CDAN domain head: condition on (f=latent_cls, g=los_logits)
    grl = GradientReversal(name="grl", initial_lambda=0.0)
    dom_logits = add_cdan_domain_head(f=latent_cls, g_prob=los_prob, n_domains=n_domains, grl=grl)

    ae = models.Model(inputs=encoder.input, outputs=[recon, dom_logits, los_prob], name="CDAN_E_AE")
    return ae, grl










def model_prediction_los(model, *, cast_to_float32: bool = False, ensure_sigmoid: bool = False):
    """
    Frozen submodel that outputs LOS probabilities.

    Assumes there is a layer named 'los_prob' (ideally post-sigmoid).
    If ensure_sigmoid=True and 'los_prob' is actually logits, we apply a sigmoid here.
    """
    try:
        out = model.get_layer("los_prob").output
    except ValueError as e:
        raise ValueError(
            "Layer 'los_prob' not found on the model. "
            "Check the head name or adjust model_prediction_los()."
        ) from e

    if ensure_sigmoid:
        out = keras.layers.Activation("sigmoid", name="los_prob_sigmoid")(out)

    if cast_to_float32:
        # Keras-graph safe cast
        out = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32), name="los_prob_cast")(out)
        # or (Keras 3 ops): out = keras.layers.Lambda(lambda t: keras.ops.astype(t, "float32"))(out)

    m = keras.Model(model.input, out, name="los_only_model")
    m.trainable = False
    return m


def model_for_entropy(model, *, cast_to_float32: bool = False):
    """
    Frozen submodel that outputs LOS logits (pre-sigmoid), for entropy.
    """
    try:
        logits = model.get_layer("los_logit").output
    except ValueError as e:
        raise ValueError(
            "Layer 'los_logit' not found on the model. "
            "Expose the tensor before the sigmoid or add a logits head."
        ) from e

    if cast_to_float32:
        logits = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32), name="los_logits_cast")(logits)

    m = keras.Model(model.input, logits, name="los_logits_model")
    m.trainable = False
    return m











    





