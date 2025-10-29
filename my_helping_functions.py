
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.losses import BinaryFocalCrossentropy

def set_seed(seed: int, *, enable_tf_op_determinism: bool = True, verbose: bool = True) -> None:
    """
    Reproducible runs on GPU without disabling parallelism.

    What it does:
      • Seeds Python, NumPy, and TensorFlow RNGs.
      • (Optionally) enables deterministic TF ops (cuDNN) when available.
      • DOES NOT force thread counts to 1 (keeps performance).

    Tips (do these in your entry script BEFORE importing TensorFlow):
      os.environ["TF_DETERMINISTIC_OPS"] = "1"
      os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
      # optional:
      # os.environ["PYTHONHASHSEED"] = str(seed)  # only effective at process start

    For tf.data pipelines, keep order deterministic:
      ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=False)
      ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    """
    import random
    import numpy as np
    import tensorflow as tf

    # RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prefer TF's built-in deterministic switch (TF ≥ 2.12)
    if enable_tf_op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # older TF versions: rely on the env vars set before import
            pass

    # Do NOT force single-thread execution — keep resources available.
    # If you ever want capped-but-parallel threads, set fixed numbers, e.g.:
    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(2)

    if verbose:
        print(f"[seed] {seed}  | TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")



def take_labels(datasets):
    datasets_roles = ["TRAIN1","TRAIN2","TRAIN3","ADAPTION","TEST"]


    df_tr1  = datasets[datasets_roles[0]]
    df_tr2  = datasets[datasets_roles[1]]
    df_tr3=datasets[datasets_roles[2]]
    df_dt   = datasets[datasets_roles[3]]
    df_test = datasets[datasets_roles[4]]

    tr1_lbl = df_tr1["Label"].astype(np.int32)
    tr2_lbl = df_tr2["Label"].astype(np.int32)
    tr3_lbl = df_tr3["Label"].astype(np.int32)
    ad_lbl = df_dt["Label"].astype(np.int32)
    test_lb = df_test["Label"].astype(np.int32)
    Labels = {
        "role": [],
        "data": []
    }
    for name, data in zip(datasets_roles, [tr1_lbl, tr2_lbl,tr3_lbl, ad_lbl, test_lb]):
        Labels["role"].append(name)
        Labels["data"].append(data)

    return  dict(zip(Labels["role"], Labels["data"]))

def take_domains(datasets):
    Domains={
        "role": [],
        "data": []
    }
    datasets_roles = ["TRAIN1","TRAIN2","TRAIN3","ADAPTION","TEST"]
    dom_tr1  = np.zeros(len(datasets[datasets_roles[0]]))
    dom_tr2  = np.ones(len(datasets[datasets_roles[1]]))
    dom_tr3  = np.full(len(datasets[datasets_roles[2]]), 2, dtype=np.int32)
    dom_adap   = np.full(len(datasets[datasets_roles[3]]), 3, dtype=np.int32)
    dom_test = np.full(len(datasets[datasets_roles[4]]), 3, dtype=np.int32)
    for name, data in zip(datasets_roles, [dom_tr1, dom_tr2,dom_tr3, dom_adap, dom_test]):
        Domains["role"].append(name)
        Domains["data"].append(data)
    return dict(zip(Domains["role"], Domains["data"]))

def take_weights(datasets,adapt_size=0):
    datasets_roles = ["TRAIN1","TRAIN2","TRAIN3","ADAPTION","TEST"]
    w_tr1  = np.ones(len(datasets[datasets_roles[0]]))
    w_tr2  = np.ones(len(datasets[datasets_roles[1]]))
    w_tr3  = np.ones(len(datasets[datasets_roles[2]]))
    w_adap   = np.zeros(len(datasets[datasets_roles[3]]))
    w_test = np.ones(len(datasets[datasets_roles[4]]))
    weights={
        "role": [],
        "data": []
    }
    if adapt_size > 0:
        idx0=np.where(datasets["ADAPTION"]["Label"]==0)[0]
        idx1=np.where(datasets["ADAPTION"]["Label"]==1)[0]
        chosen0 = np.random.choice(idx0, size=adapt_size, replace=False)
        chosen1 = np.random.choice(idx1, size=adapt_size, replace=False)
        w_adap[chosen0] = 1
        w_adap[chosen1] = 1


    for name, data in zip(datasets_roles, [w_tr1, w_tr2,w_tr3, w_adap, w_test]):
        weights["role"].append(name)
        weights["data"].append(data)
    return dict(zip(weights["role"], weights["data"]))

def train_valid_split(N,train_ratio,random_seed):
    np.random.seed(random_seed)
    indices = np.arange(N)
    np.random.shuffle(indices)
    train_size = int(train_ratio * N)  # 80% for training
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    return train_indices, valid_indices

def print_distribution(data_dis):
    unique, counts = np.unique(data_dis, return_counts=True)
    for val, count in zip(unique, counts):
            print(f"{val}: {count}")

def save_to_excel(config, excel_path):
    



    df_cfg = pd.DataFrame([config])
    save_path=excel_path/"report.xlsx"
    df_cfg.to_excel(save_path, index=False)
    print(f"Configuration written to {save_path}")


def predict_los_only(
        
    CIRS,
    labels,
    lb_rule: str,
    Los_model,
    h: dict,
    WEIGHTS=None,
    
    
  
):
    """
    Inference-only evaluation for a binary LOS head.
    - Does NOT require model.compile().
    - Uses `.predict()` and computes metrics via sklearn.
    - If `loss_fn` is provided, returns a scalar loss as well.
    """
    threshold = h.get("METRIC_THRESHOLD")
    # 1) Prepare arrays
    X = np.asarray(CIRS[lb_rule], dtype=np.float32)
    if X.ndim == 2:
        X = X[..., None]  # (N, T) -> (N, T, 1) channel dim
    y = np.asarray(labels[lb_rule], dtype=np.float32).reshape(-1)

    # 2) Apply mask from weights (skip ADAPTION if that's your rule)
    if WEIGHTS is not None and lb_rule in WEIGHTS and lb_rule != "ADAPTION":
        w = np.asarray(WEIGHTS[lb_rule], dtype=np.float32).reshape(-1)
        mask = (w != 0.0)
    else:
        mask = np.ones_like(y, dtype=bool)

    X_f = X[mask]
    y_f = y[mask].astype(int)

    # 3) Predict probabilities
    probs = Los_model.predict(X_f, verbose=0).reshape(-1).astype(np.float32)

    # 4) Thresholded predictions
    preds = (probs >= float(threshold)).astype(int)

    # 5) Metrics
    acc = accuracy_score(y_f, preds)
    f1  = f1_score(y_f, preds)
    cm  = confusion_matrix(y_f, preds)

    out = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "confusion_matrix": cm,
    }
    BFC_mean = BinaryFocalCrossentropy(gamma=h["FOCAL_GAMMA"],from_logits=False)  
    # 6) Optional scalar loss (keeps model uncompiled)
    y_tf = tf.convert_to_tensor(y_f, dtype=tf.float32)
    p_tf = tf.convert_to_tensor(probs, dtype=tf.float32)
    loss_val = BFC_mean(y_tf, p_tf)
    out["loss"] = float(loss_val)
    out["probs"] = probs
    return out

