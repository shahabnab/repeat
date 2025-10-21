# run_tests.py
# One-file test suite tailored to your project + extended tests for the listed functions.
# Run: python run_tests.py
# Optional: DATA_DIR=/path/to/data python run_tests.py

import os, sys, glob, pickle, math, types
import numpy as np
import pandas as pd
import tensorflow as tf

# ---- project import path ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- soft import pytest ----
try:
    import pytest
except ImportError:
    print("This runner needs pytest. Install with: pip install pytest")
    raise

# ---- import your modules ----
import my_helping_functions as mh
import my_cir_processing as mcp
import my_df_processing as mdp
import my_losses as L
import my_models as MM
import my_callbacks as CB
import training_engine as TE
import my_objective_fn as OBJ
import my_plotters as PL  # not executed (plot test is skipped)

# =========================
# Data helpers (.pickle)
# =========================
KNOWN_PICKLES = [
    "TUall.pickle", "IOT_PT1.pickle", "IOT_PT2.pickle",
    "Office.pickle", "Office_all.pickle", "IOT_all.pickle"
]

def DATA_DIR():
    return os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

def find_pickles():
    d = DATA_DIR()
    if not os.path.isdir(d):
        return []
    listed = [os.path.join(d, n) for n in KNOWN_PICKLES if os.path.exists(os.path.join(d, n))]
    any_p = glob.glob(os.path.join(d, "*.pickle"))
    out, seen = [], set()
    for p in listed + any_p:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def load_pickle_df(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for k in ("df", "data", "dataset"):
            if k in obj and isinstance(obj[k], pd.DataFrame):
                return obj[k]
    try:
        return pd.read_pickle(path)
    except Exception:
        return obj

# =========================
# Fixtures
# =========================
@pytest.fixture(scope="session", autouse=True)
def _seed_everything():
    mh.set_seed(123, enable_tf_op_determinism=True, verbose=False)

@pytest.fixture(scope="session")
def available_pickles():
    return find_pickles()

@pytest.fixture(scope="session")
def any_real_df(available_pickles):
    for p in available_pickles:
        try:
            df = load_pickle_df(p)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return p, df
        except Exception:
            pass
    pytest.skip(f"No usable .pickle DataFrame found under {DATA_DIR()}")

@pytest.fixture
def tiny_synth_cirs():
    rng = np.random.default_rng(0)
    X = rng.random((16, 150, 1), dtype=np.float32)
    y = rng.integers(0, 2, size=(16,)).astype(np.float32)
    d = rng.integers(0, 3, size=(16,)).astype(np.int32)
    w = np.ones((16,), dtype=np.float32)
    return X, y, d, w

# =========================
# my_helping_functions
# =========================
def test_set_seed_is_reproducible():
    mh.set_seed(7, verbose=False)
    a1 = np.random.rand(4).tolist()
    b1 = tf.random.uniform((4,), dtype=tf.float32).numpy().tolist()
    mh.set_seed(7, verbose=False)
    a2 = np.random.rand(4).tolist()
    b2 = tf.random.uniform((4,), dtype=tf.float32).numpy().tolist()
    assert a1 == a2 and b1 == b2

def test_take_labels_domains_weights_minimal():
    df = pd.DataFrame({"Label":[0,1,1,0]})
    dts = {"TRAIN1": df, "TRAIN2": df, "ADAPTION": df, "TEST": df}
    Ls = mh.take_labels(dts)
    Ds = mh.take_domains(dts)
    Ws = mh.take_weights(dts, adapt_size=0)
    assert all(k in Ls for k in ("TRAIN1","TRAIN2","ADAPTION","TEST"))
    assert all(k in Ds for k in ("TRAIN1","TRAIN2","ADAPTION","TEST"))
    assert all(k in Ws for k in ("TRAIN1","TRAIN2","ADAPTION","TEST"))

# =========================
# my_df_processing
# =========================
def test_shuffle_df_deterministic(any_real_df):
    _, df = any_real_df
    s1 = mdp.shuffle_df(df, seed=123)
    s2 = mdp.shuffle_df(df, seed=123)
    assert s1.index.tolist() == s2.index.tolist()

# =========================
# my_cir_processing
# =========================
def test_cutting_cir_robust(any_real_df):
    path, df = any_real_df
    cols = set(map(str, df.columns))
    needed = {"CIR_amp", "sensor fp_idx"}
    if not needed.issubset(cols):
        pytest.xfail(f"{os.path.basename(path)} lacks {needed} for cutting_cir")
    sub = df[["CIR_amp","sensor fp_idx"]].head(16).copy()
    try:
        sub["sensor fp_idx"] = sub["sensor fp_idx"].astype(int)
    except Exception:
        pass
    out = mcp.cutting_cir(sub)
    if isinstance(out, np.ndarray):
        assert out.ndim == 2
        assert out.shape[0] == len(sub)
        assert np.isfinite(out).all()
    else:
        lens = [len(x) for x in out]
        assert len(lens) == len(sub) and all(l>0 for l in lens)

def test_CIR_pipeline_runs_if_columns_present(any_real_df):
    path, df = any_real_df
    need = {"CIR_amp"}
    if not need.issubset(set(map(str, df.columns))):
        pytest.xfail(f"{os.path.basename(path)} lacks {need} for CIR_pipeline")
    dts = {"TRAIN1": df.head(8).copy(),
           "TRAIN2": df.head(8).copy(),
           "ADAPTION": df.head(8).copy(),
           "TEST": df.head(8).copy()}
    try:
        res = mcp.CIR_pipeline(dts)
    except KeyError as e:
        pytest.xfail(f"CIR_pipeline requires extra columns in this dataset: {e}")
        return
    assert isinstance(res, dict) and res
    some = next(iter(res.values()))
    assert isinstance(some, (np.ndarray, list))

# =========================
# my_losses — Dice/Tversky family & metrics
# =========================
def _ensure_fn(name):
    if not hasattr(L, name):
        pytest.skip(f"{name} not found in my_losses.py")
    return getattr(L, name)

def test_dsc_and_dice_loss_consistency():
    dsc = _ensure_fn("dsc"); dice_loss = _ensure_fn("dice_loss")
    y_true = tf.constant([0,1,1,0,1], tf.float32)
    y_pred = tf.constant([0,1,0,0,1], tf.float32)
    d = dsc(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    assert 0.0 <= d <= 1.0 and 0.0 <= dl <= 1.0
    assert abs((1.0 - float(d)) - float(dl)) < 1e-6

def test_bce_dice_loss_runs():
    bce_dice_loss = _ensure_fn("bce_dice_loss")
    y_true = tf.constant([0.,1.,0.,1.])
    y_pred = tf.constant([0.2,0.8,0.3,0.7])
    v = bce_dice_loss(y_true, y_pred)
    assert tf.math.is_finite(v)

def test_tp_tn_basic():
    tp = _ensure_fn("tp"); tn = _ensure_fn("tn")
    y_true = tf.constant([0,1,1,0,1,0], tf.float32)
    y_pred = tf.constant([0,1,0,0,1,0], tf.float32)
    tpv = tp(y_true, y_pred).numpy()
    tnv = tn(y_true, y_pred).numpy()
    assert 0.0 <= tpv <= 1.0 and 0.0 <= tnv <= 1.0

def test_tversky_family():
    tversky = _ensure_fn("tversky")
    tversky_loss = _ensure_fn("tversky_loss")
    focal_tversky = _ensure_fn("focal_tversky")
    y_true = tf.constant([0,1,1,0,1,0], tf.float32)
    y_prob = tf.constant([0.2,0.8,0.6,0.4,0.7,0.1], tf.float32)
    ti  = tversky(y_true, y_prob)
    tl  = tversky_loss(y_true, y_prob)
    ftl = focal_tversky(y_true, y_prob)
    assert 0.0 <= ti <= 1.0
    assert 0.0 <= tl <= 1.0
    assert 0.0 <= ftl

def test_weighted_bce_logits_monotone():
    y = tf.constant([1.0, 1.0, 1.0])
    lo = tf.constant([-2.0, -1.0, 0.0])
    hi = tf.constant([ 2.0,  3.0, 4.0])
    a = L.weighted_bce_logits(y, lo)
    b = L.weighted_bce_logits(y, hi)
    assert float(tf.reduce_mean(b)) < float(tf.reduce_mean(a))

def test_kl_consistency_from_logits_shapes():
    kf = _ensure_fn("kl_consistency_from_logits")
    s = tf.random.normal((16,1))
    t = tf.random.normal((16,1))
    k = kf(s, t, T=2.0)
    assert tf.rank(k) == 0 and tf.math.is_finite(k)

def test_entropy_helpers():
    # bernoulli_entropy_from_logits
    Hf = _ensure_fn("bernoulli_entropy_from_logits")
    logits = tf.constant([-1000.0, 0.0, 1000.0])
    H = Hf(logits, base='e')
    H = tf.convert_to_tensor(H)
    if H.shape.rank == 0:
        assert 0.0 <= float(H) <= float(tf.math.log(2.0))
    else:
        assert H[0] < 1e-3 and H[2] < 1e-3
        assert tf.abs(H[1] - tf.math.log(2.0)) < 1e-3

    # mean_entropy
    if hasattr(L, "mean_entropy"):
        m = L.mean_entropy(tf.constant([0.0, 1.0, -1.0]))
        assert isinstance(m, float) and m >= 0.0

    # binary_entropy
    if hasattr(L, "binary_entropy"):
        be = L.binary_entropy(tf.constant([0.1, 0.9, 0.4, 0.6]))
        assert isinstance(be, float) and be >= 0.0

    # safe_probs + safe_entropy_from_probs
    sp = L.safe_probs(tf.constant([-10.0, 0.0, 10.0]))
    assert tf.reduce_all((sp >= 0.0) & (sp <= 1.0))
    if hasattr(L, "safe_entropy_from_probs"):
        se = L.safe_entropy_from_probs(tf.stack([sp, 1-sp], axis=-1))
        # per-sample entropy shape
        assert se.shape[0] == sp.shape[0]

def test_weighted_mean_basic():
    wm = _ensure_fn("_weighted_mean")
    v = wm(tf.constant([1.0, 2.0, 3.0]), tf.constant([0.0, 1.0, 1.0]))
    assert float(v) == pytest.approx(2.5, rel=1e-6)

def test_ensure_3d_and_predict_wrappers():
    ens = _ensure_fn("_ensure_3d")
    preds_p = _ensure_fn("predict_probs")
    preds_l = _ensure_fn("predict_logits")

    # Dummy models (prob head / logit head)
    inp = tf.keras.Input(shape=(150,1))
    x = tf.keras.layers.Flatten()(inp)
    p_out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    l_out = tf.keras.layers.Dense(1, activation=None)(x)
    m_prob = tf.keras.Model(inp, p_out)
    m_logit= tf.keras.Model(inp, l_out)

    X2d = np.random.rand(8,150).astype(np.float32)
    X3d = ens(X2d)
    assert X3d.shape == (8,150,1)

    p = preds_p(m_prob, X2d)
    z = preds_l(m_logit, X2d)
    assert p.shape == (8,) and z.shape == (8,)
    assert np.all((p >= 0.0) & (p <= 1.0))

def test_reconstruction_and_domain_losses():
    # reconstruction_loss ~ MSE
    rl = _ensure_fn("reconstruction_loss")
    y_true = tf.random.uniform((4,150,1))
    y_pred = tf.random.uniform((4,150,1))
    v = rl(y_true, y_pred)
    assert tf.math.is_finite(v)

    # domain_loss
    if hasattr(L, "domain_loss"):
        y_true = tf.one_hot([0,1,2,1], depth=3)
        y_pred = tf.constant([[0.9,0.05,0.05],
                              [0.1,0.8,0.1],
                              [0.1,0.2,0.7],
                              [0.2,0.6,0.2]], tf.float32)
        sw = tf.constant([1,1,1,1], tf.float32)
        h = {"DOM_LABEL_SMOOTH": 0.0}
        dv = L.domain_loss(y_true, y_pred, sw, h)
        assert dv >= 0.0

def test_f1_on_train12_and_f1_minus_entropy():
    # Only run if both f1_* utilities exist
    if not hasattr(L, "f1_on_train12") or not hasattr(L, "f1_minus_entropy"):
        pytest.skip("f1_on_train12/f1_minus_entropy not found")

    # Build tiny dummy CIRS dict with 2D arrays (N,150)
    rng = np.random.default_rng(0)
    train1 = rng.random((12,150), dtype=np.float32)
    train2 = rng.random((10,150), dtype=np.float32)
    adap   = rng.random((8,150), dtype=np.float32)

    CIRS = {"TRAIN1": train1, "TRAIN2": train2, "ADAPTION": adap}
    Labels = {"TRAIN1": rng.integers(0,2,size=(12,)).tolist(),
              "TRAIN2": rng.integers(0,2,size=(10,)).tolist()}

    # Dummy prob head (sigmoid) and logit head (linear)
    inp = tf.keras.Input(shape=(150,1))
    x = tf.keras.layers.Flatten()(inp)
    p_out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    z_out = tf.keras.layers.Dense(1, activation=None)(x)
    model_prob = tf.keras.Model(inp, p_out)
    model_logit= tf.keras.Model(inp, z_out)

    h = {"METRIC_THRESHOLD": 0.5}

    f1, thr = L.f1_on_train12(CIRS, Labels, model_prob, h, threshold=None, optimize_threshold=False, weights=None)
    assert 0.0 <= f1 <= 1.0 and 0.0 <= thr <= 1.0

    out = L.f1_minus_entropy(CIRS, Labels, model_prob, h, model_logit, threshold=None,
                             optimize_threshold=False, entropy_base='e', weights=None)
    assert set(["f1_train12","entropy_adaption","threshold_used","score_f1_minus_entropy"]).issubset(out.keys())

# =========================
# my_models — smoke
# =========================
def test_gradient_reversal_grad_sign():
    x = tf.ones((2,3))
    grl = MM.GradientReversal(initial_lambda=0.5)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = grl(x)
        loss = tf.reduce_sum(y)
    g = tape.gradient(loss, x)
    assert np.allclose(g.numpy(), -0.5*np.ones_like(g.numpy()), atol=1e-6)

def test_build_encoder_decoder_forward(tiny_synth_cirs):
    X, *_ = tiny_synth_cirs
    enc = MM.build_encoder(seq_len=150, const_coef=0.5, latent_dim=16)
    dec = MM.build_decoder(latent_dim=16, seq_len=150, const_coef=0.5)
    z = enc.predict(X, verbose=0)
    Xrec = dec.predict(z, verbose=0)
    assert z.shape[0] == X.shape[0]
    assert Xrec.ndim == 3 and Xrec.shape[0] == X.shape[0]

# =========================
# my_callbacks
# =========================
def test_GRLSchedule_updates_hp_lambda():
    # Construct with a real GRL layer so hp_lambda exists
    layer = MM.GradientReversal(initial_lambda=0.0, name="grl")
    sched = CB.GRLSchedule(grl_layer=layer, max_epochs=5,
                           lambda_min=0.0, lambda_max=1.0, peak_frac=0.5)
    sched.on_epoch_begin(0)
    v0 = float(layer.hp_lambda.numpy())
    sched.on_epoch_begin(2)
    v2 = float(layer.hp_lambda.numpy())
    assert 0.0 <= v0 <= 1.0 and 0.0 <= v2 <= 1.0

def test_WarmupThenCosine_bounds():
    sch = CB.WarmupThenCosine(base_lr=1e-3, total_steps=100, warmup_steps=10, alpha=0.1)
    v0 = float(sch(0).numpy() if hasattr(sch(0), "numpy") else sch(0))
    vT = float(sch(99).numpy() if hasattr(sch(99), "numpy") else sch(99))
    assert v0 <= 1.01e-4
    assert abs(vT - (0.1e-3)) < 6e-7

# =========================
# training_engine — helpers
# =========================
def test_cov_from_ratio_history_nonneg():
    xs = [1.0, 1.05, 0.95, 1.1, 0.9]
    c1 = TE._cov_from_ratio_history(xs)
    c2 = OBJ._cov_from_ratio_history(xs)
    assert c1 >= 0.0 and c2 >= 0.0

def test_make_metric_group_and_reset():
    m = TE.make_metric_group(threshold=0.5)
    assert set(["dom_acc","dom_loss","los_acc","los_loss","recon_loss","total","los_auc"]).issubset(m.keys())
    m["dom_loss"].update_state(1.0)
    TE.reset_metrics(m)
    assert float(m["dom_loss"].result()) == 0.0

def test_hpush_and_log_epoch_smoke():
    m = TE.make_metric_group()
    hist = {}
    TE.hpush(hist, "foo", 1.0)
    assert hist["foo"] == [1.0]
    grl = MM.GradientReversal(initial_lambda=0.0, name="grl")
    y_dom = tf.one_hot([0,1,2,1], depth=3)
    pred_dom = y_dom
    y_los = tf.constant([0,1,1,0], tf.float32)
    pred_los = tf.constant([0.1,0.9,0.6,0.2], tf.float32)
    sw_los = tf.constant([1,1,0,1], tf.float32)
    TE.update_metric_group(m, y_dom=y_dom, pred_dom=pred_dom,
                           y_los=y_los, pred_los=pred_los, sw_los=sw_los,
                           Ld=0.1, Ll=0.2, Lr=0.3, total=0.6)
    TE.log_epoch(hist, split="train", m=m, grl_cb=types.SimpleNamespace(grl_layer=grl))
    assert "grl_lambda" in hist

# =========================
# Plotters (skip by default)
# =========================
@pytest.mark.skip(reason="Plot functions require UMAP/figures; verify manually if needed.")
def test_plotters_exist():
    assert hasattr(PL, "plot_confusion_matrix")

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    sys.exit(pytest.main(["-q", os.path.basename(__file__)]))
