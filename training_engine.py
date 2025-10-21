from my_losses import *
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import os
import time
from tensorflow.keras import backend as K
import optuna
from my_callbacks import *
from tensorflow.keras.losses import BinaryFocalCrossentropy
from my_plotters import *
from my_df_processing import *
import math
from my_models import *


def _cov_from_ratio_history(loss_series, window: int | None = None, eps: float = 1e-8) -> float:
                 
    if not loss_series:
        return float("inf")

    if window is not None and window > 0 and len(loss_series) > window:
        xs = loss_series[-window:]
    else:
        xs = list(loss_series)

    t = 0
    mu_L = 0.0   # mean of L
    mu_l = 0.0   # mean of ℓ
    M_l  = 0.0   # population variance of ℓ

    for L in xs:
        L = float(L)
        t += 1

        # (5) μ_L
        muL_prev = mu_L
        mu_L = (1.0 - 1.0/t) * muL_prev + (1.0/t) * L

        # ℓ_t (ℓ_1 = 1.0)
        denom = muL_prev if (t > 1 and muL_prev > eps) else max(L, eps)
        ell_t = L / denom

        # (6) μ_ℓ
        mu_l_prev = mu_l
        mu_l = (1.0 - 1.0/t) * mu_l_prev + (1.0/t) * ell_t

        # (7) M_ℓ (population variance)
        M_l = (1.0 - 1.0/t) * M_l + (1.0/t) * (ell_t - mu_l_prev) * (ell_t - mu_l)

    sigma_l = math.sqrt(max(M_l, 0.0))
    return float(sigma_l / max(mu_l, eps))


def reset_metrics(group: dict):
            for m in group.values():
                m.reset_state()
def update_metric_group(m: dict, *,
                y_dom, pred_dom,
                y_los, pred_los, sw_los,
                Ld, Ll, Lr,Ld_no_cdane, total):

    m["dom_acc"].update_state(y_dom, pred_dom)
    m["dom_loss"].update_state(Ld)

    m["los_acc"].update_state(y_los, pred_los, sample_weight=sw_los)
    m["los_auc"].update_state(y_los, pred_los, sample_weight=sw_los)

    m["los_loss"].update_state(Ll)
    m["recon_loss"].update_state(Lr)
    m["total"].update_state(total)
    m["dom_loss_no_cdane"].update_state(Ld_no_cdane)


def hpush(history: dict, key: str, value) -> None:
        history.setdefault(key, []).append(float(value))

def log_epoch(history: dict, split: str, m: dict, *, grl_cb=None):
    # split in {"train","val","test"}
    pfx = f"{split}_"
    hpush(history, "loss" if split=="train" else f"{split}_loss", m["total"].result())
    hpush(history, f"{pfx}dom_acc",    m["dom_acc"].result())
    hpush(history, f"{pfx}dom_loss_no_cdane", m["dom_loss_no_cdane"].result())
    hpush(history, f"{pfx}dom_loss",   m["dom_loss"].result())
    hpush(history, f"{pfx}los_acc",    m["los_acc"].result())
    hpush(history, f"{pfx}los_loss",   m["los_loss"].result())
    hpush(history, f"{pfx}recon_loss", m["recon_loss"].result())
    hpush(history, f"{pfx}los_auc",    m["los_auc"].result())

    # 🔹 Add GRL lambda tracking (only once per epoch, usually at train logging)
    if split == "train" and grl_cb is not None:
        hpush(history, "grl_lambda", float(grl_cb.grl_layer.hp_lambda.numpy()))


def make_metric_group(threshold: float = 0.5):
            return {
                "dom_acc":    tf.keras.metrics.CategoricalAccuracy(name="dom_acc"),
                "dom_loss":   tf.keras.metrics.Mean(name="dom_loss"),
               "dom_loss_no_cdane": tf.keras.metrics.Mean(name="dom_loss_no_cdane"),
                "los_acc":    tf.keras.metrics.BinaryAccuracy(threshold=threshold, name="los_acc"),
                "los_loss":   tf.keras.metrics.Mean(name="los_loss"),
                "recon_loss": tf.keras.metrics.Mean(name="recon_loss"),
                "total":      tf.keras.metrics.Mean(name="total"),
                "los_auc":    tf.keras.metrics.AUC(curve="ROC", name="los_auc"),


                
            }
             

def train_model(ae, grl, train_ds, val_ds,test_ds,h, trial, config, num_dom,steps_per_epoch, enable_test_results: bool = False):
            device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
            BFC_mean = BinaryFocalCrossentropy(gamma=h["FOCAL_GAMMA"],from_logits=False)  
            with tf.device(device):
            
               
                
                
                
                total_steps     = max(1, h["AE_EPOCHS"] * steps_per_epoch)
                warmup_steps    = h["LR_WARMUP_EPOCHS"] * steps_per_epoch  # int

                lr_sched = WarmupThenCosine(
                    base_lr=h["BASE_LR"],
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    alpha=h["COSINE_ALPHA"],
                )
            

            
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched,clipnorm=h["CLIPNORM"])

                lw_start = tf.constant([h["LW_REC_START"], h["LW_DOM_START"], h["LW_LOS_START"]], tf.float32)
                lw_end   = tf.constant([h["LW_REC_END"],   h["LW_DOM_END"],   h["LW_LOS_END"]],   tf.float32)
                def interp_lw(epoch, total):
                    p = tf.cast(epoch, tf.float32) / tf.cast(tf.maximum(1, total-1), tf.float32)
                    return (1.0 - p) * lw_start + p * lw_end    


                

                
                    
            grl_cb = GRLSchedule(grl, h["AE_EPOCHS"],
                        lambda_min=0.0, lambda_max=h["GRL_LAMBDA_MAX"], peak_frac=h["GRL_PEAK_FRAC"]
                    )

            
            history = {
                    "loss":[], "train_dom_acc":[], "train_dom_loss":[], "train_los_acc":[], "train_los_loss":[], "train_recon_loss":[],
                    "val_loss":[], "val_dom_acc":[], "val_dom_loss":[], "val_los_acc":[], "val_los_loss":[], "val_recon_loss":[],
                    "grl_lambda":[],"optuna_score":[], "train_wLr":[], "train_wLd":[], "train_wLi":[],
                        "val_wLr":[],   "val_wLd":[],   "val_wLi":[],"val_cov_rec": [], "val_cov_dom": [], "val_cov_los": []
             
                }

            # === Shared helpers ==========================================================
            EPS = tf.constant(1e-6, tf.float32)

            def _entropy_weight(pred_los, beta):
                """CDAN-E task-entropy weight: w_e = 1 + exp(-beta * H(p))."""
                p  = tf.clip_by_value(tf.cast(pred_los, tf.float32), EPS, 1.0 - EPS)  # (B,1)
                g2 = tf.concat([p, 1.0 - p], axis=-1)                                  # (B,2)
                ent = -tf.reduce_sum(g2 * tf.math.log(g2 + EPS), axis=-1)              # (B,)
                w_e = 1.0 + tf.exp(-beta * ent)
                return tf.stop_gradient(w_e)             # (B,)


            
    

            
            @tf.function
            def _forward_and_losses(x, y_rec, y_dom, y_los, sw_los, lw_vec, training):
                """One pass (train/eval). Returns: total, pred_dom, pred_los, Ld, Ll, Lr."""
                # Forward
                pred_rec, pred_dom, pred_los = ae(x, training=training)
                # Reconstruction & LOS
                Lr = reconstruction_loss(y_rec, pred_rec)                         # scalar
                Ll = los_loss(y_los, pred_los, sw_los, BFC_mean)                  # scalar
                beta = tf.cast(h.get("ENT_TEMP", 1.0), tf.float32)
                w_e  = _entropy_weight(pred_los, beta)                # (B,)
                Ld = domain_loss(y_dom, pred_dom, w_e, h)                  # scalar
                Ld_no_cdane=domain_loss_without_cdane(y_dom, pred_dom,h)
                lw0 = tf.cast(lw_vec[0], tf.float32)  # recon
                lw1 = tf.cast(lw_vec[1], tf.float32)  # domain
                lw2 = tf.cast(lw_vec[2], tf.float32)  # los
                total = lw0 * Lr + lw1 * Ld + lw2 * Ll
                return total, pred_dom, pred_los, Ld, Ll, Lr, Ld_no_cdane

            # === Train / Val / Test steps ===============================================

            @tf.function
            def train_step(x, y_rec, y_dom, y_los, sw_los, lw_vec):
                """Single training step with CDAN-E weighting."""
                
                with tf.GradientTape() as tape:

                    total, pred_dom, pred_los, Ld, Ll, Lr, Ld_no_cdane = _forward_and_losses(
                        x, y_rec, y_dom, y_los, sw_los, lw_vec, training=True
                    )
                  
                grads = tape.gradient(total, ae.trainable_variables)
                optimizer.apply_gradients(zip(grads, ae.trainable_variables))
                return total, pred_dom, pred_los, Ld, Ll, Lr, Ld_no_cdane

            @tf.function
            def val_step(x, y_rec, y_dom, y_los, sw_los, lw_vec):
                """Validation step (no grads)."""
                return _forward_and_losses(x, y_rec, y_dom, y_los, sw_los, lw_vec, training=False)
            start = time.time()
            COV_WINDOW = int(h.get("COV_WINDOW", 0)) 
            prune_after = 5   
            score_hist = []
            train_wLr_m = tf.keras.metrics.Mean(); train_wLd_m = tf.keras.metrics.Mean(); train_wLi_m = tf.keras.metrics.Mean()
            val_wLr_m   = tf.keras.metrics.Mean(); val_wLd_m   = tf.keras.metrics.Mean(); val_wLi_m   = tf.keras.metrics.Mean()
            @tf.function
            def test_step(x, y_rec, y_dom, y_los, sw_los, lw_vec):
                """Test step (no grads)."""
                return _forward_and_losses(x, y_rec, y_dom, y_los, sw_los, lw_vec, training=False)

            for epoch in range(h["AE_EPOCHS"]):
                        grl_cb.on_epoch_begin(epoch)
                    
                        #prog_unfreeze.on_epoch_begin(epoch)
                        lw_vec = interp_lw(epoch, h["AE_EPOCHS"])
                        train_m = make_metric_group(h["METRIC_THRESHOLD"])
                        val_m   = make_metric_group(h["METRIC_THRESHOLD"])
                        test_m   = make_metric_group(h["METRIC_THRESHOLD"])
                        train_wLr_m.reset_state(); train_wLd_m.reset_state(); train_wLi_m.reset_state()
                        val_wLr_m.reset_state();   val_wLd_m.reset_state();   val_wLi_m.reset_state()

                      
                        reset_metrics(train_m)
                        for x, (y_rec, y_dom, y_los,sw_los) in train_ds:
                            total,pred_dom,pred_los,Ld,Ll,Lr,Ld_no_cdane = train_step(x, y_rec, y_dom, y_los, sw_los, lw_vec)
                            update_metric_group(train_m, y_dom=y_dom, pred_dom=pred_dom,
                                    y_los=y_los, pred_los=pred_los, sw_los=sw_los,
                                    Ld=Ld, Ll=Ll, Lr=Lr,Ld_no_cdane=Ld_no_cdane, total=total)
                            
                            lw0 = tf.cast(lw_vec[0], tf.float32)
                            lw1 = tf.cast(lw_vec[1], tf.float32)
                            lw2 = tf.cast(lw_vec[2], tf.float32)
                           

                            train_wLr_m.update_state(lw0 * Lr)
                            train_wLd_m.update_state(lw1 * Ld)
                            train_wLi_m.update_state(lw2 * Ll)
                        reset_metrics(val_m)
                        for x, (y_rec, y_dom, y_los,sw_los) in val_ds:
                            total,pred_dom,pred_los,Ld,Ll,Lr,Ld_no_cdane = val_step(x, y_rec, y_dom, y_los, sw_los, lw_vec)
                            update_metric_group(val_m, y_dom=y_dom, pred_dom=pred_dom,
                                    y_los=y_los, pred_los=pred_los, sw_los=sw_los,
                                    Ld=Ld, Ll=Ll, Lr=Lr,Ld_no_cdane=Ld_no_cdane, total=total)
                            lw0 = tf.cast(lw_vec[0], tf.float32)
                            lw1 = tf.cast(lw_vec[1], tf.float32)
                            lw2 = tf.cast(lw_vec[2], tf.float32)
                            val_wLr_m.update_state(lw0 * Lr)
                            val_wLd_m.update_state(lw1 * Ld)
                            val_wLi_m.update_state(lw2 * Ll) 
                        history["train_wLr"].append(float(train_wLr_m.result().numpy()))
                        history["train_wLd"].append(float(train_wLd_m.result().numpy()))
                        history["train_wLi"].append(float(train_wLi_m.result().numpy()))
                        history["val_wLr"].append(float(val_wLr_m.result().numpy()))
                        history["val_wLd"].append(float(val_wLd_m.result().numpy()))
                        history["val_wLi"].append(float(val_wLi_m.result().numpy()))
                        log_epoch(history, "train", train_m, grl_cb=grl_cb)
                        log_epoch(history, "val",   val_m)
                        
                        # --- Compute CoVs (use confusion series for domain to match the objective) ---
                        cov_rec = _cov_from_ratio_history(history["val_recon_loss"], window=COV_WINDOW)
                        cov_los = _cov_from_ratio_history(history["val_los_loss"],   window=COV_WINDOW)

                        K = float(num_dom)
                        logK = float(np.log(K))

                        # Smooth last-W losses for robustness
                        W = max(3, COV_WINDOW or 5)
                        def med_last(xs):
                            xs = xs[-W:] if len(xs) >= W else xs
                            return float(np.median(xs)) if xs else float("inf")

                        loss_rec = med_last(history["val_recon_loss"])
                        #loss_dom_raw = med_last(history["val_dom_loss"])
                        loss_dom_raw = med_last(history["val_dom_loss_no_cdane"])
                        loss_los = med_last(history["val_los_loss"])

                        # Domain confusion loss (0 is best), normalized by ln K
                        dom_confusion_loss = abs(loss_dom_raw - logK) / max(1e-9, logK)

                        # Keep a dedicated history for the confusion loss so CoV aligns with objective
                        history.setdefault("val_dom_conf", []).append(dom_confusion_loss)
                        cov_dom = _cov_from_ratio_history(history["val_dom_conf"], window=COV_WINDOW)

                        # Guard CoVs
                        cov_rec = cov_rec if np.isfinite(cov_rec) else 1e6
                        cov_dom = cov_dom if np.isfinite(cov_dom) else 1e6
                        cov_los = cov_los if np.isfinite(cov_los) else 1e6
                        mean_weights=(cov_rec + cov_dom + cov_los) / 3.0

                        L   = np.array([loss_los, dom_confusion_loss, loss_rec], dtype=float)

                        # Tiny floor to avoid exact zeros; array-wise op!
                        #######################################################
                        cov_vec = np.array([cov_los, cov_dom, cov_rec], float)

                        # 1) Drop bad entries
                        cov_vec = np.where(np.isfinite(cov_vec), cov_vec, 0.0)

                        # 2) If everything collapsed, fall back to equal weights
                        eps = 1e-8
                        if cov_vec.sum() <= eps:
                            cov_vec[:] = 1.0

                        # 3) Optional tiny floor for numerics (keeps proportions; doesn’t resurrect dead entries)
                        cov_vec = np.maximum(cov_vec, 1e-12)

                        alpha = cov_vec / cov_vec.sum()

                        # Use SAME losses you train with (LOS, DOM-CONF, RECON) in SAME order
                        L = np.array([loss_los, dom_confusion_loss, loss_rec], dtype=float)
                        optuna_score = float((alpha * L).sum())
                        ###########################################################
                        ###########################################################
                        """ lam_cov = h.get("COV_LAMBDA", 0.3)
                        cov = np.clip(np.array([cov_los, cov_dom, cov_rec], float), 0.0, 10.0)
                        
                        w = 1.0 + lam_cov * cov
                        alpha = w / w.sum()

                        optuna_score = float((alpha * L).sum()) """
                        ###########################################################

                        ####
                        ##################################
                        #if enable_test_results:
                        reset_metrics(test_m)
                        for x, (y_rec, y_dom, y_los,sw_los) in test_ds:
                            total,pred_dom,pred_los,Ld,Ll,Lr,Ld_no_cdane = test_step(x, y_rec, y_dom, y_los, sw_los, lw_vec)
                            update_metric_group(test_m, y_dom=y_dom, pred_dom=pred_dom,
                                    y_los=y_los, pred_los=pred_los, sw_los=sw_los,
                                    Ld=Ld, Ll=Ll, Lr=Lr,Ld_no_cdane=Ld_no_cdane, total=total)
                        log_epoch(history, "test",  test_m)    
                            
                           


                        ##################################
                        ####
                        




                       
                        # Save CoVs and the objective you optimize
                        history["val_cov_rec"].append(cov_rec)
                        history["val_cov_dom"].append(cov_dom)
                        history["val_cov_los"].append(cov_los)

                        history.setdefault("optuna_score", []).append(float(optuna_score))
                        score_hist.append(float(optuna_score))

                        # Prune/track on the SAME metric you optimize
                        trial.report(float(optuna_score), step=epoch)
                        if epoch >= prune_after and trial.should_prune():
                            trial.set_user_attr("optuna_score_history", score_hist)
                            raise optuna.TrialPruned()
                        print_epoch_line_from_history(epoch, h, history)
              
            history["optuna_score"]=score_hist
            training_time = time.time() - start
            # 1) Evaluate last-epoch weights on TEST
            plot_history_dashboard(
                history,
                save_path=h["save_plots"],
                smoothing=0,        # optional
                marker_every=1,       # every epoch has a symbols
                filename_prefix="metric"
            )

            # domain confusion (optional)
            chance = 1.0 / float(num_dom)
            val_dom_acc_now = float(history["val_dom_acc"][-1]) if history["val_dom_acc"] else chance
            den = max(1e-9, 1.0 - chance)
            dom_confusion = 1.0 - abs(val_dom_acc_now - chance) / den

            Lr_t=history["train_wLr"][-1]
            Ld_t=history["train_wLd"][-1]
            Ll_t=history["train_wLi"][-1]
            total_t=Lr_t+Ld_t+Ll_t
            Lr_V=history["val_wLr"][-1]
            Ld_V=history["val_wLd"][-1]
            LD_v_no_cdane=history["val_dom_loss_no_cdane"][-1]
            Ll_v=history["val_wLi"][-1]

            total_v=Lr_V+Ld_V+Ll_v
            config.update({
                            "dom_confusion": dom_confusion,
                            "Lr_t":Lr_t,
                            "Ld_t":Ld_t,
                            "Ll_t":Ll_t,
                            "total_t":total_t,
                            "Lr_V":Lr_V,
                            "Ld_V":Ld_V,
                            "LD_v_no_cdane":LD_v_no_cdane,
                            "Ll_v":Ll_v,
                            "total_v":total_v, 
                            "cov_rec": cov_rec,
                            "cov_dom": cov_dom,
                            "cov_los": cov_los,
                            "optuna_score": float(optuna_score),
                            "COV_WINDOW": COV_WINDOW,
                            "training_time": training_time
                        })
           
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(h["save_plots"], "history.csv"), index=False)
            history_df.to_pickle(os.path.join(h["save_plots"], "history.pkl"))
    
            return optuna_score
