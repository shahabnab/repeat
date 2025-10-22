from pathlib import Path
import numpy as np
import tensorflow as tf

# import your helpers
from my_callbacks import *

from my_plotters import *
from my_models import *
from my_helping_functions import *
from training_engine import *
import time
from tensorflow.keras.losses import BinaryFocalCrossentropy
import optuna
from my_losses import _binary_crossentropy_mean, _binary_focal_mean
import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

from my_df_processing import *



def get_hyperparams(trial, epochs, batch, CONST_COEF, DATASET_NAMES, TRAIN_SIZE, ADAPTION_WITH_LABEL, SEED, save_dir):

    return dict(
        AE_EPOCHS=epochs,
        AE_BATCH=batch,
        GRL_LAMBDA_MAX=trial.suggest_float("GRL_LAMBDA_MAX", 0.6, 1.2),
        FOCAL_GAMMA=trial.suggest_float("FOCAL_GAMMA", 2.0, 3.5),
        PL_DOMAIN_ID=2,
        ENC_CONST_COEF=CONST_COEF,
        DEC_CONST_COEF=CONST_COEF,
        LATENT_DIM=trial.suggest_categorical("LATENT_DIM", [8, 16, 32, 64, 128]),
        COSINE_ALPHA=trial.suggest_float("COSINE_ALPHA", 0.02, 0.30),
        LR_WARMUP_EPOCHS=trial.suggest_categorical("LR_WARMUP_EPOCHS", [0, 2, 4, 6]),
        BASE_LR=trial.suggest_float("BASE_LR", 1e-5, 5e-4, log=True),
        CLIPNORM=trial.suggest_categorical("CLIPNORM", [1.0, 3.0, 5.0]),
        GRL_PEAK_FRAC=trial.suggest_float("GRL_PEAK_FRAC", 0.2, 1.0),
        DOM_LABEL_SMOOTH=trial.suggest_categorical("DOM_LABEL_SMOOTH", [0.0, 0.05, 0.1, 0.15]),
        ENT_TEMP=trial.suggest_categorical("ENT_TEMP", [0.5, 1.0, 1.5, 2.0]),
        COV_WINDOW=trial.suggest_categorical("COV_WINDOW", [0,5, 10]),
        LW_REC_START=1.0, LW_REC_END=0.5,
        LW_LOS_START=0.4547, LW_LOS_END=2.0,
        LW_DOM_START=0.02178, LW_DOM_END=1.5,
        METRIC_THRESHOLD=0.5,
        SEED=SEED,
        save_plots=save_dir,
        TRAIN_SIZE=TRAIN_SIZE,
        TRAIN1_NAME=DATASET_NAMES[0],
        TRAIN2_NAME=DATASET_NAMES[1],
        TEST_NAME=DATASET_NAMES[2],
        ADAPTION_WITH_LABEL=ADAPTION_WITH_LABEL,
    )


def prepare_datasets(X_tr, d_tr, l_tr, w_tr, X_val, d_val, l_val, w_val, 
                     X_test, d_test, l_test, w_test, h, num_dom):
    train_ds = make_dataset(X_tr, d_tr, l_tr, w_tr, h["AE_BATCH"], num_dom,
                            seed=h["SEED"], split="train", pl_mode="hide", pl_domain_id=h["PL_DOMAIN_ID"])
    val_ds   = make_dataset(X_val, d_val, l_val, w_val, h["AE_BATCH"], num_dom,
                            seed=h["SEED"], split="val", pl_mode="hide", pl_domain_id=h["PL_DOMAIN_ID"])
    test_ds  = make_dataset(X_test, d_test, l_test.squeeze(), w_test, h["AE_BATCH"], num_dom,
                            seed=h["SEED"], split="test", pl_mode="use", pl_domain_id=h["PL_DOMAIN_ID"])
    return train_ds, val_ds, test_ds

def build_models(seq_len, num_dom, latent_dim, enc_coef, dec_coef, trial_dir):
    ae, grl = build_DA_AE(seq_len, num_dom, latent_dim, enc_coef, dec_coef)
    return ae, grl

def find_threshold_for_f1(y_true, y_probs):
                best_f1 = -1.0
                best_thresh = 0.5
                for thresh in np.arange(0.0, 1.01, 0.01):
                    preds = (y_probs >= thresh).astype(int)
                    f1 = f1_score(y_true, preds)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                return best_thresh

def evaluate_model(ae,train_ds,val_ds, CIRS, LosLabels, Weights, h, config):
    
            
            ################### Los models ###############################
            los_model = model_prediction_los(ae)   # prob head
            los_model_entropy = model_for_entropy(ae)  # logits head
            ###################Validation Metrics#########################
            Xv, lv, wv=unpack_dataset(val_ds)
            mask = (wv != 0)
            los_probs = ae.predict(Xv, verbose=0)[2].squeeze()
            lv_filtered    = lv[mask]
            probs_filt     = los_probs[mask]
            h["METRIC_THRESHOLD"] = find_threshold_for_f1(lv_filtered, probs_filt)
            thrr=h["METRIC_THRESHOLD"]
            print(f"#######calculated metric threshold: "+str(thrr)+"#############")
            preds_bin = (los_probs >= float(h["METRIC_THRESHOLD"])).astype(int)
            preds_filtered = preds_bin[mask]
            val_auroc = float(roc_auc_score(lv_filtered,probs_filt))
            accuracy_valid = accuracy_score(lv_filtered, preds_filtered)
            f1_valid       = f1_score    (lv_filtered, preds_filtered)
            cm_valid       = confusion_matrix(lv_filtered, preds_filtered)
            val_logits =los_model_entropy.predict(Xv, verbose=0).reshape(-1)
            valid_entropy= mean_entropy(val_logits[~mask])
            validation_binary_entropy = binary_entropy(los_probs[~mask])
            f1_entropy_valid = f1_valid - valid_entropy
            validation_dsc = tf.keras.losses.dice(
            tf.cast(lv_filtered, tf.float32),
            tf.cast(probs_filt, tf.float32))
            validation_dice_loss = 1 - validation_dsc
            validation_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.cast(lv_filtered, tf.float32),
            tf.cast(probs_filt, tf.float32)))
            validation_bce_dice = validation_bce + validation_dice_loss

            validation_tp = tp(lv_filtered, preds_filtered)
            validation_tn = tn(lv_filtered, preds_filtered)
            
            validation_balanced_accuracy_score=balanced_accuracy_score(lv_filtered, preds_filtered)
            validation_focal_tversky = focal_tversky(lv_filtered, preds_filtered)
            validation_binary_focal_crossentropy = tf.reduce_mean(
                tf.keras.losses.binary_focal_crossentropy(
                    tf.cast(lv_filtered, tf.float32),
                    tf.cast(probs_filt, tf.float32),
                    apply_class_balancing=False,
                    alpha=0.25,#h.get("FOCAL_ALPHA", 0.25),
                    gamma=h.get("FOCAL_GAMMA", 2.0),
                    from_logits=False,
                    label_smoothing=0.0,
                    axis=-1
                )
            )
            val_logits = los_model_entropy.predict(Xv, verbose=0).reshape(-1)
            mask = (wv != 0)
            validation_weighted_bce_logits = weighted_bce_logits(
                 tf.cast(lv_filtered, tf.float32),
                 tf.cast(val_logits[mask], tf.float32)
             )



            ###################Training Metrics#########################
            Xtr, ltr, wtr=unpack_dataset(train_ds)
            los_probs = los_model.predict(Xtr, verbose=0).reshape(-1)  # probabilities in (0,1)
            mask = (wtr != 0)                     # supervised samples
            ltr_filtered = ltr[mask]
            probs_filt   = los_probs[mask]
            preds_filt   = (probs_filt >= float(h["METRIC_THRESHOLD"])).astype(int)

            # -------- supervised TRAIN metrics (correctly use probs where required) --------
            accuracy_train = accuracy_score(ltr_filtered, preds_filt)
            f1_train       = f1_score(ltr_filtered, preds_filt)
            cm_train       = confusion_matrix(ltr_filtered, preds_filt)

            # ---- entropy on UNLABELED (w==0) using LOGITS model (correct entropy input) ----
            train_logits = los_model_entropy.predict(Xtr, verbose=0).reshape(-1)   # logits
            unlabeled_mask = ~mask
            # NOTE: use your new helper bernoulli_entropy_from_logits() (see my previous message)
            train_entropy = float(bernoulli_entropy_from_logits(train_logits[unlabeled_mask]).numpy())
            train_mean_entropy= mean_entropy(train_logits[unlabeled_mask])
            f1_entropy_train = f1_train - train_entropy

            # also report Bernoulli entropy from PROBS for reference (different definition)
            train_binary_entropy = binary_entropy(los_probs[unlabeled_mask])  # expects probs

            # ---- BCE/Dice (BCE must use probabilities, not hard labels) ----
            # If you have a custom dice, use it; otherwise ensure this symbol exists in your codebase.
            train_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.cast(ltr_filtered, tf.float32),
                tf.cast(probs_filt,   tf.float32)
            ))
            # If you rely on a custom dice: dice_coef(y_true, y_pred_binary)
            train_dsc = tf.keras.losses.dice(  # <-- if this is not your custom, replace with your own dice function
                tf.cast(ltr_filtered, tf.float32),
                tf.cast(preds_filt,   tf.float32)
            )
            train_dice_loss = 1.0 - train_dsc
            train_bce_dice  = train_bce + train_dice_loss

            # other per-sample stats (your custom helpers)
            train_tp = tp(ltr_filtered, preds_filt)
            train_tn = tn(ltr_filtered, preds_filt)
            train_balanced_accuracy_score=balanced_accuracy_score(ltr_filtered, preds_filt)
            train_focal_tversky = focal_tversky(ltr_filtered, preds_filt)
            train_logits = los_model_entropy.predict(Xtr, verbose=0).reshape(-1)
            train_weighted_bce_logits = weighted_bce_logits(
                tf.cast(ltr_filtered, tf.float32),
                tf.cast(train_logits[mask], tf.float32)   # <- logits, not probs
            )

            # focal cross-entropy: use probabilities, read alpha/gamma from h if present
            train_binary_focal_crossentropy = tf.reduce_mean(
                tf.keras.losses.binary_focal_crossentropy(
                    tf.cast(ltr_filtered, tf.float32),
                    tf.cast(probs_filt,   tf.float32),
                    apply_class_balancing=False,
                    alpha=0.25,#h.get("FOCAL_ALPHA", 0.25),
                    gamma=h.get("FOCAL_GAMMA", 2.0),
                    from_logits=False,
                    label_smoothing=0.0,
                    axis=-1
                )
            )

            
            ################################# End Training Metrics #####################################


           
                        
        


            # ---------------- split-wise summaries using BEST weights ----------------
            TRAIN1_res  = predict_los_only(CIRS, LosLabels, "TRAIN1",  los_model,h, WEIGHTS=Weights)
            TRAIN2_res  = predict_los_only(CIRS, LosLabels, "TRAIN2",  los_model,h, WEIGHTS=Weights)
            ADAPTION_res= predict_los_only(CIRS, LosLabels, "ADAPTION",los_model,h, WEIGHTS=Weights)
            TEST_res    = predict_los_only(CIRS, LosLabels, "TEST",    los_model,h, WEIGHTS=Weights)

        
            

            

            # --- F1(TRAIN1∪TRAIN2) – Entropy(ADAPTION) ---
            f1ent = f1_minus_entropy(
                CIRS=CIRS,
                Labels=LosLabels,
                model_prob_head=los_model,                # probabilities
                h=h,
                model_logit_head=los_model_entropy,       # logits
                entropy_base='e',                         # 'e' for nats; use '2' for bits
                weights=Weights
            )


            

            
            # Probabilities (prob-head) per split
            probs_tr1  = predict_probs(los_model, CIRS["TRAIN1"],  Weights["TRAIN1"])
            probs_tr2  = predict_probs(los_model, CIRS["TRAIN2"],  Weights["TRAIN2"])
            probs_test = predict_probs(los_model, CIRS["TEST"],    Weights["TEST"])

            # Logits (logit-head) for ADAPTION (unsupervised regularizers)
            logits_adap = predict_logits(los_model_entropy, CIRS["ADAPTION"])

            mask_tr1 = (np.asarray(Weights["TRAIN1"]) != 0)
            mask_tr2 = (np.asarray(Weights["TRAIN2"]) != 0)

            y_tr1  = np.asarray(LosLabels["TRAIN1"])[mask_tr1].astype(int)
            y_tr2  = np.asarray(LosLabels["TRAIN2"])[mask_tr2].astype(int)
            y_test = np.asarray(LosLabels["TEST"]).astype(int)

            # Concatenate Train1+Train2 (source supervised)
            probs_tr12 = np.concatenate([probs_tr1, probs_tr2])
            y_tr12     = np.concatenate([y_tr1, y_tr2])

            # --- Supervised losses (source & test) on probabilities ---
            bce_train12   = _binary_crossentropy_mean(y_tr12, probs_tr12)
            focal_train12 = _binary_focal_mean      (y_tr12, probs_tr12)

            bce_test   = _binary_crossentropy_mean(y_test, probs_test)
            focal_test = _binary_focal_mean      (y_test, probs_test)

            

            # --- Unsupervised regularizers on ADAPTION (use logits for entropy) ---
            entropy_adaption_e    = float(bernoulli_entropy_from_logits(logits_adap, base='e').numpy())
            entropy_adaption_bits = float(bernoulli_entropy_from_logits(logits_adap, base=2).numpy())

            # --- Combined scoring examples ---
            LAMBDA_ENT = float(h.get("LAMBDA_ENTROPY", 1.0))
            combined_ce_ent_train12_adapt   = bce_train12   + LAMBDA_ENT * entropy_adaption_e
            combined_focal_ent_train12_adapt= focal_train12 + LAMBDA_ENT * entropy_adaption_e

            # For reporting both entropy forms on ADAPTION:
            # logits already computed above; reuse them
            adapt_entropy        = float(bernoulli_entropy_from_logits(logits_adap, base='e').numpy())
            adapt_binary_entropy = binary_entropy(tf.math.sigmoid(tf.convert_to_tensor(logits_adap, tf.float32)))



            los_model.save(h["save_plots"] / "los_model.keras")
            ae.save(h["save_plots"] / "ae_model.keras")
            adapt_probs = los_model.predict(CIRS["ADAPTION"][..., None], verbose=0).reshape(-1)
            K = max(100, int(0.02 * adapt_probs.size))          # 2% or at least 100 samples
            K = min(K, max(1, adapt_probs.size // 4))           # keep sane upper bound
            topK = np.partition(adapt_probs, -K)[-K:]
            botK = np.partition(adapt_probs,  K)[:K]
            pl_margin_adapt = float(topK.mean() - botK.mean())   # higher is better


        
            H_bits = float(entropy_adaption_bits)


            # prevalence mismatch (use supervised source only: TRAIN1+TRAIN2 with w!=0)
            src_mask_tr1 = (Weights["TRAIN1"] != 0)
            src_mask_tr2 = (Weights["TRAIN2"] != 0)
            y_src = np.concatenate([LosLabels["TRAIN1"][src_mask_tr1], LosLabels["TRAIN2"][src_mask_tr2]]).astype(float)
            y_src_mean = float(np.mean(y_src))
            p_tgt_mean = float(np.mean(adapt_probs))
            prev_gap = abs(p_tgt_mean - y_src_mean)
            config.update( { 
                            
                            "METRIC_THRESHOLD":h["METRIC_THRESHOLD"],
                           "CIRS_shapes Train1": CIRS["TRAIN1"].shape,
                            "CIRS_shapes Train2": CIRS["TRAIN2"].shape,
                            "CIRS_shapes Adaption": CIRS["ADAPTION"].shape,
                            "CIRS_shapes Test": CIRS["TEST"].shape,
                            "labels_shapes Train1": LosLabels["TRAIN1"].shape,
                            "labels_shapes Train2": LosLabels["TRAIN2"].shape,
                            "labels_shapes Adaption": LosLabels["ADAPTION"].shape,
                            "labels_shapes Test": LosLabels["TEST"].shape,
                            "Train1 weights": pd.Series(Weights["TRAIN1"]).value_counts().to_dict(),
                            "Train2 weights": pd.Series(Weights["TRAIN2"]).value_counts().to_dict(),
                            "Train1Accuracy":TRAIN1_res["accuracy"],
                            "Train1Loss": TRAIN1_res["loss"],
                            "Train1F1Score": TRAIN1_res["f1_score"],
                            "Train1_confusion_matrix": TRAIN1_res["confusion_matrix"],
                            "Train2Accuracy": TRAIN2_res["accuracy"],
                            "Train2Loss": TRAIN2_res["loss"],
                            "Train2F1Score": TRAIN2_res["f1_score"],
                            "Train2_confusion_matrix": TRAIN2_res["confusion_matrix"],
                            "AdaptionAccuracy": ADAPTION_res["accuracy"],
                            "AdaptionLoss": ADAPTION_res["loss"],
                            "AdaptionF1Score": ADAPTION_res["f1_score"],
                            "Adaption_confusion_matrix": ADAPTION_res["confusion_matrix"],
                            "Validation Accuracy_filtered": accuracy_valid,
                            "Validation F1 Score_filtered": f1_valid,
                            "Validation Confusion Matrix_filtered": cm_valid,
                            "Validation Entropy_w0": valid_entropy,
                            "TestLoss": TEST_res["loss"],
                            "TestAccuracy": TEST_res["accuracy"],
                            "TestF1Score": TEST_res["f1_score"],
                            "Test_confusion_matrix": TEST_res["confusion_matrix"],
                            "AccuracyTrain_filtered": accuracy_train,
                            "F1Train_filtered": f1_train,
                            "ConfusionMatrixTrain_filtered": cm_train,
                            "EntropyTrain_w0": train_entropy,
                            "F1EntropyTrain": f1_entropy_train,
                            "Validation F1 (entropy)": f1_entropy_valid,
                            "score_f1_minus_entropy": f1ent["score_f1_minus_entropy"],
                            "f1_train12": f1ent["f1_train12"],
                            "entropy_adaption": f1ent["entropy_adaption"],
                            "bce_train12": bce_train12,
                            "focal_train12": focal_train12,
                            "bce_test": bce_test,
                            "focal_test": focal_test,
                            "entropy_adaption_e": entropy_adaption_e,
                            "entropy_adaption_bits": entropy_adaption_bits,
                            "lambda_entropy": LAMBDA_ENT,
                            "combined_ce_plus_lambda_entropy": combined_ce_ent_train12_adapt,
                            "combined_focal_plus_lambda_entropy": combined_focal_ent_train12_adapt,
                            "validation_binary_entropy": validation_binary_entropy,
                            "validation_dsc": validation_dsc.numpy().item(),
                            "validation_dice_loss": validation_dice_loss.numpy().item(),
                            "validation_bce_dice": validation_bce_dice.numpy().item(),
                            "validation_bce": validation_bce.numpy().item(),
                            "validation_tp": validation_tp.numpy().item(),
                            "validation_tn": validation_tn.numpy().item(),
                            "validation_focal_tversky": validation_focal_tversky.numpy().item(),
                            "validation_binary_focal_crossentropy": validation_binary_focal_crossentropy.numpy().item(),
                            "train_binary_entropy": train_binary_entropy,
                            "train_dsc": train_dsc.numpy().item(),
                            "train_dice_loss": train_dice_loss.numpy().item(),
                            "train_bce_dice": train_bce_dice.numpy().item(),
                            "train_bce": train_bce.numpy().item(),
                            "train_tp": train_tp.numpy().item(),
                            "train_tn": train_tn.numpy().item(),
                            "train_focal_tversky": train_focal_tversky.numpy().item(),
                            "train_bce_focal_crossentropy": train_binary_focal_crossentropy.numpy().item(),
                            "adapt_entropy": adapt_entropy,
                            "adapt_binary_entropy": adapt_binary_entropy,
                            "train_balanced_accuracy_score":train_balanced_accuracy_score,
                            "validation_balanced_accuracy_score":validation_balanced_accuracy_score,
                            "train_mean_entropy":train_mean_entropy,
                            "train_weighted_bce_logits": train_weighted_bce_logits.numpy().item(),
                            "validation_weighted_bce_logits": validation_weighted_bce_logits.numpy().item(),
                            "val_auroc": val_auroc,
                            "pl_margin_adapt": pl_margin_adapt,
                            "H_bits": H_bits,
                            "prev_gap": prev_gap,

                        })
            config = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in config.items()}
            for k, v in config.items():
                if isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    config[k] = float(v) if '.' in v else int(v)

            save_to_excel(config, h["save_plots"])
          
def objective(
    trial,
    X_tr, X_val, X_test,
    d_tr, d_val, d_test,
    w_tr, w_val, w_test,
    l_tr, l_val, l_test,
    num_dom,
    balanced_dtsets,
    CIRS, LosLabels, Domains, Weights,
    SAVE_PLOTS_ROOT: Path,
    TRAIN_SIZE: int,
    DATASET_NAMES: list[str],
    ADAPTION_WITH_LABEL: int,
    SEED: int,
    force_epochs: int | None = None,
    force_batch: int | None = None,
    save_tag: str | None = None,
    enable_test_results: bool = False
):
    trial_dir = SAVE_PLOTS_ROOT / (save_tag or f"trial_{trial.number:03d}")
    trial_dir.mkdir(parents=True, exist_ok=True)

    epochs = 30 if force_epochs is None else int(force_epochs)
    batch  = 128 if force_batch is None else int(force_batch)
    CONST_COEF = trial.suggest_float("enc_const", 0.5, 3.0, step=0.5)

    h = get_hyperparams(
        trial, epochs, batch, CONST_COEF,
        DATASET_NAMES, TRAIN_SIZE, ADAPTION_WITH_LABEL, SEED, trial_dir
    )
    config = dict(h)

    config.update({
        "Train12 weights": pd.Series(w_tr).value_counts().to_dict(),
         "validation weights": pd.Series(w_val).value_counts().to_dict(),          
        "Train1 Domains": pd.Series(Domains["TRAIN1"]).value_counts().to_dict(),
        "Train2 Domains": pd.Series(Domains["TRAIN2"]).value_counts().to_dict(),
        "Train12 Domains": pd.Series(d_tr).value_counts().to_dict(),
        "validation Domains": pd.Series(d_val).value_counts().to_dict(),
        "Domains_shapes": {key: Domains[key].shape for key in Domains.keys()},
        "balanced_dsets_shapes": {key: balanced_dtsets[key].shape for key in balanced_dtsets.keys()},
    })

    train_ds, val_ds, test_ds = prepare_datasets(
        X_tr, d_tr, l_tr, w_tr,
        X_val, d_val, l_val, w_val,
        X_test, d_test, l_test, w_test,
        h, num_dom
    )
    ae, grl = build_models(
        seq_len=X_tr.shape[1], num_dom=num_dom,
        latent_dim=h["LATENT_DIM"],
        enc_coef=h["ENC_CONST_COEF"],
        dec_coef=h["DEC_CONST_COEF"],
        trial_dir=trial_dir
    )
    steps_per_epoch = math.ceil(len(X_tr) / h["AE_BATCH"])
    optuna_score = train_model(ae, grl, train_ds, val_ds,test_ds, h, trial, config, num_dom, steps_per_epoch, enable_test_results)
    evaluate_model(ae, train_ds, val_ds, CIRS, LosLabels, Weights, h,config)
    plotting_figures(ae, CIRS, LosLabels, Domains, X_test, h)
    trial.set_user_attr("optuna_score", float(optuna_score))
    return optuna_score



def plotting_figures(ae, CIRS, LosLabels, Domains, X_test, h):
            
            encoder = tf.keras.Model(
                        inputs  = ae.input,
                        outputs = ae.get_layer("latent_vector").output,
                        name    = "encoder"
                    )
            los_model = model_prediction_los(ae)   # prob head
            decoder_layer = ae.get_layer("reconstruction")   # THIS is a tf.keras.Model
            # 2) re-wrap on fresh Input
            latent_in  = tf.keras.Input(shape=(h["LATENT_DIM"],), name="latent_in")
            recon_out  = decoder_layer(latent_in)
            decoder    = tf.keras.Model(latent_in, recon_out, name="decoder")

           
            plot_latent_umap(encoder, CIRS, Domains,h["save_plots"])
            plot_latent_umap_by_los(encoder, CIRS, Domains, LosLabels, h["save_plots"])

            plot_latent_umap_input(CIRS, Domains,h["save_plots"])
            
            plot_umap_input_by_los(CIRS, Domains, LosLabels, h["save_plots"])
            y_pred_los     = los_model.predict(X_test).flatten()        # P(NLOS)

            plot_encoded_signals_pro(
                ae=ae, decoder=decoder, encoder=encoder,
                CIRS=CIRS, labels=LosLabels,
                save_plots=h["save_plots"]
            )
            y_test_los  = LosLabels["TEST"].astype(int) # shape: (N,)
            plot_predicted_vs_real(y_pred_los, y_test_los, h["save_plots"])
            probe_layer_names = ["latent_vector", "latent_cls", "attention_gated","los_prob"]
            probe_outputs = [ae.get_layer(name).output
                                for name in probe_layer_names]
            probe_model   = tf.keras.Model(inputs=ae.input,
                                    outputs=probe_outputs)
            probe_ae(ae, CIRS,LosLabels, h)
            evaluate_model_performance(probe_model, CIRS, LosLabels, h)
            plot_confusion_matrix(los_model, CIRS, LosLabels, h)
            plot_confusion_matrices_grid( los_model, CIRS, LosLabels, h)






            