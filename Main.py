# ===== Standard library =====
import os
import sys
import math
import time
import gc
from pathlib import Path
import argparse
from functools import partial

# ===== Third-party =====
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg', force=True)   # set BEFORE importing pyplot
import matplotlib.pyplot as plt

# Set TF log level BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision, backend as K

import optuna
import optuna.visualization.matplotlib as ovm
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

# ===== Local project modules =====
from my_callbacks import *
from my_losses import *
from my_plotters import *
from my_cir_processing import *
from my_df_processing import *
from my_models import *
from my_helping_functions import *
from training_engine import *
from my_objective_fn import objective


# ===== Global config =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # quieter TF logs
matplotlib.use('Agg', force=True)         # non-interactive backend for saving plots

# Enable dynamic GPU memory allocation (important for multi-run stability)
#putting in try catch in the case of no GPU available
try:
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
except Exception as e:
    print(f"⚠️ Could not set memory growth: {e}")




        







def run_scenario(DATASET_NAMES,epochs_num,batch_size,opt_add, TRAIN_SIZE, n_trials, direction, SEED=42):

        print(f"=== Scenario: {DATASET_NAMES} with TRAIN_SIZE={TRAIN_SIZE} ===")
        #setting the seed for reproducibility
        tf.random.set_seed(SEED)
        set_seed(SEED)

        #setting different roles for the datasets
        DATASET_ROLES = ["TRAIN1", "TRAIN2", "ADAPTION", "TEST"]
        #the number if samples with labels that can be used from the target domain
        ADAPTION_WITH_LABEL = 0

        scenario_tag = f"{DATASET_NAMES[0]}_{DATASET_NAMES[1]}_to_{DATASET_NAMES[2]}"
        SAVE_PLOTS_ROOT = Path(scenario_tag)
        SAVE_PLOTS_ROOT.mkdir(exist_ok=True)
        WHOLE_RES_XLSX = "whole_res.xlsx"
        CONFIG_ATTR = "config"

        
        print("📥  Downloading + slicing datasets …")
        #downloadig the datasets and slicing them according to the roles
        raw_datasets   = download_dataset(DATASET_NAMES, SEED)
        balanced_dtsets = slicing_dts(
            raw_datasets,
            save_path=f"res_SEED/{SEED}/sample_{ADAPTION_WITH_LABEL}",
            datasets_names=DATASET_NAMES,
            dt_rules=DATASET_ROLES,
            tr_size=TRAIN_SIZE,
            SEED=SEED,
        )
        #extracting the CIRs, labels, domains and weights from the datasets

        CIRS    = CIR_pipeline(balanced_dtsets)
        LosLabels  = take_labels(balanced_dtsets)
        Domains = take_domains(balanced_dtsets)
        Weights = take_weights(balanced_dtsets, ADAPTION_WITH_LABEL)

        # ---- assemble arrays ---- 
        X  = np.concatenate([CIRS[k] for k in ("TRAIN1","TRAIN2","ADAPTION")])[..., None].astype('float32', copy=True)
        d  = np.concatenate([Domains[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.int32, copy=True)
        w  = np.concatenate([Weights[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.float32,copy=True)
        l  = np.concatenate([LosLabels[k] for k in ("TRAIN1","TRAIN2","ADAPTION")]).astype(np.float32,copy=True)
        

        #test set
        X_test = np.array(CIRS["TEST"])[..., None].astype('float32')
        d_test = np.array(Domains["TEST"]).astype(np.int32,copy=True)
        l_test = np.array(LosLabels["TEST"]).astype(np.float32,copy=True).reshape(-1, 1)
        w_test = np.array(Weights["TEST"]).astype(np.float32,copy=True)
        #number of domains
        num_dom = int(d.max() + 1)
        #spliting the train and validation 
        joint = list(zip(d, l))
        X_tr, X_val, d_tr, d_val, w_tr, w_val, l_tr, l_val = train_test_split(
            X, d, w, l, test_size=0.3, random_state=SEED, stratify=joint, shuffle=True
        ) 


        
     
        
        objective_fn = partial(
                objective,
                X_tr=X_tr, X_val=X_val, X_test=X_test,
                d_tr=d_tr, d_val=d_val, d_test=d_test,
                w_tr=w_tr, w_val=w_val, w_test=w_test,
                l_tr=l_tr, l_val=l_val, l_test=l_test,
                num_dom=num_dom, balanced_dtsets=balanced_dtsets,
                CIRS=CIRS, LosLabels=LosLabels, Domains=Domains, Weights=Weights,
                SAVE_PLOTS_ROOT=SAVE_PLOTS_ROOT,
                TRAIN_SIZE=TRAIN_SIZE,
                DATASET_NAMES=DATASET_NAMES,
                ADAPTION_WITH_LABEL=ADAPTION_WITH_LABEL,
                SEED=SEED,
                )
        run_optuna(
            objective_fn,epochs_num,opt_add,batch_size,
            SEED, SAVE_PLOTS_ROOT, CONFIG_ATTR, WHOLE_RES_XLSX,
            n_trials=n_trials, direction=direction
        )
       

        
        

    
        








# ---------------- utilities ----------------
def combine_trial_reports(scenario_dir: Path):
    paths = list(scenario_dir.glob("trial_*/report.xlsx"))
    if not paths:
        print(f"ℹ️  No report.xlsx under {scenario_dir}. Skipping.")
        return
    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p)
            df["trial_id"] = p.parent.name
            frames.append(df)
        except Exception as e:
            print(f"⚠️  Could not read {p}: {e}")
    if not frames:
        print(f"ℹ️  No readable report files under {scenario_dir}.")
        return
    combined = pd.concat(frames, ignore_index=True)
    out_xlsx = scenario_dir / "all_reports.xlsx"
    combined.to_excel(out_xlsx, index=False)
    try:
        combined.to_parquet(scenario_dir / "all_reports.parquet", index=False)
    except Exception as e:
        print(f"⚠️  Parquet write failed: {e}")
    print(f"✅  Combined {len(paths)} files → {out_xlsx}")


def save_parallel_wide(study, out_dir, params=None,
                       inch_per_param=2.0, height=7, max_width_in=40, dpi=240):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # If no params specified, plot all *varying* params to avoid singular axes.
    if params is None:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        keys = sorted({k for t in completed for k in t.params})
        varying = []
        for k in keys:
            vals = [t.params[k] for t in completed if k in t.params]
            if not vals: continue
            try:
                if max(vals) != min(vals): varying.append(k)
            except TypeError:
                if len(set(vals)) > 1: varying.append(k)
        params = varying or keys

    ax = ovm.plot_parallel_coordinate(study, params=params)
    fig = ax.figure

    # Width grows with number of params
    width = min(max_width_in, max(12, inch_per_param * len(params)))
    fig.set_size_inches(width, height)

    # Improve readability
    for a in fig.axes:
        a.tick_params(axis="x", labelrotation=35, labelsize=10)
        a.tick_params(axis="y", labelsize=9)
        for line in getattr(a, "lines", []):
            line.set_alpha(0.35)
            line.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out / "parallel_coordinate_wide.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── Run the study ───────────────────────────
def run_optuna(objective_fn,epochs_num,opt_add,batch_size,SEED, SAVE_PLOTS_ROOT, CONFIG_ATTR, WHOLE_RES_XLSX,
               n_trials: int = 60, direction: str = "maximize") -> None:
        """Create a study, run optimisation, aggregate configs."""
       # _pruner  = optuna.pruners.MedianPruner(n_startup_trials=3)
        """ _pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=2)

        
        study = optuna.create_study(
           direction=direction,  
           sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, n_startup_trials=20),
           storage=f"sqlite:///{SAVE_PLOTS_ROOT / 'optuna_study.db'}",
           pruner=_pruner    # check every epoch
        ) 

        

        study.optimize(
            objective_fn,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True
           
        )

        # ─ summary ─
        print("\n═════════ Best trial ═════════")
        print("Trial # :", study.best_trial.number)
        print("Score   :", study.best_value)
        print("Params  :", study.best_params)
        

        # ─ aggregate every config dict ─
        records = {
            f"trial_{t.number:03d}": t.user_attrs[CONFIG_ATTR]
            for t in study.trials
            if t.state == TrialState.COMPLETE and CONFIG_ATTR in t.user_attrs
        }

        if records:
            (pd.DataFrame(records)
            .T.reset_index()
            .rename(columns={"index": "trial_name"})
            .to_excel(SAVE_PLOTS_ROOT/WHOLE_RES_XLSX, index=False))
            print(f"\n📝  Saved {len(records)} configs → {WHOLE_RES_XLSX}")
        else:
            print("\n⚠️   No completed trials produced configs – skipped aggregation.")

        # optional: full Optuna history
        study.trials_dataframe().to_csv(SAVE_PLOTS_ROOT/"optuna_results.csv", index=False)
        print("📊  Full trial history → optuna_results.csv")
        df = study.trials_dataframe(attrs=("number", "state", "value", "params", "datetime_start", "datetime_complete", "duration", "user_attrs"))
        df.to_excel(SAVE_PLOTS_ROOT/"trials.xlsx", index=False)
        combine_trial_reports(SAVE_PLOTS_ROOT)

      

         out = Path(SAVE_PLOTS_ROOT)
        out.mkdir(parents=True, exist_ok=True)

        ax = ovm.plot_optimization_history(study)
        fig = ax.figure
       
        
        ax.figure.savefig(out / "opt_history.png", dpi=200, bbox_inches="tight")
        plt.close(ax.figure)

        ax = ovm.plot_param_importances(study)
        
        ax.figure.savefig(out / "param_importances.png", dpi=200, bbox_inches="tight")
        plt.close(ax.figure)

        save_parallel_wide(study, SAVE_PLOTS_ROOT, inch_per_param=2.2, height=8, max_width_in=60) 
        tf.keras.backend.clear_session()
        gc.collect()"""
        storage = f"sqlite:///{f'{opt_add}/optuna_study.db'}"  # <— use scenario dir
        summaries = optuna.study.get_all_study_summaries(storage)
        if not summaries:
            raise RuntimeError(f"No studies found in storage: {storage}")

        # Pick the most recently started study (or replace with a fixed name you know)
        summaries = sorted(summaries, key=lambda s: (s.datetime_start or pd.Timestamp.min), reverse=True)
        study_name = summaries[0].study_name

        study = optuna.load_study(study_name=study_name, storage=storage)
        print("\n═════════ Loaded study ═════════")
        print("Study name:", study.study_name)
        print("Best value:", study.best_value)
        print("Best params:", study.best_params)

        # ✅ you were missing this line
        best_params = study.best_params
        
        print("\n═════════ Loaded study ═════════")
        print("Study name:", study.study_name)
        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
        fixed = optuna.trial.FixedTrial(best_params)
        final_tag = f"FINAL_best_{epochs_num}e_bs_{batch_size}"
        final_score = objective_fn(
            fixed,
            force_epochs=epochs_num,      # your requirement
            force_batch=batch_size,       # your requirement
            save_tag=final_tag    # keep outputs separate
        )
        print(f"\n✅ Final retrain ({final_tag}) optuna_score: {final_score:.6f}")
        print(f"📂 Artifacts under: {SAVE_PLOTS_ROOT / final_tag}")



if __name__ == "__main__":
        Seed=42
        print("Using fixed seed:", Seed)
    

        parser = argparse.ArgumentParser(
            description="Optuna tuning wrapper for `main`."
        )

        # run exactly 100 trials unless the user passes a different value
        parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials."
    )
        parser.add_argument(
                "--test-dataset",       # use dash in the CLI
                dest="test_dataset",    # store in test_dataset
                type=str,
                choices=["IOT", "Office", "TU", "all"],
                default="IOT",
                help="Dataset to use as test set: IOT | Office | TU | all",
            )
        parser.add_argument(
        "--direction",
        choices=["maximize", "minimize"],
        default="minimize",
        help="Optimisation direction."
    )

       
        #python Main.py --test-dataset IOT --trials 50
        #python Main.py --test-dataset Office --trials 50
        #python Main.py --test-dataset TU --trials 50
        #python Main.py --test-dataset all --trials 30 --direction minimize


        args = parser.parse_args()
        DATASET_CONFIG = {
        "TU":     (["IOT", "Office", "TU"],"TU",    6500),
        "Office": (["IOT", "TU", "Office"],"Office",8000),
        "IOT":    (["TU", "Office", "IOT"], "IOT",   7000),
        
        
    }
        
        if args.test_dataset != "all":
            dt_names,opt_add, train_size = DATASET_CONFIG[args.test_dataset]
            batch_size=[512,256]
            epochs_num=[100,120,150]
            print(f"running {args.test_dataset} as test dataset")
            for ep in epochs_num:
                for b in batch_size:
                    run_scenario(dt_names,epochs_num,b,opt_add, train_size, n_trials=args.trials, direction=args.direction, SEED=Seed)
        else:
            print("running all datasets as test dataset")
            batch_size=[512,256,128]
            epochs_num=[100,120,150]
            print(f"running {args.test_dataset} as test dataset, batch size: {batch_size}")
            for ep in epochs_num:
                for b in batch_size:
                    for name, (dt_names,opt_add, train_size) in DATASET_CONFIG.items():
                        print(f"--- {name} as test ---batch size: {b}")
                        run_scenario(dt_names,ep,b,opt_add, train_size, n_trials=args.trials, direction=args.direction, SEED=Seed)
