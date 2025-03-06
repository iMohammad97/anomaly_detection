# Suppose we have a custom LSTMAE model in anomaly_models/AELSTMAE.py
import glob
import os
import shutil

from anomaly_models.AE_Classes.AE_Class import LSTMAutoencoder

# The new functional mixture-of-experts utilities
from anomaly_models.mixture_of_experts import (
    create_moe,
    train_moe,
    evaluate_moe,
    plot_expert1,
    plot_final_moe
)

from anomaly_models.mixture_of_experts.MixtureOfExpertsLSTMAutoencoder import MixtureOfExpertsLSTMAutoencoder, \
    evaluate_model_and_save_results


def main():
    # 1) Load data
    X_train = ...
    X_test  = ...
    Y_test  = ...

    # 2) Create MoE
    moe = create_moe(
        ExpertClass=LSTMAutoencoder,
        train_data=X_train,
        test_data=X_test,
        labels=Y_test,
        timesteps=128,
        features=1,
        loss_type='mse',    # or 'max_diff_loss'
        threshold_sigma=2.0,
        # plus any extra arguments needed by LSTMAutoencoder, e.g. latent_dim=32
        latent_dim=32,
        lstm_units=64
    )

    # 3) Train
    train_moe(moe, epochs=50, batch_size=32, patience=10, optimizer='adam')

    # 4) Evaluate
    evaluate_moe(moe, batch_size=32)

    # 5) Plot final gating
    plot_final_moe(moe, save_path='final_moe.html')

    # 6) Do your metrics, etc. (moe['final_preds'], moe['final_scores'])
    # ...

def my_sample_main():
    run_list = ["2", "37", "45", "23", "43", "21", "4", "8", "11", "20", "50", "51", "69", "31", "6"]
    directory = "/kaggle/working/anomaly_detection/UCR/UCR2_preprocessed/"

    HYPER_PARAMS = {
        "TIMESTEPS": 128,
        "LATENT_DIM": 32,
        "STEP_SIZE": 10,
        "EPOCHS": 200,
        "LOSS": "mse"
    }

    # Prepare results directory
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")

    train_files = sorted(glob.glob(os.path.join(directory, "*_train.npy")))

    for ts in train_files:
        ts_id = ts.split("/")[-1].split("_")[0]
        if ts_id not in run_list:
            continue
        print(f"\n===========================\nStarting w/ ts_id = {ts_id}\n===========================")

        # Identify test/label files
        train_file = ts
        test_file = ts.replace("_train.npy", "_test.npy")
        label_file = ts.replace("_train.npy", "_labels.npy")

        if (not os.path.exists(test_file)) or (not os.path.exists(label_file)):
            print(f"Skipping {ts_id}, missing test/label files.")
            continue

        # Load data
        X_train = np.load(train_file)
        X_test = np.load(test_file)
        Y_test = np.load(label_file)

        # Instantiate MixtureOfExperts
        mixture_model = MixtureOfExpertsLSTMAutoencoder(
            train_data=X_train,
            test_data=X_test,
            labels=Y_test,
            timesteps=HYPER_PARAMS["TIMESTEPS"],
            features=1,  # assuming univariate
            latent_dim=HYPER_PARAMS["LATENT_DIM"],
            lstm_units=64,
            step_size=HYPER_PARAMS["STEP_SIZE"],
            threshold_sigma=2.0,
            seed=0,
            loss='max_diff_loss' if HYPER_PARAMS["LOSS"] == "max_diff_loss" else 'mse'
        )

        # Train
        mixture_model.train(
            epochs=HYPER_PARAMS["EPOCHS"],
            batch_size=32,
            patience=10,
            optimizer='adam'
        )

        # Evaluate (final gating approach)
        mixture_model.evaluate(batch_size=32)

        # Plot Expert1 alone
        plot_e1_path = f"results/moe_{ts_id}_expert1.html"
        mixture_model.plot_expert1_results(save_path=plot_e1_path)

        # Plot Expert2 alone
        plot_e2_path = f"results/moe_{ts_id}_expert2.html"
        mixture_model.plot_expert2_results(save_path=plot_e2_path)

        # Plot Final gating-based reconstruction
        plot_final_path = f"results/moe_{ts_id}_final.html"
        mixture_model.plot_moe_final_results(save_path=plot_final_path)

        # Save the 2 sub-models
        moe_dir = f"results/moe_{ts_id}"
        mixture_model.save_models(moe_dir)

        # Evaluate and Save Results to CSV
        results_csv = f"results/moe_{ts_id}_results.csv"
        evaluate_model_and_save_results(
            model_class=MixtureOfExpertsLSTMAutoencoder,
            model_path="(no single path, using moe_dir)",  # not used for MoE
            results_csv_path=results_csv,
            train_path=train_file,
            test_path=test_file,
            label_path=label_file,
            loss_type='mse',
            moe_dir_path=moe_dir
        )

    print("\nAll done!")