import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import optuna
import random
from datetime import datetime
import sys

# --- Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from src.loss import MinkowskiLoss
from data.dataset import ZarrMixupDataset
from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)

# --- Config Setup ---
config_path = os.path.join(parent_path, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Constants ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
BATCH_SIZE = config.get("BATCH_SIZE", 128)
NUM_EPOCHS = config.get("NUM_EPOCHS", 25)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 5)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

# --- Constraint Configuration ---
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.25)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.1)
CONSTRAINT_WARMUP_EPOCHS = config.get("CONSTRAINT_WARMUP_EPOCHS", 5)

EXPERIMENT_NAME = "GammaEmulator"


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_fraction=1.0):
    print(f"\n--- Initializing Datasets (Train Fraction: {data_fraction}) ---")

    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path))
    else:
        print(f"Scaler file not found at {scaler_path}. Using default value of 5.01.")
        scaler_val = 5.01

    train_zarr_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_dataset.zarr")

    train_dataset = ZarrMixupDataset(
        zarr_path=train_zarr_path,
        split="train",
        scaler_val=scaler_val,
        augment=True,
        include_original=True,
        include_mixup=True,
        subset_fraction=data_fraction,
    )

    val_subsample = data_fraction
    print(f"--- Initializing Validation (Subsample Fraction: {val_subsample}) ---")

    val_dataset = ZarrMixupDataset(
        zarr_path=train_zarr_path,
        split="validation",
        scaler_val=scaler_val,
        augment=False,
        include_original=True,
        include_mixup=False,
        subset_fraction=val_subsample,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.get("NUM_WORKERS", 8),
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 8),
        pin_memory=True,
    )

    return train_loader, val_loader


def run_training_session(hyperparams, train_loader, val_loader, args, trial=None):
    learning_rate = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]

    current_arch = args.arch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"T{trial.number}" if trial is not None else "SingleRun"
    run_name = f"{EXPERIMENT_NAME}_{current_arch}_{run_id}_{timestamp}"
    output_dir = os.path.join("final_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if current_arch == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif current_arch == "Lipschitz":
        model = LipschitzCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif current_arch == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        )
    else:
        raise ValueError(f"Unknown architecture: {current_arch}")

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    criterion = MinkowskiLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    log_file_path = os.path.join(output_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,"
            "train_loss_total,train_loss_A,train_loss_P,train_loss_B0,train_loss_B1,"
            "val_loss_total,val_loss_A,val_loss_P,val_loss_B0,val_loss_B1\n"
        )

    best_val_loss = float("inf")
    patience_counter = 0

    # Moved outside the loop to prevent redundant evaluation
    if current_arch in ["Constrained", "Lipschitz"]:
        w_a, w_p, w_b0, w_b1 = 3.0, 1.0, 1.5, 1.5
    else:
        w_a, w_p, w_b0, w_b1 = 1.0, 1.0, 1.0, 1.0

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_metrics = {
            k: 0.0 for k in ["total", "loss_A", "loss_P", "loss_B0", "loss_B1"]
        }

        for input_data, log_target_gamma, _, _ in train_loader:
            input_data = input_data.to(device)
            log_target_gamma = log_target_gamma.to(device)

            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            weighted_total_loss, loss_a, loss_p, loss_b0, loss_b1 = criterion(
                predicted_gamma_log, log_target_gamma, w_a, w_p, w_b0, w_b1
            )
            total_loss = torch.mean(weighted_total_loss)

            total_loss.backward()
            optimizer.step()

            running_metrics["total"] += total_loss.item()
            running_metrics["loss_A"] += torch.mean(loss_a).item()
            running_metrics["loss_P"] += torch.mean(loss_p).item()
            running_metrics["loss_B0"] += torch.mean(loss_b0).item()
            running_metrics["loss_B1"] += torch.mean(loss_b1).item()

        avg_train = {k: v / len(train_loader) for k, v in running_metrics.items()}

        model.eval()
        val_metrics = {
            k: 0.0 for k in ["total", "loss_A", "loss_P", "loss_B0", "loss_B1"]
        }

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in val_loader:
                input_data = input_data.to(device)
                log_target_gamma = log_target_gamma.to(device)

                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)

                weighted_total_loss, loss_a, loss_p, loss_b0, loss_b1 = criterion(
                    predicted_gamma_log, log_target_gamma, w_a, w_p, w_b0, w_b1
                )
                total_loss = torch.mean(weighted_total_loss)

                val_metrics["total"] += total_loss.item()
                val_metrics["loss_A"] += torch.mean(loss_a).item()
                val_metrics["loss_P"] += torch.mean(loss_p).item()
                val_metrics["loss_B0"] += torch.mean(loss_b0).item()
                val_metrics["loss_B1"] += torch.mean(loss_b1).item()

        avg_val = {k: v / len(val_loader) for k, v in val_metrics.items()}
        scheduler.step(avg_val["total"])

        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},"
                f"{avg_train['total']:.6f},{avg_train['loss_A']:.6f},{avg_train['loss_P']:.6f},"
                f"{avg_train['loss_B0']:.6f},{avg_train['loss_B1']:.6f},"
                f"{avg_val['total']:.6f},{avg_val['loss_A']:.6f},{avg_val['loss_P']:.6f},"
                f"{avg_val['loss_B0']:.6f},{avg_val['loss_B1']:.6f}\n"
            )

        if trial is not None:
            trial.report(avg_val["total"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if avg_val["total"] < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val["total"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "hyperparameters": hyperparams,
                    "arch": current_arch,
                },
                os.path.join(output_dir, "best_model_checkpoint.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if trial is None:
                    print(f"Early stopping triggered (Epoch {epoch}).")
                break

    return best_val_loss


def optuna_objective(trial, train_loader, val_loader, args):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hyperparams = {"lr": learning_rate, "weight_decay": weight_decay}
    return run_training_session(hyperparams, train_loader, val_loader, args, trial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="Baseline",
        choices=["Baseline", "Lipschitz", "Constrained"],
        help="Architecture: Baseline, Lipschitz, or Constrained",
    )
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"--- Emulator Training: {args.arch} ---")

    train_loader, val_loader = get_data_loaders(data_fraction=args.data_fraction)

    if args.optimize:
        print(f"Starting Optuna for {args.arch}...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda t: optuna_objective(t, train_loader, val_loader, args),
            n_trials=args.n_trials,
        )

        print("Best params:", study.best_trial.params)
        save_filename = os.path.join(
            parent_path, "training_params", f"best_params_{args.arch}.yaml"
        )
        with open(save_filename, "w") as f:
            yaml.dump(study.best_trial.params, f)
        print(f"Saved to {save_filename}")

        print("\n[INFO] Starting final training with best parameters...")
        best_params = study.best_trial.params
        run_training_session(
            {"lr": best_params["lr"], "weight_decay": best_params["weight_decay"]},
            train_loader,
            val_loader,
            args,
        )
    else:
        lr, wd = args.lr, args.wd
        if args.load_params and os.path.exists(args.load_params):
            with open(args.load_params, "r") as f:
                p = yaml.safe_load(f)
                lr = p.get("lr", lr)
                wd = p.get("weight_decay", wd)
            print(f"Loaded params: LR={lr}, WD={wd}")

        run_training_session(
            {"lr": lr, "weight_decay": wd}, train_loader, val_loader, args
        )


if __name__ == "__main__":
    main()
