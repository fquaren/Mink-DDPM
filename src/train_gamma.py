import argparse
import tqdm
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import optuna
import random
from datetime import datetime
import sys
import logging

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


# --- Logger Setup ---
def setup_base_logger():
    logger = logging.getLogger("emulator_logger")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = logging.getLogger("emulator_logger")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_fraction=1.0):
    logger.info(f"--- Initializing Datasets (Train Fraction: {data_fraction}) ---")

    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path).item())
    else:
        logger.warning(
            f"Scaler file not found at {scaler_path}. Using default value of 5.01."
        )
        scaler_val = 5.01

    train_zarr_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_dataset.zarr")

    # Dataset internally subsets itself
    train_dataset = ZarrMixupDataset(
        zarr_path=train_zarr_path,
        split="train",
        scaler_val=scaler_val,
        augment=True,
        include_original=True,
        include_mixup=True,
        subset_fraction=data_fraction,
        load_in_ram=False,
    )

    logger.info(
        f"--- Initializing Validation (Subsample Fraction: {data_fraction}) ---"
    )

    val_dataset = ZarrMixupDataset(
        zarr_path=train_zarr_path,
        split="validation",
        scaler_val=scaler_val,
        augment=False,
        include_original=True,
        include_mixup=False,
        subset_fraction=data_fraction,
        load_in_ram=False,
    )

    # Directly use the datasets in the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config.get("NUM_WORKERS", 8),
        pin_memory=True,
        persistent_workers=False,
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
    output_dir = os.path.join("ci26_revision_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Attach file handler for this specific run
    fh = logging.FileHandler(os.path.join(output_dir, "process.log"))
    print(f"Logging to: {os.path.join(output_dir, 'process.log')}")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Initialized output directory: {output_dir}")
    logger.info(f"Using device: {device}")

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
        logger.error(f"Unknown architecture: {current_arch}")
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
            "train_loss_total,train_loss_A,train_loss_P,train_loss_B0,"
            "val_loss_total,val_loss_A,val_loss_P,val_loss_B0\n"
        )

    best_val_loss = float("inf")
    patience_counter = 0

    if current_arch in ["Constrained", "Lipschitz", "Baseline"]:
        w_a, w_p, w_b0 = 3.0, 1.0, 3.5
    else:
        print(
            f"Warning: Unrecognized architecture {current_arch}. Using default loss weights."
        )
        w_a, w_p, w_b0 = 1.0, 1.0, 1.0

    logger.info(f"Starting training loop. Total epochs: {NUM_EPOCHS}")

    for epoch in tqdm.tqdm(range(NUM_EPOCHS), desc="Training"):

        model.train()
        running_metrics = {k: 0.0 for k in ["total", "loss_A", "loss_P", "loss_B0"]}

        for input_data, log_target_gamma, _, _ in tqdm.tqdm(
            train_loader, desc="Training Batches"
        ):
            input_data = input_data.to(device)
            # Ensure target tensor aligns with the 3 outputs: A, P, B0
            log_target_gamma = log_target_gamma[:, :3, :].to(device)

            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            weighted_total_loss, loss_a, loss_p, loss_b0 = criterion(
                predicted_gamma_log, log_target_gamma, w_a, w_p, w_b0
            )
            total_loss = torch.mean(weighted_total_loss)

            total_loss.backward()
            optimizer.step()

            running_metrics["total"] += total_loss.item()
            running_metrics["loss_A"] += torch.mean(loss_a).item()
            running_metrics["loss_P"] += torch.mean(loss_p).item()
            running_metrics["loss_B0"] += torch.mean(loss_b0).item()

        avg_train = {k: v / len(train_loader) for k, v in running_metrics.items()}

        model.eval()
        val_metrics = {k: 0.0 for k in ["total", "loss_A", "loss_P", "loss_B0"]}

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in val_loader:
                input_data = input_data.to(device)
                # Ensure target tensor aligns with the 3 outputs: A, P, B0
                log_target_gamma = log_target_gamma[:, :3, :].to(device)

                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)

                weighted_total_loss, loss_a, loss_p, loss_b0 = criterion(
                    predicted_gamma_log, log_target_gamma, w_a, w_p, w_b0
                )
                total_loss = torch.mean(weighted_total_loss)

                val_metrics["total"] += total_loss.item()
                val_metrics["loss_A"] += torch.mean(loss_a).item()
                val_metrics["loss_P"] += torch.mean(loss_p).item()
                val_metrics["loss_B0"] += torch.mean(loss_b0).item()

        avg_val = {k: v / len(val_loader) for k, v in val_metrics.items()}
        scheduler.step(avg_val["total"])

        logger.debug(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train['total']:.4f} | Val Loss: {avg_val['total']:.4f}"
        )

        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},"
                f"{avg_train['total']:.6f},{avg_train['loss_A']:.6f},{avg_train['loss_P']:.6f},{avg_train['loss_B0']:.6f},"
                f"{avg_val['total']:.6f},{avg_val['loss_A']:.6f},{avg_val['loss_P']:.6f},{avg_val['loss_B0']:.6f}\n"
            )

        if trial is not None:
            trial.report(avg_val["total"], epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}.")
                logger.removeHandler(fh)
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
            logger.debug(f"New best validation loss: {best_val_loss:.6f}. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if trial is None:
                    logger.info(f"Early stopping triggered at Epoch {epoch}.")
                break

    logger.info(f"Session complete. Best Validation Loss: {best_val_loss:.6f}")

    # Detach file handler to prevent memory leaks across serial optuna runs
    logger.removeHandler(fh)
    fh.close()

    return best_val_loss


def optuna_objective(trial, train_loader, val_loader, args):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hyperparams = {"lr": learning_rate, "weight_decay": weight_decay}
    return run_training_session(hyperparams, train_loader, val_loader, args, trial)


def main():
    setup_base_logger()

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

    logger.info(f"--- Emulator Training: {args.arch} ---")

    train_loader, val_loader = get_data_loaders(data_fraction=args.data_fraction)

    if args.optimize:
        logger.info(f"Starting Optuna for {args.arch}...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda t: optuna_objective(t, train_loader, val_loader, args),
            n_trials=args.n_trials,
        )

        logger.info(f"Best params: {study.best_trial.params}")
        save_filename = os.path.join(
            parent_path, "training_params", f"best_params_{args.arch}.yaml"
        )
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        with open(save_filename, "w") as f:
            yaml.dump(study.best_trial.params, f)
        logger.info(f"Saved to {save_filename}")

        logger.info("\n[INFO] Starting final training with best parameters...")
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
            logger.info(f"Loaded params: LR={lr}, WD={wd}")

        run_training_session(
            {"lr": lr, "weight_decay": wd}, train_loader, val_loader, args
        )


if __name__ == "__main__":
    main()
