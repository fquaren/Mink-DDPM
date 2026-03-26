import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import yaml
import os
import numpy as np
import json
import time
import sys
import argparse
import optuna

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

from models.SR.deterministic.unet import LogSpaceResidualUNet
from data.dataset import DeterministicSRDataset

# Hyperparameters
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
GEOMETRIC_WEIGHT = config.get("MINKOWSKI_TARGET_WEIGHT", 1.0)


# ------------------------------------------------------------------------------
# 1. Analytical Loss Module
# ------------------------------------------------------------------------------
class AnalyticalMinkowskiLoss(nn.Module):
    def __init__(self, thresholds, init_factor=0.1, min_temp=1e-3):
        super().__init__()
        self.register_buffer(
            "thresholds", torch.tensor(thresholds, dtype=torch.float32)
        )

        base_temps = np.maximum(np.array(thresholds) * init_factor, min_temp)
        self.register_buffer(
            "base_temps", torch.tensor(base_temps, dtype=torch.float32)
        )

    def forward(self, pred_phys, target_gamma_log, anneal_factor=1.0):
        """
        pred_phys: Tensor [B, 1, H, W] in physical space.
        target_gamma_log: Tensor [B, 4, Q] containing log1p-transformed targets.
        """
        B, _, H, W = pred_phys.shape
        areas, perimeters, eulers = [], [], []

        for q_idx, thresh in enumerate(self.thresholds):
            current_temp = self.base_temps[q_idx] * anneal_factor

            p = torch.sigmoid((pred_phys - thresh) / current_temp)

            area = torch.sum(p, dim=(1, 2, 3))

            dx = p[:, :, :, 1:] - p[:, :, :, :-1]
            dy = p[:, :, 1:, :] - p[:, :, :-1, :]
            perimeter = torch.sum(
                torch.sqrt(dx[:, :, :-1, :] ** 2 + dy[:, :, :, :-1] ** 2 + 1e-8),
                dim=(1, 2, 3),
            )

            V = torch.sum(p, dim=(1, 2, 3))
            E_x = torch.sum(p[:, :, :, :-1] * p[:, :, :, 1:], dim=(1, 2, 3))
            E_y = torch.sum(p[:, :, :-1, :] * p[:, :, 1:, :], dim=(1, 2, 3))
            F = torch.sum(
                p[:, :, :-1, :-1]
                * p[:, :, :-1, 1:]
                * p[:, :, 1:, :-1]
                * p[:, :, 1:, 1:],
                dim=(1, 2, 3),
            )
            euler = V - E_x - E_y + F

            areas.append(area)
            perimeters.append(perimeter)
            eulers.append(euler)

        pred_gamma_phys = torch.stack(
            [
                torch.stack(areas, dim=1),
                torch.stack(perimeters, dim=1),
                torch.stack(eulers, dim=1),
            ],
            dim=1,
        )

        pred_gamma_log = torch.sign(pred_gamma_phys) * torch.log1p(
            torch.abs(pred_gamma_phys)
        )

        # Exact inverse transform to linear space
        target_raw = torch.sign(target_gamma_log) * torch.expm1(
            torch.abs(target_gamma_log)
        )

        target_area = target_raw[:, 0, :]
        target_perim = target_raw[:, 1, :]
        target_euler = target_raw[:, 2, :] - target_raw[:, 3, :]

        target_gamma_processed = torch.stack(
            [target_area, target_perim, target_euler], dim=1
        )

        # Re-apply log transform to the combined tensor
        target_gamma_log_processed = torch.sign(target_gamma_processed) * torch.log1p(
            torch.abs(target_gamma_processed)
        )

        abs_diff = torch.abs(pred_gamma_log - target_gamma_log_processed.float())
        dist = torch.trapezoid(abs_diff, self.thresholds, dim=2)

        return dist.sum(dim=1).mean()


# ------------------------------------------------------------------------------
# 2. Training Loop & Optuna Objective
# ------------------------------------------------------------------------------
def objective(trial, args, dem_stats, max_val):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Optuna suggestions
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs = args.optuna_epochs if args.tune else config["NUM_EPOCHS"]

    train_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=max_val,
        split="train",
        load_in_ram=False,
        data_percentage=args.data_percentage,
    )

    # Dataset subsetting
    dataset_size = len(train_dataset)
    subset_size = int(dataset_size * args.data_percentage / 100.0)
    indices = np.arange(subset_size)

    subset_dataset = Subset(train_dataset, indices)
    subset_weights = train_dataset.sample_weights[indices]

    sampler = WeightedRandomSampler(
        weights=subset_weights, num_samples=len(subset_dataset), replacement=True
    )

    train_loader = DataLoader(
        dataset=subset_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    criterion_geom = AnalyticalMinkowskiLoss(thresholds=QUANTILE_LEVELS).to(device)
    criterion_mse = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    max_val_tensor = torch.tensor(max_val, device=device, dtype=torch.float32)

    final_loss = float("inf")

    for epoch in range(epochs):
        current_anneal_factor = max(0.05, np.exp(-3.0 * epoch / epochs))

        model.train()
        running_total_loss = 0.0

        pbar = tqdm(
            train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs}"
        )
        for X, Y_norm, Y_gamma_log in pbar:
            X, Y_norm, Y_gamma_log = (
                X.to(device),
                Y_norm.to(device),
                Y_gamma_log.to(device),
            )

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda"),
            ):
                Y_pred_norm = model(X)

                loss_mse = criterion_mse(Y_pred_norm, Y_norm)

                pred_scaled = torch.clamp(Y_pred_norm[:, 0:1] * max_val_tensor, max=7.0)
                pred_phys = F.relu(torch.expm1(pred_scaled))

                loss_geom = criterion_geom(
                    pred_phys, Y_gamma_log, anneal_factor=current_anneal_factor
                )
                total_loss = loss_mse + (GEOMETRIC_WEIGHT * loss_geom)

            scaler.scale(total_loss).backward()

            # Gradient clipping to prevent explosion from exponentiated space
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_total_loss += total_loss.item()
            pbar.set_postfix(
                Total=f"{total_loss.item():.4f}", MSE=f"{loss_mse.item():.4f}"
            )

        final_loss = running_total_loss / len(train_loader)

        # Report intermediate values to Optuna for pruning unpromising trials
        trial.report(final_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if not args.tune:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(
            "sr_experiment_runs", f"UNet_AnalyticalBaseline_{timestamp}"
        )
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(out_dir, "unet_ana_final.pth"),
        )

    return final_loss


# ------------------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train deterministic UNet with Minkowski Loss."
    )
    parser.add_argument(
        "--data_percentage",
        type=float,
        default=100.0,
        help="Fraction of dataset to use for training (0.0 to 100.0).",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run Optuna hyperparameter tuning."
    )
    parser.add_argument(
        "--params_path", type=str, default=os.path.join(parent_path, "unet_params.yaml")
    )
    parser.add_argument(
        "--optuna_trials", type=int, default=10, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--optuna_epochs", type=int, default=3, help="Epochs per Optuna trial."
    )
    args = parser.parse_args()

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    max_val = float(
        np.load(os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")).item()
    )

    if args.tune:
        print(f"--- Starting Optuna Tuning with {args.data_percentage}% of data ---")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, args, dem_stats, max_val),
            n_trials=args.optuna_trials,
        )

        print("\n--- Optuna Tuning Complete ---")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        print("Saving best hyperparameters to unet_ana_params.yaml")
        with open(os.path.join(parent_path, "unet_ana_params.yaml"), "w") as f:
            yaml.dump(study.best_params, f)
    else:
        print(
            f"--- Starting Standard Training with {args.data_percentage}% of data ---"
        )
        best_params_path = os.path.join(parent_path, "unet_ana_params.yaml")
        if os.path.exists(best_params_path):
            with open(best_params_path, "r") as f:
                best_params = yaml.safe_load(f)
            print(f"Loaded best hyperparameters from {best_params_path}")
            for key, value in best_params.items():
                setattr(args, key, value)
        else:
            print(
                f"No best hyperparameters found at {best_params_path}. Using defaults."
            )

        final_loss = objective(optuna.trial.FixedTrial({}), args, dem_stats, max_val)
        print(f"Final training loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
