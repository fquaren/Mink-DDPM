import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
import os
import numpy as np
import json
import time
import sys
import argparse
import optuna
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
from scipy.stats import wasserstein_distance

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

from models.SR.deterministic.unet import LogSpaceResidualUNet
from data.dataset import DeterministicSRDataset
from loss import AnalyticalMinkowskiLoss

# Hyperparameters
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_WORKERS = config["NUM_WORKERS"]
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PATIENCE = config.get("PATIENCE", 7)


# ------------------------------------------------------------------------------
# 1. Utilities
# ------------------------------------------------------------------------------
class DataDenormalizer:
    def __init__(self, max_val):
        self.max_val = max_val

    def unnormalize(self, x_norm):
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()
        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        return np.maximum(x_phys, 0.0)

    def unnormalize_torch(self, x_norm):
        max_val_tensor = torch.tensor(
            self.max_val, device=x_norm.device, dtype=x_norm.dtype
        )
        x_scaled = x_norm * max_val_tensor
        x_scaled = torch.clamp(x_scaled, max=50.0)
        x_phys = torch.expm1(x_scaled)
        return F.relu(x_phys)


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def state_dict(self):
        return {
            "patience": self.patience,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "val_loss_min": self.val_loss_min,
            "delta": self.delta,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]
        self.val_loss_min = state_dict["val_loss_min"]
        self.delta = state_dict["delta"]


def save_sample_images(model, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    X_batch, Y_batch, _ = next(iter(loader))
    X_batch = X_batch.to(device, non_blocking=True)
    Y_batch = Y_batch.to(device, non_blocking=True)

    X_phys = denormalizer.unnormalize_torch(X_batch[:, 0])
    total_precip = X_phys.sum(dim=(1, 2))
    max_idx = total_precip.argmax().item()
    X_sample = X_batch[max_idx : max_idx + 1]
    Y_sample = Y_batch[max_idx : max_idx + 1]

    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu"):
            x_generated = model(X_sample)

    X_norm = X_sample[:, 0].float().cpu().numpy()
    Y_norm = Y_sample[:, 0].float().cpu().numpy()
    Gen_norm = x_generated[:, 0].float().cpu().clamp(0.0, 1.0).numpy()

    img_in = denormalizer.unnormalize(X_norm)[0]
    img_target = denormalizer.unnormalize(Y_norm)[0]
    img_gen = denormalizer.unnormalize(Gen_norm)[0]

    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, f"sample_data_epoch_{epoch:03d}.npz")
    np.savez_compressed(
        data_path, img_in=img_in, img_target=img_target, img_gen=img_gen
    )

    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    def mask_low_values(img, threshold=0.1):
        masked = img.copy()
        masked[masked <= threshold] = np.nan
        return masked

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(
        np.nanmax([np.nanmax(img_in), np.nanmax(img_target), np.nanmax(img_gen)]), 1.0
    )
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    axs[0].imshow(mask_low_values(img_in), cmap=precip_cmap, norm=norm, origin="lower")
    axs[0].set_title(f"Input (LR) | Max: {np.nanmax(img_in):.2f}")
    axs[0].axis("off")

    axs[1].imshow(mask_low_values(img_gen), cmap=precip_cmap, norm=norm, origin="lower")
    axs[1].set_title(f"Generated (SR) | Max: {np.nanmax(img_gen):.2f}")
    axs[1].axis("off")

    im3 = axs[2].imshow(
        mask_low_values(img_target), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[2].set_title(f"Target (HR) | Max: {np.nanmax(img_target):.2f}")
    axs[2].axis("off")

    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"),
        bbox_inches="tight",
        dpi=100,
    )
    plt.close()


def compute_physical_metrics(real_batch, gen_batch, drizzle_threshold=0.1):
    real_batch = real_batch * (real_batch > drizzle_threshold).astype(float)
    gen_batch = gen_batch * (gen_batch > drizzle_threshold).astype(float)
    real_flat, gen_flat = real_batch.flatten(), gen_batch.flatten()
    if len(real_flat) == 0 or len(gen_flat) == 0:
        return {"wasserstein_dist": 0.0, "max_intensity_err": 0.0}
    wd = wasserstein_distance(real_flat, gen_flat)
    max_err = abs(np.max(real_flat) - np.max(gen_flat)) if len(real_flat) > 0 else 0
    return {"wasserstein_dist": wd, "max_intensity_err": max_err}


# ------------------------------------------------------------------------------
# 2. Training Loop & Optuna Objective
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modified Training Loop for Script 1 (Analytical Version)
# ------------------------------------------------------------------------------
def objective(trial, args, dem_stats, max_val):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_str = f"_trial_{trial.number}" if trial else ""
    out_dir = os.path.join(
        "sr_experiment_runs", f"UNet_AnalyticalBaseline{trial_str}_{timestamp}"
    )
    if not args.tune:
        os.makedirs(out_dir, exist_ok=True)

    denormalizer = DataDenormalizer(max_val)

    if trial:
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        lr = getattr(args, "lr", 1e-3)
        weight_decay = getattr(args, "weight_decay", 1e-4)

    epochs = args.optuna_epochs if args.tune else config["NUM_EPOCHS"]

    # This acts as w_max in the cosine schedule
    GEOMETRIC_WEIGHT = (
        args.weight_geom
        if args.weight_geom is not None
        else config.get("MINKOWSKI_TARGET_WEIGHT", 1.0)
    )

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
    val_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=max_val,
        split="validation",
        load_in_ram=False,
        data_percentage=args.data_percentage,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        ),
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    criterion_mse = nn.MSELoss()
    criterion_geom = AnalyticalMinkowskiLoss(thresholds=QUANTILE_LEVELS).to(device)

    # Removed dynamic weights from AdamW parameters
    optimizer = optim.AdamW(
        [{"params": model.parameters()}], lr=lr, weight_decay=weight_decay
    )

    scheduler_patience = max(1, PATIENCE // 2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    scaler_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    max_val_tensor = torch.tensor(max_val, device=device, dtype=torch.float32)

    best_val_loss = float("inf")

    # Replaced log_var keys with geom_weight
    history = {
        "epoch": [],
        "train_loss_total": [],
        "train_loss_mse": [],
        "train_loss_geom": [],
        "val_loss_total": [],
        "val_loss_mse": [],
        "val_loss_geom": [],
        "geom_weight": [],
        "learning_rate": [],
    }

    start_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume) and not trial:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler_enabled:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        early_stopper.load_state_dict(checkpoint["early_stop_state"])
        history = checkpoint.get("history", history)

    for epoch in range(start_epoch, epochs):
        current_anneal_factor = max(0.05, np.exp(-3.0 * epoch / epochs))
        current_lr_value = optimizer.param_groups[0]["lr"]

        # Calculate cosine annealing schedule for current epoch
        w_min = 0.0
        current_geom_weight = w_min + 0.5 * (GEOMETRIC_WEIGHT - w_min) * (
            1 + np.cos(np.pi * epoch / epochs)
        )

        model.train()
        running_total_loss, running_mse_loss, running_geom_loss = 0.0, 0.0, 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Trial {trial.number if trial else ''} | Epoch {epoch+1}/{epochs} [W_geom={current_geom_weight:.4f}]",
            mininterval=15.0,
        )
        for X, Y_norm, Y_gamma_log in pbar:
            X, Y_norm, Y_gamma_log = (
                X.to(device, non_blocking=True),
                Y_norm.to(device, non_blocking=True),
                Y_gamma_log.to(device, non_blocking=True),
            )

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda" if device.type == "cuda" else "cpu", enabled=scaler_enabled
            ):
                Y_pred_norm = model(X)
                loss_mse = criterion_mse(Y_pred_norm, Y_norm)
                loss_geom = torch.tensor(0.0, device=device)

                if current_geom_weight > 0.0:
                    pred_scaled = torch.clamp(
                        Y_pred_norm[:, 0:1] * max_val_tensor, max=7.0
                    )
                    pred_phys = F.relu(torch.expm1(pred_scaled))
                    loss_geom = criterion_geom(
                        pred_phys, Y_gamma_log, anneal_factor=current_anneal_factor
                    )

                # Linear combination using scheduled weight
                total_loss = loss_mse + current_geom_weight * loss_geom

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += total_loss.item()
            running_mse_loss += loss_mse.item()
            running_geom_loss += (
                loss_geom.item() if isinstance(loss_geom, torch.Tensor) else loss_geom
            )

            if not args.tune:
                pbar.set_postfix(
                    Tot=f"{total_loss.item():.3f}",
                    MSE=f"{loss_mse.item():.4f}",
                    Geom=f"{loss_geom.item() if isinstance(loss_geom, torch.Tensor) else loss_geom:.4f}",
                )

        avg_train_tot = running_total_loss / len(train_loader)
        avg_train_mse = running_mse_loss / len(train_loader)
        avg_train_geom = running_geom_loss / len(train_loader)

        model.eval()
        val_total_loss, val_mse_loss, val_geom_loss = 0.0, 0.0, 0.0

        with torch.no_grad():
            for X, Y_norm, Y_gamma_log in val_loader:
                X, Y_norm, Y_gamma_log = (
                    X.to(device, non_blocking=True),
                    Y_norm.to(device, non_blocking=True),
                    Y_gamma_log.to(device, non_blocking=True),
                )
                with torch.amp.autocast(
                    "cuda" if device.type == "cuda" else "cpu", enabled=scaler_enabled
                ):
                    Y_pred_norm = model(X)
                    loss_m = criterion_mse(Y_pred_norm, Y_norm)
                    loss_g = torch.tensor(0.0, device=device)

                    if current_geom_weight > 0.0:
                        pred_scaled = torch.clamp(
                            Y_pred_norm[:, 0:1] * max_val_tensor, max=7.0
                        )
                        pred_phys = F.relu(torch.expm1(pred_scaled))
                        loss_g = criterion_geom(
                            pred_phys, Y_gamma_log, anneal_factor=current_anneal_factor
                        )

                    loss_t = loss_m + current_geom_weight * loss_g

                val_total_loss += loss_t.item()
                val_mse_loss += loss_m.item()
                val_geom_loss += (
                    loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g
                )

        avg_val_tot = val_total_loss / len(val_loader)
        avg_val_mse = val_mse_loss / len(val_loader)
        avg_val_geom = val_geom_loss / len(val_loader)

        scheduler.step(avg_val_tot)

        history["epoch"].append(epoch + 1)
        history["train_loss_total"].append(avg_train_tot)
        history["train_loss_mse"].append(avg_train_mse)
        history["train_loss_geom"].append(avg_train_geom)
        history["val_loss_total"].append(avg_val_tot)
        history["val_loss_mse"].append(avg_val_mse)
        history["val_loss_geom"].append(avg_val_geom)
        history["geom_weight"].append(current_geom_weight)
        history["learning_rate"].append(current_lr_value)

        # Check if current validation loss is the lowest observed
        # Note: We keep avg_val_tot for saving the 'best' weights as it includes the physics constraint
        is_best = avg_val_tot < best_val_loss
        best_val_loss = min(best_val_loss, avg_val_tot)

        if not args.tune:
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler_enabled else {},
                "scheduler_state_dict": scheduler.state_dict(),
                "early_stop_state": early_stopper.state_dict(),
                "history": history,
            }
            torch.save(checkpoint_state, os.path.join(out_dir, "unet_latest.pth"))
            if is_best:
                torch.save(checkpoint_state, os.path.join(out_dir, "unet_best.pth"))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                save_sample_images(
                    model, val_loader, device, out_dir, epoch + 1, denormalizer
                )

        # --- Early Stopping & Reporting (Modified to MSE) ---
        if trial:
            trial.report(avg_val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Trigger early stopping based on validation MSE
        if early_stopper(avg_val_mse):
            if not args.tune:
                print(
                    f"Early stopping triggered based on validation MSE at epoch {epoch+1}."
                )
            break

    if not args.tune:
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(out_dir, "unet_ana_final.pth"),
        )

    return best_val_loss


# ------------------------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train deterministic UNet with Minkowski Loss."
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.")
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
        "--weight_geom",
        type=float,
        default=None,
        help="Static scaling weight for the geometric loss. Overrides config value.",
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

        final_loss = objective(None, args, dem_stats, max_val)
        print(f"Final best validation loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
