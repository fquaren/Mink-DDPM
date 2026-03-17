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
import matplotlib.pyplot as plt
import argparse
from scipy.stats import wasserstein_distance
import copy
import matplotlib.colors as mcolors
import sys
import optuna

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Project imports
from models.SR.deterministic.unet import LogSpaceResidualUNet
from data.dataset import DeterministicSRDataset
from src.loss import MinkowskiLoss
from src.utils import load_emulator

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = config["BATCH_SIZE"]
LR = config["LEARNING_RATE"]
WD = config["WEIGHT_DECAY"]
EPOCHS = config["NUM_EPOCHS"]
PATIENCE = config["PATIENCE"]
NUM_WORKERS = config["NUM_WORKERS"]
EXPERIMENT_NAME = "UNet_SR_Minkowski"
N_QUANTILES = len(config["QUANTILE_LEVELS"])

# --- Minkowski Loss Configuration ---
GEOMETRIC_TARGET_WEIGHT = config.get("MINKOWSKI_TARGET_WEIGHT", 0.0)
GEOMETRIC_WARMUP_EPOCHS = config.get("MINKOWSKI_WARMUP_EPOCHS", 5)
TRUST_TAU = config.get("TRUST_TAU", 0.1)
EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", "checkpoints/emulator_best.pth")


# ------------------------------------------------------------------------------
# 1. Helper Classes & Functions
# ------------------------------------------------------------------------------
class DataDenormalizer:
    def __init__(self, stats_path):
        try:
            data = np.load(stats_path)
            self.max_val = float(data.item())
            print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")
        except FileNotFoundError:
            print(
                f"Warning: Scaling stats not found at {stats_path}. Defaulting to 1.0."
            )
            self.max_val = 1.0

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
        self.val_loss_min = np.Inf
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
        self.val_loss_min = np.Inf

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


def compute_geometric_loss_component(
    emulator,
    criterion,
    denormalizer,
    Y_pred_norm,
    Y_clean_norm,
    Y_gamma_log,
    compute_trust=True,
):
    loss_geom = torch.tensor(0.0, device=Y_pred_norm.device)
    avg_trust_val = 1.0
    trust_weights = torch.ones(Y_pred_norm.shape[0], device=Y_pred_norm.device)

    if compute_trust:
        with torch.no_grad():
            Y_phys = denormalizer.unnormalize_torch(Y_clean_norm)
            Y_phys = Y_phys * (Y_phys > 0.1).float()
            gamma_truth_phys = emulator(Y_phys)
            gamma_truth_log_pred = torch.log1p(gamma_truth_phys)
            diff_trust = (gamma_truth_log_pred - Y_gamma_log).float()
            emu_error_sq = diff_trust.pow(2).mean(dim=(1, 2))
            trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_sq)
            avg_trust_val = trust_weights.mean().item()

    pred_phys = denormalizer.unnormalize_torch(Y_pred_norm)
    pred_phys = pred_phys * (pred_phys > 0.1).float()
    pred_gamma_phys = emulator(pred_phys)
    pred_gamma_log = torch.log1p(pred_gamma_phys)
    raw_geom_loss, _, _, _ = criterion(pred_gamma_log, Y_gamma_log)
    weight_factor = trust_weights.view(-1)
    weighted_loss = raw_geom_loss * weight_factor
    loss_geom = weighted_loss.mean()

    return loss_geom, avg_trust_val


def save_sample_images(model, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    selected_X, selected_Y = [], []
    dry_count, wet_count = 0, 0
    target_dry, target_wet = 1, 4
    drizzle_threshold = 0.1

    for X_batch, Y_batch, _ in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        X_phys = denormalizer.unnormalize_torch(X_batch[:, 0])
        max_precip = X_phys.amax(dim=(1, 2))

        for i in range(X_batch.size(0)):
            if max_precip[i] <= drizzle_threshold and dry_count < target_dry:
                selected_X.append(X_batch[i : i + 1])
                selected_Y.append(Y_batch[i : i + 1])
                dry_count += 1
            elif max_precip[i] > drizzle_threshold and wet_count < target_wet:
                selected_X.append(X_batch[i : i + 1])
                selected_Y.append(Y_batch[i : i + 1])
                wet_count += 1

            if dry_count == target_dry and wet_count == target_wet:
                break
        if dry_count == target_dry and wet_count == target_wet:
            break

    if not selected_X:
        return

    X_sample = torch.cat(selected_X, dim=0)
    Y_sample = torch.cat(selected_Y, dim=0)
    n_samples = X_sample.size(0)

    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in device else "cpu"):
            x_generated = model(X_sample)

    X_norm = X_sample[:, 0].float().cpu().numpy()
    Y_norm = Y_sample[:, 0].float().cpu().numpy()
    Gen_norm = x_generated[:, 0].float().cpu().clamp(0.0, 1.0).numpy()

    X_phys = denormalizer.unnormalize(X_norm)
    Y_phys = denormalizer.unnormalize(Y_norm)
    Gen_phys = denormalizer.unnormalize(Gen_norm)

    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    def mask_low_values(img, threshold=0.1):
        masked = img.copy()
        masked[masked <= threshold] = np.nan
        return masked

    _, axs = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples), squeeze=False)

    for i in range(n_samples):
        img_in, img_target, img_gen = X_phys[i], Y_phys[i], Gen_phys[i]
        vmax = max(
            np.nanmax([np.nanmax(img_in), np.nanmax(img_target), np.nanmax(img_gen)]),
            1.0,
        )
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        axs[i, 0].imshow(
            mask_low_values(img_in), cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 0].set_title(f"Input (LR) | Max: {np.nanmax(img_in):.2f}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(
            mask_low_values(img_gen), cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 1].set_title(f"Generated (SR) | Max: {np.nanmax(img_gen):.2f}")
        axs[i, 1].axis("off")

        im3 = axs[i, 2].imshow(
            mask_low_values(img_target), cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 2].set_title(f"Target (HR) | Max: {np.nanmax(img_target):.2f}")
        axs[i, 2].axis("off")
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
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


def run_training(args, trial=None):
    subset_fraction = args.data_percentage / 100.0

    if torch.cuda.is_available():
        device = "cuda"
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_str = f"_trial_{trial.number}" if trial else ""
    run_name = f"{EXPERIMENT_NAME}{trial_str}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    denormalizer = DataDenormalizer(
        os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    )

    train_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=denormalizer.max_val,
        split="train",
        subset_fraction=subset_fraction,
    )
    val_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=denormalizer.max_val,
        split="validation",
        subset_fraction=subset_fraction,
    )

    # Initialize the sampler using the computed weights
    # Replacement strategy ensures the epoch length matches the dataset size,
    # while over-sampling wet instances and under-sampling dry instances.
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    # Create the dataloader. IMPORTANT: shuffle must be False when using a custom sampler.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    geometric_criterion = MinkowskiLoss(quantile_levels=config["QUANTILE_LEVELS"]).to(
        device
    )
    emulator = load_emulator(EMULATOR_PATH, config, device)
    emulator.eval()
    for param in emulator.parameters():
        param.requires_grad = False

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)

    # --- Hyperparameter Sampling ---
    if trial:
        current_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        current_wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        current_lr = LR
        current_wd = WD

    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)
    mse = nn.MSELoss()

    # --- Scheduler Implementation ---
    scheduler_patience = max(1, PATIENCE // 2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
    )

    scaler_enabled = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_geom": [],
        "val_geom": [],
        "avg_trust": [],
        "geom_weight": [],
        "learning_rate": [],
    }

    start_epoch = 0
    geometric_phase = False
    geometric_start_epoch = None
    best_wd_metric = float("inf")

    if args.resume and os.path.isfile(args.resume) and not trial:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler_enabled:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        geometric_phase = checkpoint.get("geometric_phase", False)
        geometric_start_epoch = checkpoint.get("geometric_start_epoch", None)
        early_stopper.load_state_dict(checkpoint["early_stop_state"])
        history = checkpoint.get("history", history)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        current_geom_weight = 0.0
        if geometric_phase and geometric_start_epoch is not None:
            progress = min(
                1.0,
                max(
                    0.0,
                    (epoch - geometric_start_epoch) / float(GEOMETRIC_WARMUP_EPOCHS),
                ),
            )
            current_geom_weight = GEOMETRIC_TARGET_WEIGHT * progress

        current_lr_value = optimizer.param_groups[0]["lr"]
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [W={current_geom_weight:.3f}, LR={current_lr_value:.2e}]",
            disable=bool(trial),
        )
        running_loss, running_geom, running_trust = 0.0, 0.0, 0.0

        for X, Y, Y_gamma in pbar:
            X, Y, Y_gamma = (
                X.to(device, non_blocking=True),
                Y.to(device, non_blocking=True),
                Y_gamma.to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            with torch.amp.autocast(
                "cuda" if device == "cuda" else "cpu", enabled=scaler_enabled
            ):
                Y_pred = model(X)
                loss_mse = mse(Y_pred, Y)

                loss_geom = torch.tensor(0.0, device=device)
                avg_trust_val = 1.0

                if current_geom_weight > 0.0:
                    loss_geom, avg_trust_val = compute_geometric_loss_component(
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        Y_pred,
                        Y,
                        Y_gamma,
                        compute_trust=True,
                    )

                total_loss = loss_mse + (current_geom_weight * loss_geom)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_mse.item()
            running_geom += loss_geom.item()
            running_trust += avg_trust_val

            if not trial:
                pbar.set_postfix(
                    MSE=f"{loss_mse.item():.4f}", Geom=f"{loss_geom.item():.4f}"
                )

        # --- Validation Loop ---
        model.eval()
        val_mse_loss, val_geom_loss = 0.0, 0.0
        with torch.no_grad():
            for X, Y, Y_gamma in val_loader:
                X, Y, Y_gamma = X.to(device), Y.to(device), Y_gamma.to(device)
                with torch.amp.autocast(
                    "cuda" if device == "cuda" else "cpu", enabled=scaler_enabled
                ):
                    Y_pred = model(X)
                    loss_m = mse(Y_pred, Y)
                    loss_g, _ = compute_geometric_loss_component(
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        Y_pred,
                        Y,
                        Y_gamma,
                        compute_trust=False,
                    )
                val_mse_loss += loss_m.item()
                val_geom_loss += loss_g.item()

        avg_val_mse = val_mse_loss / len(val_loader)

        scheduler.step(avg_val_mse)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(avg_val_mse)
        history["geom_weight"].append(current_geom_weight)
        history["learning_rate"].append(current_lr_value)

        checkpoint_latest = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler_enabled else {},
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stop_state": early_stopper.state_dict(),
            "geometric_phase": geometric_phase,
            "geometric_start_epoch": geometric_start_epoch,
            "history": history,
        }
        torch.save(checkpoint_latest, os.path.join(out_dir, "unet_latest.pth"))

        if not trial and ((epoch + 1) % 5 == 0 or epoch == 0):
            save_sample_images(
                model, val_loader, device, out_dir, epoch + 1, denormalizer
            )

        # Physical Validation
        val_metrics = {"wd": []}
        with torch.no_grad():
            for i, (X_val, Y_val, _) in enumerate(val_loader):
                if i >= 10:
                    break
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                is_wet = X_val[:, 0].amax(dim=(1, 2)) > 1e-6
                wet_idx = torch.where(is_wet)[0]
                if len(wet_idx) == 0:
                    continue

                Y_pred_wet = model(X_val[wet_idx])
                Y_phys = denormalizer.unnormalize(Y_val[wet_idx, 0])
                Gen_phys = denormalizer.unnormalize(Y_pred_wet[:, 0])

                val_metrics["wd"].append(
                    compute_physical_metrics(Y_phys, Gen_phys)["wasserstein_dist"]
                )

        if val_metrics["wd"]:
            mean_wd = np.mean(val_metrics["wd"])
            best_wd_metric = min(best_wd_metric, mean_wd)

            if not trial:
                print(f"  > Wasserstein Dist: {mean_wd:.4f}")

            if early_stopper(mean_wd):
                if not geometric_phase:
                    if not trial:
                        print(
                            "!!! Convergence Reached. Triggering Geometric Curriculum !!!"
                        )
                    geometric_phase = True
                    geometric_start_epoch = epoch + 1
                    early_stopper.reset()

                    # --- METHOD 3 IMPLEMENTATION: STATE FLUSHING & LR RESET ---
                    new_lr = current_lr * 0.1
                    if not trial:
                        print("!!! Re-initializing Optimizer and Scheduler !!!")
                        print(
                            f"Flushing historical momentum. Setting base LR to {new_lr:.2e}"
                        )

                    optimizer = optim.AdamW(
                        model.parameters(), lr=new_lr, weight_decay=current_wd
                    )
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.5,
                        patience=scheduler_patience,
                        verbose=not bool(trial),
                    )
                    # ----------------------------------------------------------
                else:
                    if not trial:
                        print(
                            "!!! Convergence Reached in Geometric Phase. Stopping training."
                        )
                    break

        if trial:
            trial.report(avg_val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_wd_metric if trial else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_percentage", type=float, default=100.0)
    parser.add_argument(
        "--tune", action="store_true", help="Run optuna hyperparameter optimization"
    )
    parser.add_argument("--weight_geom", type=float, default=GEOMETRIC_TARGET_WEIGHT)
    args = parser.parse_args()

    if args.tune:
        print("Starting hyperparameter optimization study...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: run_training(args, trial), n_trials=20)

        print("\nOptimization complete.")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        run_training(args)


if __name__ == "__main__":
    main()
