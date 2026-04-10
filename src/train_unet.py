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
EPOCHS = config["NUM_EPOCHS"]
PATIENCE = config["PATIENCE"]
NUM_WORKERS = config["NUM_WORKERS"]
EXPERIMENT_NAME = "UNet_SR_Minkowski"
N_QUANTILES = len(config["QUANTILE_LEVELS"])

# --- Minkowski Loss Configuration ---
GEOMETRIC_WARMUP_EPOCHS = config.get("MINKOWSKI_WARMUP_EPOCHS", 5)
TRUST_TAU = config.get("TRUST_TAU", 0.1)
EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", "checkpoints/emulator_best.pth")


class DataDenormalizer:
    def __init__(self, max_val):
        self.max_val = float(max_val)
        print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")

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

    device_type = "cuda" if Y_pred_norm.device.type == "cuda" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        Y_pred_norm_f32 = Y_pred_norm.float()
        Y_clean_norm_f32 = Y_clean_norm.float()
        Y_gamma_log_f32 = Y_gamma_log.float()

        if compute_trust:
            with torch.no_grad():
                Y_phys = denormalizer.unnormalize_torch(Y_clean_norm_f32)
                Y_phys = Y_phys * (Y_phys > 0.1).float()
                gamma_truth_phys = emulator(Y_phys)
                gamma_truth_log_pred = torch.log1p(gamma_truth_phys)
                diff_trust = gamma_truth_log_pred - Y_gamma_log_f32[:, :3, :]
                emu_error_sq = diff_trust.pow(2).mean(dim=(1, 2))
                trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_sq)
                avg_trust_val = trust_weights.mean().item()

        pred_phys = denormalizer.unnormalize_torch(Y_pred_norm_f32)
        pred_phys = pred_phys * (pred_phys > 0.1).float()
        pred_gamma_phys = emulator(pred_phys)
        pred_gamma_log = torch.log1p(pred_gamma_phys)

        batch_total_dist, _, _, _ = criterion(pred_gamma_log, Y_gamma_log_f32[:, :3, :])

        weighted_loss = batch_total_dist * trust_weights.view(-1)
        loss_geom = weighted_loss.mean()

    return loss_geom, avg_trust_val


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
# Modified Training Loop for Script 2 (Emulator Version)
# ------------------------------------------------------------------------------
def run_training(args, dem_stats, max_val, trial=None):
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

    params_path = args.params_path
    with open(params_path, "r") as file:
        unet_params = yaml.safe_load(file)

    denormalizer = DataDenormalizer(max_val)

    train_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=max_val,
        split="train",
        data_percentage=args.data_percentage,
        load_in_ram=False,
    )
    val_dataset = DeterministicSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL,
        DEM_DATA_DIR,
        dem_stats,
        scaler_max_val=max_val,
        split="validation",
        data_percentage=args.data_percentage,
        load_in_ram=False,
    )

    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Constant target weight acts as w_max
    GEOMETRIC_TARGET_WEIGHT = (
        args.weight_geom
        if args.weight_geom is not None
        else config.get("MINKOWSKI_TARGET_WEIGHT", 0.0)
    )

    emulator = None
    geometric_criterion = None
    if GEOMETRIC_TARGET_WEIGHT > 0.0:
        geometric_criterion = MinkowskiLoss(
            quantile_levels=config["QUANTILE_LEVELS"]
        ).to(device)
        emulator = load_emulator(EMULATOR_PATH, config, device)
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)

    if trial:
        current_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        current_wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        current_lr = unet_params["lr"]
        current_wd = unet_params["weight_decay"]

    optimizer = optim.AdamW(
        [{"params": model.parameters()}], lr=current_lr, weight_decay=current_wd
    )

    mse = nn.MSELoss()
    scheduler_patience = max(1, PATIENCE // 2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
    scaler_enabled = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    history = {
        "epoch": [],
        "train_loss_total": [],
        "train_loss_mse": [],
        "train_loss_geom": [],
        "val_loss_total": [],
        "val_loss_mse": [],
        "val_loss_geom": [],
        "avg_trust": [],
        "geom_weight": [],
        "learning_rate": [],
    }

    start_epoch = 0
    best_wd_metric = float("inf")
    best_val_loss = float("inf")  # Add initialization here

    if args.resume and os.path.isfile(args.resume) and not trial:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler_enabled:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        early_stopper.load_state_dict(checkpoint["early_stop_state"])
        history = checkpoint.get("history", history)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        current_lr_value = optimizer.param_groups[0]["lr"]

        # Calculate cosine annealing schedule
        w_min = 0.0
        current_geom_weight = w_min + 0.5 * (GEOMETRIC_TARGET_WEIGHT - w_min) * (
            1 + np.cos(np.pi * epoch / EPOCHS)
        )

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [W_geom={current_geom_weight:.4f}]",
            mininterval=30.0,
        )

        running_total, running_mse, running_geom, running_trust = 0.0, 0.0, 0.0, 0.0

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

                if current_geom_weight > 0.0 and emulator is not None:
                    loss_geom, avg_trust_val = compute_geometric_loss_component(
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        Y_pred,
                        Y,
                        Y_gamma,
                        compute_trust=True,
                    )

                total_loss = loss_mse + current_geom_weight * loss_geom

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total += total_loss.item()
            running_mse += loss_mse.item()
            running_geom += (
                loss_geom.item() if isinstance(loss_geom, torch.Tensor) else loss_geom
            )
            running_trust += avg_trust_val

            pbar.set_postfix(
                Tot=f"{total_loss.item():.3f}",
                MSE=f"{loss_mse.item():.4f}",
                Geom=f"{loss_geom.item() if isinstance(loss_geom, torch.Tensor) else loss_geom:.4f}",
                Trust=f"{avg_trust_val:.3f}",
            )

        model.eval()
        val_total_loss, val_mse_loss, val_geom_loss, val_trust_acc = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for X, Y, Y_gamma in val_loader:
                X, Y, Y_gamma = X.to(device), Y.to(device), Y_gamma.to(device)
                with torch.amp.autocast(
                    "cuda" if device == "cuda" else "cpu", enabled=scaler_enabled
                ):
                    Y_pred = model(X)
                    loss_m = mse(Y_pred, Y)
                    loss_g = torch.tensor(0.0, device=device)
                    v_trust = 1.0

                    if current_geom_weight > 0.0 and emulator is not None:
                        loss_g, v_trust = compute_geometric_loss_component(
                            emulator,
                            geometric_criterion,
                            denormalizer,
                            Y_pred,
                            Y,
                            Y_gamma,
                            compute_trust=False,
                        )

                    loss_t = loss_m + current_geom_weight * loss_g

                val_total_loss += loss_t.item()
                val_mse_loss += loss_m.item()
                val_geom_loss += (
                    loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g
                )
                val_trust_acc += v_trust

        avg_val_tot = val_total_loss / len(val_loader)
        avg_val_mse = val_mse_loss / len(val_loader)
        avg_val_geom = val_geom_loss / len(val_loader)
        avg_val_trust = val_trust_acc / len(val_loader)

        scheduler.step(avg_val_tot)

        history["epoch"].append(epoch + 1)
        history["train_loss_total"].append(running_total / len(train_loader))
        history["train_loss_mse"].append(running_mse / len(train_loader))
        history["train_loss_geom"].append(running_geom / len(train_loader))
        history["val_loss_total"].append(avg_val_tot)
        history["val_loss_mse"].append(avg_val_mse)
        history["val_loss_geom"].append(avg_val_geom)
        history["avg_trust"].append(running_trust / len(train_loader))
        history["geom_weight"].append(current_geom_weight)
        history["learning_rate"].append(current_lr_value)

        # Check if current validation loss is the lowest observed
        is_best = avg_val_tot < best_val_loss
        best_val_loss = min(best_val_loss, avg_val_tot)

        checkpoint_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler_enabled else {},
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stop_state": early_stopper.state_dict(),
            "history": history,
        }

        # Always save the latest checkpoint
        torch.save(checkpoint_state, os.path.join(out_dir, "unet_latest.pth"))

        # Save the best checkpoint based on total validation loss
        if is_best:
            torch.save(checkpoint_state, os.path.join(out_dir, "unet_best.pth"))

        if not trial and ((epoch + 1) % 5 == 0 or epoch == 0):
            save_sample_images(
                model, val_loader, device, out_dir, epoch + 1, denormalizer
            )

        # --- Early Stopping Logic ---
        # Changed from mean_wd to avg_val_mse
        if early_stopper(avg_val_mse):
            if not trial:
                print(
                    f"Early stopping triggered based on validation MSE plateau at epoch {epoch+1}."
                )
            break

        # Optional: Physical metrics calculation (kept for logging/visualization only)
        val_metrics = {"wd": []}
        if not trial:
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
                    Y_phys = denormalizer.unnormalize(Y_val[wet_idx, 0].cpu())
                    Gen_phys = denormalizer.unnormalize(Y_pred_wet[:, 0].cpu())
                    val_metrics["wd"].append(
                        compute_physical_metrics(Y_phys, Gen_phys)["wasserstein_dist"]
                    )

            if val_metrics["wd"]:
                mean_wd = np.mean(val_metrics["wd"])
                print(
                    f"  > Val Tot: {avg_val_tot:.4f} | MSE: {avg_val_mse:.4f} | WD: {mean_wd:.4f}"
                )

        if trial:
            trial.report(avg_val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_wd_metric if trial else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_percentage", type=float, default=100.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--weight_geom", type=float, default=None)
    parser.add_argument(
        "--params_path", type=str, default=os.path.join(parent_path, "unet_params.yaml")
    )
    args = parser.parse_args()

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    try:
        max_val_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
        max_val = float(np.load(max_val_path).item())
    except FileNotFoundError:
        print(f"Warning: Scaling stats not found at {max_val_path}. Defaulting to 1.0.")
        max_val = 1.0

    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: run_training(args, dem_stats, max_val, trial), n_trials=10
        )
        with open(os.path.join(parent_path, "unet_params.yaml"), "w") as f:
            yaml.dump(study.best_params, f)
    else:
        run_training(args, dem_stats, max_val)


if __name__ == "__main__":
    main()
