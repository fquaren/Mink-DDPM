import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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
import logging

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Project imports
from models.SR.ddpm.ddpm import ContextUnet
from models.SR.ddpm.diffusion import Diffusion
from data.dataset import DiffusionSRDataset

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["NUM_EPOCHS"]
PATIENCE = config["PATIENCE"]
NUM_WORKERS = config["NUM_WORKERS"]
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "DDPM_SR_Standard")


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
        self.patience, self.delta, self.verbose = patience, delta, verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

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
        for k, v in state_dict.items():
            setattr(self, k, v)


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    model.eval()

    # Retrieve the first batch
    X_batch, Y_batch, *_ = next(iter(loader))
    X_batch = X_batch.to(device, non_blocking=True)
    Y_batch = Y_batch.to(device, non_blocking=True)

    # DDPM data is scaled to [-1, 1], shift back to [0, 1] for unnormalization
    X_shifted = (X_batch[:, 0] + 1.0) / 2.0
    X_phys = denormalizer.unnormalize_torch(X_shifted)
    total_precip = X_phys.sum(dim=(1, 2))

    # Select the sample with the maximum total precipitation
    max_idx = total_precip.argmax().item()
    X_sample = X_batch[max_idx : max_idx + 1]
    Y_sample = Y_batch[max_idx : max_idx + 1]

    # Generate sample using DDIM
    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu"):
            x_generated = diffusion.sample_ddim(
                model, n=1, conditions=X_sample, ddim_steps=50
            )

    # Shift tensors from [-1, 1] back to [0, 1] before unnormalizing
    X_norm = (X_sample[:, 0].float().cpu().numpy() + 1.0) / 2.0
    Y_norm = (Y_sample[:, 0].float().cpu().numpy() + 1.0) / 2.0
    Gen_norm = (x_generated[:, 0].float().cpu().clamp(-1.0, 1.0).numpy() + 1.0) / 2.0

    img_in = denormalizer.unnormalize(X_norm)[0]
    img_target = denormalizer.unnormalize(Y_norm)[0]
    img_gen = denormalizer.unnormalize(Gen_norm)[0]

    os.makedirs(out_dir, exist_ok=True)

    # Save raw physical data
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
    real_f = real_batch[real_batch > drizzle_threshold].flatten()
    gen_f = gen_batch[gen_batch > drizzle_threshold].flatten()
    if len(real_f) == 0 or len(gen_f) == 0:
        return {"wasserstein_dist": 0.0, "max_intensity_err": 0.0}
    return {
        "wasserstein_dist": wasserstein_distance(real_f, gen_f),
        "max_intensity_err": abs(np.max(real_f) - np.max(gen_f)),
    }


def run_training(args, trial=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{EXPERIMENT_NAME}_trial_{trial.number}_{timestamp}"
        if trial
        else f"{EXPERIMENT_NAME}_{timestamp}"
    )
    out_dir = os.path.join("ci26_revision_runs", "sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    denormalizer = DataDenormalizer(
        os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    )

    train_ds = DiffusionSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN,
        DEM_DATA_DIR,
        dem_stats,
        denormalizer.max_val,
        "train",
        args.data_percentage,
    )
    val_ds = DiffusionSRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL,
        DEM_DATA_DIR,
        dem_stats,
        denormalizer.max_val,
        "validation",
        args.data_percentage,
    )
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    params_path = args.params_path
    with open(params_path, "r") as file:
        ddpm_params = yaml.safe_load(file)

    if trial:
        current_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        current_wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    else:
        current_lr = ddpm_params["lr"]
        current_wd = ddpm_params["weight_decay"]

    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)
    mse_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=PATIENCE // 2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    best_val_mse = float("inf")

    for epoch in tqdm(range(EPOCHS), desc="Overall Training Progress"):
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
        )

        # Use *_ to unpack any potential extra returned values from the dataset (e.g. gamma targets)
        for X, Y, *_ in pbar:
            X, Y = X.to(device), Y.to(device)
            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred_noise = model(x_t, t, X)
                loss_mse = mse_loss_fn(noise, pred_noise)

            scaler.scale(loss_mse).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_mse.item()

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for X, Y, *_ in val_loader:
                X, Y = X.to(device), Y.to(device)
                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred_n = model(x_t, t, X)
                    loss_m = mse_loss_fn(noise, pred_n)
                val_mse += loss_m.item()

        avg_val_mse = val_mse / len(val_loader)
        scheduler.step(avg_val_mse)

        # Save model weights
        torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_latest.pth"))
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_best.pth"))

        # Triggers and Checkpoints
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if not trial:
                save_sample_images(
                    model,
                    diffusion,
                    val_loader,
                    device,
                    out_dir,
                    epoch + 1,
                    denormalizer,
                )

            # Physical metric check for early stopping
            model.eval()
            wds = []
            with torch.no_grad():
                for i, (Xv, Yv, *_) in enumerate(val_loader):
                    if i >= 10:
                        break
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    Xv_norm = (Xv[:, 0] + 1.0) / 2.0
                    idx = torch.where(Xv_norm.amax(dim=(1, 2)) > 1e-6)[0]
                    if len(idx) == 0:
                        continue
                    gv = diffusion.sample_ddim(model, len(idx), Xv[idx], 50)
                    wds.append(
                        compute_physical_metrics(
                            denormalizer.unnormalize((Yv[idx].cpu().numpy() + 1) / 2),
                            denormalizer.unnormalize((gv.cpu().numpy() + 1) / 2),
                        )["wasserstein_dist"]
                    )

            if wds:
                mean_wd = np.mean(wds)
                if early_stopper(mean_wd):
                    print("Early stopping triggered. Convergence reached.")
                    break

        if trial:
            trial.report(avg_val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return avg_val_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_percentage", type=float, default=100.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--params_path", type=str, default=None)
    args = parser.parse_args()

    if args.tune:
        logger.info("Starting Optuna ...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: run_training(args, trial), n_trials=10)

        # Dump best hyperparameters if tuning was performed
        if study.best_trial:
            logger.info("\nBest Hyperparameters: ")
            for key, value in study.best_trial.params.items():
                logger.info(f"{key}: {value}")
            with open(os.path.join(parent_path, "ddpm_params.yaml"), "w") as f:
                yaml.dump(study.best_trial.params, f)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
