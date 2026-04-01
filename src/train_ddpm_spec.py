import argparse
import copy
import json
import os
import sys
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
from tqdm import tqdm

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

from models.SR.ddpm.ddpm import ContextUnet
from models.SR.ddpm.diffusion import Diffusion
from data.dataset import DiffusionSRDataset
from src.loss import (
    FFT2DKernelCRPSLoss,
    ImageSSIMLoss,
    OpticalFlowConsistencyLoss,
    WeightedSoftWetAreaLoss,
)

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
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "DDPM_SR_SpecLoss")

AUX_LOSS_TYPE = str(config.get("AUX_LOSS_TYPE", "fft2d")).lower()
AUX_TARGET_WEIGHT_DEFAULT = float(config.get("AUX_TARGET_WEIGHT", 0.0))
AUX_WARMUP_EPOCHS = int(config.get("AUX_WARMUP_EPOCHS", 5))
AUX_T_THRESHOLD = int(config.get("AUX_T_THRESHOLD", 250))
FFT2D_ALPHA = float(config.get("FFT2D_ALPHA", 1.0))
SSIM_WINDOW_SIZE = int(config.get("SSIM_WINDOW_SIZE", 11))
WET_AREA_THRESHOLD = float(config.get("WET_AREA_THRESHOLD", 0.0))
WET_AREA_TEMPERATURE = float(config.get("WET_AREA_TEMPERATURE", 0.1))
WET_AREA_FALSE_POSITIVE_WEIGHT = float(config.get("WET_AREA_FALSE_POSITIVE_WEIGHT", 1.0))
WET_AREA_FALSE_NEGATIVE_WEIGHT = float(config.get("WET_AREA_FALSE_NEGATIVE_WEIGHT", 1.0))
OPTICAL_FLOW_PATCH_SIZE = int(config.get("OPTICAL_FLOW_PATCH_SIZE", 21))
OPTICAL_FLOW_SEARCH_RADIUS = int(config.get("OPTICAL_FLOW_SEARCH_RADIUS", 6))
OPTICAL_FLOW_DOWNSAMPLE = int(config.get("OPTICAL_FLOW_DOWNSAMPLE", 4))
OPTICAL_FLOW_PREPROCESS_SIGMA = float(config.get("OPTICAL_FLOW_PREPROCESS_SIGMA", 1.0))
OPTICAL_FLOW_DELTA = float(config.get("OPTICAL_FLOW_DELTA", 0.2))
OPTICAL_FLOW_MASK_THRESHOLD = config.get("OPTICAL_FLOW_MASK_THRESHOLD", 0.05)
OPTICAL_FLOW_LOSS_TYPE = str(config.get("OPTICAL_FLOW_LOSS_TYPE", "huber"))


class DataDenormalizer:
    def __init__(self, stats_path):
        try:
            data = np.load(stats_path)
            self.max_val = float(data.item())
            print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")
        except FileNotFoundError:
            print(f"Warning: Scaling stats not found at {stats_path}. Defaulting to 1.0.")
            self.max_val = 1.0

    def unnormalize(self, x_norm):
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()
        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        return np.maximum(x_phys, 0.0)


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


def reconstruct_x0_scaled(diffusion, x_t, predicted_noise, t):
    alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_hat_t)
    pred_x0_scaled = (x_t - sqrt_one_minus * predicted_noise) / (sqrt_alpha_hat + 1e-8)
    return torch.clamp(pred_x0_scaled, -1.0, 1.0)


def build_aux_criterion(loss_type: str, img_size: int = 128):
    if loss_type == "fft2d":
        return FFT2DKernelCRPSLoss(alpha=FFT2D_ALPHA)
    if loss_type == "ssim":
        return ImageSSIMLoss(window_size=SSIM_WINDOW_SIZE)
    if loss_type == "wet_area":
        return WeightedSoftWetAreaLoss(
            threshold=WET_AREA_THRESHOLD,
            temperature=WET_AREA_TEMPERATURE,
            false_positive_weight=WET_AREA_FALSE_POSITIVE_WEIGHT,
            false_negative_weight=WET_AREA_FALSE_NEGATIVE_WEIGHT,
        )
    if loss_type == "optical_flow":
        return OpticalFlowConsistencyLoss(
            x_dim=img_size,
            y_dim=img_size,
            patch_size=OPTICAL_FLOW_PATCH_SIZE,
            search_radius=OPTICAL_FLOW_SEARCH_RADIUS,
            downsample=OPTICAL_FLOW_DOWNSAMPLE,
            preprocess_sigma=OPTICAL_FLOW_PREPROCESS_SIGMA,
            delta=OPTICAL_FLOW_DELTA,
            mask_threshold=OPTICAL_FLOW_MASK_THRESHOLD,
            loss_type=OPTICAL_FLOW_LOSS_TYPE,
        )
    raise ValueError(f"Unsupported AUX_LOSS_TYPE: {loss_type}")


def compute_aux_loss_component(diffusion, criterion, x_t, predicted_noise, t, Y_clean_scaled, X_cond):
    loss_aux = torch.tensor(0.0, device=x_t.device)
    mask_time = (t < AUX_T_THRESHOLD).float()
    if mask_time.sum() == 0:
        return loss_aux

    idx = torch.where(mask_time > 0.0)[0]
    pred_x0_scaled = reconstruct_x0_scaled(diffusion, x_t, predicted_noise, t)
    pred_sel = pred_x0_scaled[idx].float()
    target_sel = Y_clean_scaled[idx].float()

    if AUX_LOSS_TYPE == "optical_flow":
        reference = ((X_cond[idx, 0:1] + 1.0) / 2.0).float()
        pred_in = ((pred_sel + 1.0) / 2.0).float()
        target_in = ((target_sel + 1.0) / 2.0).float()
        loss_aux = criterion(pred_in, target_in, reference=reference)
    elif AUX_LOSS_TYPE == "wet_area":
        pred_in = ((pred_sel + 1.0) / 2.0).float()
        target_in = ((target_sel + 1.0) / 2.0).float()
        loss_aux = criterion(pred_in, target_in)
    else:
        loss_aux = criterion(pred_sel, target_sel)

    return loss_aux


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    selected_X, selected_Y = [], []
    dry_count, wet_count, target_dry, target_wet = 0, 0, 1, 4
    drizzle_threshold = 0.1

    for X_batch, Y_batch, _ in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        X_shifted = (X_batch[:, 0] + 1.0) / 2.0
        X_phys = denormalizer.unnormalize(X_shifted)
        max_precip = X_phys.max(axis=(1, 2))

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
    X_sample, Y_sample = torch.cat(selected_X, dim=0), torch.cat(selected_Y, dim=0)
    n_samples = X_sample.size(0)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            x_generated = diffusion.sample_ddim(model, n=n_samples, conditions=X_sample, ddim_steps=50)

    X_norm = (X_sample[:, 0].cpu().numpy() + 1.0) / 2.0
    Y_norm = (Y_sample[:, 0].cpu().numpy() + 1.0) / 2.0
    Gen_norm = (x_generated[:, 0].cpu().clamp(-1.0, 1.0).numpy() + 1.0) / 2.0
    X_p, Y_p, G_p = (
        denormalizer.unnormalize(X_norm),
        denormalizer.unnormalize(Y_norm),
        denormalizer.unnormalize(Gen_norm),
    )

    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axs = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples), squeeze=False)
    for i in range(n_samples):
        vmax = max(np.nanmax(Y_p[i]), 1.0)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        for j, (img, title) in enumerate(zip([X_p[i], G_p[i], Y_p[i]], ["Input", "Generated", "Target"])):
            m_img = img.copy()
            m_img[m_img <= drizzle_threshold] = np.nan
            im = axs[i, j].imshow(m_img, cmap=precip_cmap, norm=norm, origin="lower")
            axs[i, j].set_title(f"{title} | Max: {np.nanmax(img):.2f}")
            axs[i, j].axis("off")
            if j == 2:
                plt.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"), bbox_inches="tight", dpi=100)
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
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}_trial_{trial.number}_{timestamp}" if trial else f"{EXPERIMENT_NAME}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    denormalizer = DataDenormalizer(os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy"))

    aux_target_weight = args.loss_weight if args.loss_weight is not None else AUX_TARGET_WEIGHT_DEFAULT
    print(f"Using auxiliary loss type: {AUX_LOSS_TYPE}")
    print(f"Using auxiliary target weight: {aux_target_weight:.4f}")

    train_ds = DiffusionSRDataset(PREPROCESSED_DATA_DIR, METADATA_TRAIN, DEM_DATA_DIR, dem_stats, denormalizer.max_val, "train", args.data_percentage)
    val_ds = DiffusionSRDataset(PREPROCESSED_DATA_DIR, METADATA_VAL, DEM_DATA_DIR, dem_stats, denormalizer.max_val, "validation", args.data_percentage)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    aux_criterion = build_aux_criterion(AUX_LOSS_TYPE, img_size=config.get("PATCH_SIZE", 128)).to(device)

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    current_lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) if trial else LR
    current_wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if trial else WD
    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)
    mse_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE // 2)
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    aux_phase, aux_start_epoch = False, None

    for epoch in range(EPOCHS):
        model.train()
        current_aux_weight = 0.0
        if aux_phase and aux_start_epoch is not None:
            current_aux_weight = aux_target_weight * min(1.0, (epoch - aux_start_epoch) / float(AUX_WARMUP_EPOCHS))

        running_loss, running_aux = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [W={current_aux_weight:.4f}]", disable=bool(trial))

        for X, Y, _ in pbar:
            X, Y = X.to(device), Y.to(device)
            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred_noise = model(x_t, t, X)
                loss_mse = mse_loss_fn(noise, pred_noise)
                loss_aux = (
                    compute_aux_loss_component(diffusion, aux_criterion, x_t, pred_noise, t, Y, X)
                    if current_aux_weight > 0
                    else torch.tensor(0.0, device=device)
                )
                total_loss = loss_mse + (current_aux_weight * loss_aux)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_mse.item()
            running_aux += loss_aux.item()

        model.eval()
        val_mse, val_aux = 0.0, 0.0
        with torch.no_grad():
            for X, Y, _ in val_loader:
                X, Y = X.to(device), Y.to(device)
                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)
                with torch.amp.autocast("cuda"):
                    pred_n = model(x_t, t, X)
                    loss_m = mse_loss_fn(noise, pred_n)
                    loss_a = compute_aux_loss_component(diffusion, aux_criterion, x_t, pred_n, t, Y, X)
                val_mse += loss_m.item()
                val_aux += loss_a.item()

        avg_val_mse = val_mse / len(val_loader)
        scheduler.step(avg_val_mse)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if not trial:
                save_sample_images(model, diffusion, val_loader, device, out_dir, epoch + 1, denormalizer)
            model.eval()
            wds = []
            with torch.no_grad():
                for i, (Xv, Yv, _) in enumerate(val_loader):
                    if i >= 10:
                        break
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    idx = torch.where(Xv[:, 0].amax(dim=(1, 2)) > 1e-6)[0]
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
                if early_stopper(mean_wd) and not aux_phase:
                    print("!!! Convergence Reached. Triggering Auxiliary-Loss Curriculum !!!")
                    aux_phase, aux_start_epoch = True, epoch + 1
                    early_stopper.reset()
                    optimizer = optim.AdamW(model.parameters(), lr=current_lr * 0.1, weight_decay=current_wd)
                elif early_stopper(mean_wd) and aux_phase:
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
    parser.add_argument("--loss_weight", type=float, default=None)
    args = parser.parse_args()
    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: run_training(args, trial), n_trials=10)
        if study.best_trial:
            print("\nBest Hyperparameters:")
            for key, value in study.best_trial.params.items():
                print(f"{key}: {value}")
            with open(os.path.join(parent_path, "ddpm_params.yaml"), "w") as f:
                yaml.dump(study.best_trial.params, f)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
