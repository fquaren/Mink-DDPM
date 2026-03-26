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
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "DDPM_SR_Minkowski")
N_QUANTILES = len(config["QUANTILE_LEVELS"])

# --- Minkowski Loss Configuration ---
GEOMETRIC_WARMUP_EPOCHS = config.get("MINKOWSKI_WARMUP_EPOCHS", 5)
GEOMETRIC_T_THRESHOLD = config.get("MINKOWSKI_T_THRESHOLD", 250)
TRUST_TAU = config.get("TRUST_TAU", 0.1)
EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", "checkpoints/emulator_best.pth")


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


def compute_geometric_loss_component(
    diffusion,
    emulator,
    criterion,
    denormalizer,
    x_t,
    predicted_noise,
    t,
    Y_clean_scaled,
    Y_gamma_log,
    compute_trust=True,
):
    loss_geom = torch.tensor(0.0, device=x_t.device)
    avg_trust_val = 1.0

    # x0 estimation from noisy x_t
    alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_hat_t)
    pred_x0_scaled = (x_t - sqrt_one_minus * predicted_noise) / (sqrt_alpha_hat + 1e-8)
    pred_x0_scaled = torch.clamp(pred_x0_scaled, -1.0, 1.0)

    # Time thresholding for geometric loss
    mask_time = (t < GEOMETRIC_T_THRESHOLD).float()

    if mask_time.sum() > 0:
        batch_size = x_t.shape[0]
        trust_weights = torch.ones(batch_size, device=x_t.device)

        if compute_trust:
            with torch.no_grad():
                Y_norm = (Y_clean_scaled + 1.0) / 2.0
                Y_phys = denormalizer.unnormalize_torch(Y_norm)
                Y_phys = Y_phys * (Y_phys > 0.1).float()
                gamma_truth_phys = emulator(Y_phys)
                gamma_truth_log_pred = torch.log1p(gamma_truth_phys)
                diff_trust = (gamma_truth_log_pred - Y_gamma_log).float()
                emu_error_sq = diff_trust.pow(2).mean(
                    dim=tuple(range(1, diff_trust.ndim))
                )
                trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_sq)
                avg_trust_val = trust_weights.mean().item()

        # Scale back to log-physical space
        pred_x0_norm = (pred_x0_scaled + 1.0) / 2.0
        pred_x0_phys = denormalizer.unnormalize_torch(pred_x0_norm)
        pred_x0_phys = pred_x0_phys * (pred_x0_phys > 0.1).float()
        pred_gamma_phys = emulator(pred_x0_phys)
        pred_gamma_log = torch.log1p(pred_gamma_phys)

        # Handle 5-tuple return [B]
        total_dist_b, _, _, _, _ = criterion(pred_gamma_log, Y_gamma_log)

        # Apply trust and time masks [B]
        weight_factor = trust_weights * mask_time
        weighted_loss = total_dist_b * weight_factor
        loss_geom = weighted_loss.sum() / (mask_time.sum() + 1e-8)

    return loss_geom, avg_trust_val


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    selected_X, selected_Y = [], []
    dry_count, wet_count, target_dry, target_wet = 0, 0, 1, 4
    drizzle_threshold = 0.1

    for X_batch, Y_batch, _ in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        X_shifted = (X_batch[:, 0] + 1.0) / 2.0
        X_phys = denormalizer.unnormalize_torch(X_shifted)
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
    X_sample, Y_sample = torch.cat(selected_X, dim=0), torch.cat(selected_Y, dim=0)
    n_samples = X_sample.size(0)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            x_generated = diffusion.sample_ddim(
                model, n=n_samples, conditions=X_sample, ddim_steps=50
            )

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
        for j, (img, title) in enumerate(
            zip([X_p[i], G_p[i], Y_p[i]], ["Input", "Generated", "Target"])
        ):
            m_img = img.copy()
            m_img[m_img <= drizzle_threshold] = np.nan
            im = axs[i, j].imshow(m_img, cmap=precip_cmap, norm=norm, origin="lower")
            axs[i, j].set_title(f"{title} | Max: {np.nanmax(img):.2f}")
            axs[i, j].axis("off")
            if j == 2:
                plt.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
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
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{EXPERIMENT_NAME}_trial_{trial.number}_{timestamp}"
        if trial
        else f"{EXPERIMENT_NAME}_{timestamp}"
    )
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    denormalizer = DataDenormalizer(
        os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    )

    GEOMETRIC_TARGET_WEIGHT = (
        args.geom_weight
        if args.geom_weight is not None
        else config.get("MINKOWSKI_TARGET_WEIGHT", 0.0)
    )
    print(f"Using geometric target weight: {GEOMETRIC_TARGET_WEIGHT:.4f}")

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

    geometric_criterion = MinkowskiLoss(config["QUANTILE_LEVELS"]).to(device)
    emulator = load_emulator(EMULATOR_PATH, config, device)
    emulator.eval()

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    current_lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) if trial else LR
    current_wd = (
        trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if trial else WD
    )
    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_wd)
    mse_loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=PATIENCE // 2
    )
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=not bool(trial))

    geometric_phase, geometric_start_epoch = False, None

    for epoch in range(EPOCHS):
        model.train()
        current_geom_weight = 0.0
        if geometric_phase and geometric_start_epoch is not None:
            current_geom_weight = GEOMETRIC_TARGET_WEIGHT * min(
                1.0, (epoch - geometric_start_epoch) / float(GEOMETRIC_WARMUP_EPOCHS)
            )

        running_loss, running_geom, running_trust = 0.0, 0.0, 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [W={current_geom_weight:.4f}]",
            disable=bool(trial),
        )

        for X, Y, Y_gamma in pbar:
            X, Y, Y_gamma = X.to(device), Y.to(device), Y_gamma.to(device)
            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred_noise = model(x_t, t, X)
                loss_mse = mse_loss_fn(noise, pred_noise)
                loss_geom, trust_v = (
                    compute_geometric_loss_component(
                        diffusion,
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        x_t,
                        pred_noise,
                        t,
                        Y,
                        Y_gamma,
                    )
                    if current_geom_weight > 0
                    else (torch.tensor(0.0, device=device), 1.0)
                )
                total_loss = loss_mse + (current_geom_weight * loss_geom)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_mse.item()
            running_geom += loss_geom.item()
            running_trust += trust_v

        # Validation
        model.eval()
        val_mse, val_geom = 0.0, 0.0
        with torch.no_grad():
            for X, Y, Y_gamma in val_loader:
                X, Y, Y_gamma = X.to(device), Y.to(device), Y_gamma.to(device)
                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)
                with torch.amp.autocast("cuda"):
                    pred_n = model(x_t, t, X)
                    loss_m = mse_loss_fn(noise, pred_n)
                    loss_g, _ = compute_geometric_loss_component(
                        diffusion,
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        x_t,
                        pred_n,
                        t,
                        Y,
                        Y_gamma,
                        False,
                    )
                val_mse += loss_m.item()
                val_geom += loss_g.item()

        avg_val_mse = val_mse / len(val_loader)
        scheduler.step(avg_val_mse)

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
            # Physical metric check for trigger
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
                if early_stopper(mean_wd) and not geometric_phase:
                    print(
                        "!!! Convergence Reached. Triggering Geometric Curriculum !!!"
                    )
                    geometric_phase, geometric_start_epoch = True, epoch + 1
                    early_stopper.reset()
                    optimizer = optim.AdamW(
                        model.parameters(), lr=current_lr * 0.1, weight_decay=current_wd
                    )
                elif early_stopper(mean_wd) and geometric_phase:
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
    parser.add_argument("--geom_weight", type=float, default=None)
    args = parser.parse_args()
    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: run_training(args, trial), n_trials=10)

        # Dump best hyperparameters if tuning was performed
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
