import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import copy
import matplotlib.colors as mcolors

from src.ddpm.model_ddpm import ContextUnet
from src.ddpm.diffusion import Diffusion
from data.dataset import SRDataset

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 100
PATIENCE = 10
NUM_WORKERS = 4
EXPERIMENT_NAME = "DDPM_SR"


class DataDenormalizer:
    """
    Handles the inverse transformation of data from Model Space ([0,1])
    back to Physical Space (mm/h).
    """

    def __init__(self, stats_path):
        try:
            # Load the numpy array
            data = np.load(stats_path)

            # FIX: Use .item() instead of [0].
            # .item() works for both 0-d scalars (array(5.0)) and 1-d arrays of size 1 (array([5.0]))
            self.max_val = float(data.item())

            print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")
        except FileNotFoundError:
            print(
                f"Warning: Scaling stats not found at {stats_path}. Defaulting to 1.0."
            )
            self.max_val = 1.0
        except Exception as e:
            print(
                f"Warning: Failed to load denormalizer stats ({e}). Defaulting to 1.0."
            )
            self.max_val = 1.0

    def unnormalize(self, x_norm):
        """
        Inverse Pipeline:
        1. Scale up: x' = x_norm * max_val
        2. Inverse Log: x_phys = exp(x') - 1
        """
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()

        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        # Physical constraint: Precip >= 0
        return np.maximum(x_phys, 0.0)


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
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return True


def compute_physical_metrics(real_batch, gen_batch, drizzle_threshold=0.1):
    """
    Computes physical metrics.
    Expects INPUTS to be in PHYSICAL UNITS (mm/h).
    """
    # --- Apply Threshold ---
    # Zero out background noise < 0.1 mm/h to match SR evaluation standards
    real_batch = real_batch * (real_batch > drizzle_threshold).astype(float)
    gen_batch = gen_batch * (gen_batch > drizzle_threshold).astype(float)

    real_flat = real_batch.flatten()
    gen_flat = gen_batch.flatten()

    # 1. Wasserstein Distance (Earth Mover's Distance between intensity histograms)
    # Using a subset for speed if arrays are massive is often wise,
    # but for batch-level validation, full calculation is fine.
    wd = wasserstein_distance(real_flat, gen_flat)

    # 2. Max Intensity Error
    real_max = np.max(real_flat)
    gen_max = np.max(gen_flat)
    max_err = abs(real_max - gen_max)

    return {"wasserstein_dist": wd, "max_intensity_err": max_err}


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    """
    Runs inference on a batch and saves visualization plots matching the
    style of _plot_comprehensive_sample (Gray background for zeros, shared scales).
    """
    model.eval()
    try:
        X, Y, _ = next(iter(loader))
    except StopIteration:
        return

    # --- INFERENCE LOGIC ---
    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

    n_samples = min(5, X.shape[0])
    X_sample = X[:n_samples]
    Y_sample = Y[:n_samples]

    x_generated = torch.zeros(
        (n_samples, 1, diffusion.img_size, diffusion.img_size), device=device
    )

    # Channel 0 is Precip
    input_precip = X_sample[:, 0, :, :]
    is_wet_mask = input_precip.amax(dim=(1, 2)) > 1e-6
    wet_indices = torch.where(is_wet_mask)[0]
    n_wet = len(wet_indices)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            if n_wet > 0:
                X_wet = X_sample[wet_indices]
                # Replace diffusion.sample with sample_ddim
                gen_wet = diffusion.sample_ddim(
                    model, n=n_wet, conditions=X_wet, ddim_steps=50
                )
                x_generated[wet_indices] = gen_wet

    # --- DATA PREPARATION ---
    # Move to CPU
    X_cpu = X_sample.float().cpu().numpy()
    Y_cpu = Y_sample.float().cpu().numpy()
    Gen_cpu = x_generated.float().cpu().numpy()

    # --- NEW: Invert the [-1, 1] domain shift before denormalizing ---
    X_cpu = (X_cpu + 1.0) / 2.0
    Y_cpu = (Y_cpu + 1.0) / 2.0
    Gen_cpu = (Gen_cpu + 1.0) / 2.0

    # Denormalize Precip channels (Index 0)
    X_phys = denormalizer.unnormalize(X_cpu[:, 0])
    Y_phys = denormalizer.unnormalize(Y_cpu[:, 0])
    Gen_phys = denormalizer.unnormalize(Gen_cpu[:, 0])

    # --- STYLE ADAPTATION ---

    # 1. Define Colormap (Blues with Grey background for NaNs)
    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    # 2. Masking Helper
    def mask_low_values(img, threshold=0.1):
        """Masks values below threshold to NaN for plotting."""
        masked = img.copy()
        masked[masked <= threshold] = np.nan
        return masked

    # 3. Setup Figure
    # Adjust figsize: Width 18 (3 cols * 6), Height 4 * n_samples
    _, axs = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples), squeeze=False)

    for i in range(n_samples):
        # Extract fields
        img_in = X_phys[i]
        img_target = Y_phys[i]
        img_gen = Gen_phys[i]

        # Determine Global Max for this sample (Shared Colorbar)
        # We ensure min vmax is 1.0 to avoid errors on empty frames
        local_max = np.nanmax(
            [np.nanmax(img_in), np.nanmax(img_target), np.nanmax(img_gen)]
        )
        vmax = max(local_max, 1.0)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        # Apply masking
        img_in_masked = mask_low_values(img_in)
        img_target_masked = mask_low_values(img_target)
        img_gen_masked = mask_low_values(img_gen)

        # Plot A: Input (LR)
        im1 = axs[i, 0].imshow(
            img_in_masked, cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 0].set_title(f"Input (LR) | Max: {np.nanmax(img_in):.2f} mm/h")
        axs[i, 0].axis("off")
        plt.colorbar(im1, ax=axs[i, 0], fraction=0.046, pad=0.04, label="mm/h")

        # Plot B: Generated (SR)
        im2 = axs[i, 1].imshow(
            img_gen_masked, cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 1].set_title(f"Generated (SR) | Max: {np.nanmax(img_gen):.2f} mm/h")
        axs[i, 1].axis("off")
        plt.colorbar(im2, ax=axs[i, 1], fraction=0.046, pad=0.04, label="mm/h")

        # Plot C: Ground Truth (HR)
        im3 = axs[i, 2].imshow(
            img_target_masked, cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 2].set_title(f"Target (HR) | Max: {np.nanmax(img_target):.2f} mm/h")
        axs[i, 2].axis("off")
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04, label="mm/h")

    plt.tight_layout()

    # Ensure directory exists before saving
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    print(f"Active Device: {torch.cuda.get_device_name(0)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # --- Data & Denormalizer ---
    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    # Initialize Denormalizer
    stats_path = os.path.join(PREPROCESSED_DATA_DIR, "precip_max_val.npy")
    denormalizer = DataDenormalizer(stats_path)

    # We load the scaler directly from the class since it parsed it
    scaler_val = denormalizer.max_val

    train_dataset = SRDataset(
        preprocessed_data_dir=PREPROCESSED_DATA_DIR,
        metadata_file=METADATA_TRAIN,
        dem_patches_dir=DEM_DATA_DIR,
        dem_stats=dem_stats,
        scaler_max_val=scaler_val,
        split="train",
        wet_dry_ratio=1.0,  # 1:1 Balancing
    )
    val_dataset = SRDataset(
        preprocessed_data_dir=PREPROCESSED_DATA_DIR,
        metadata_file=METADATA_VAL,
        dem_patches_dir=DEM_DATA_DIR,
        dem_stats=dem_stats,
        scaler_max_val=scaler_val,
        split="validation",
        wet_dry_ratio=None,  # Keep validation true to climatology
    )

    # --- Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "timestamp": []}

    print("Starting DDPM Training (Mixed Precision)...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for X, Y, _ in pbar:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predicted_noise = model(x_t, t, X)
                loss = mse(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train MSE: {avg_loss:.6f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y, _ in val_loader:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)

                with torch.amp.autocast("cuda"):
                    predicted_noise = model(x_t, t, X)
                    loss = mse(noise, predicted_noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val MSE: {avg_val_loss:.6f}")

        # --- Logging ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["timestamp"].append(time.strftime("%H:%M:%S"))
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, "loss_history.csv"), index=False
        )

        if (epoch + 1) % 5 == 0 or (epoch == 0):
            save_sample_images(
                model, diffusion, val_loader, device, out_dir, epoch + 1, denormalizer
            )

        torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_latest.pth"))
        if epoch > 0 and avg_val_loss < min(history["val_loss"][:-1]):
            torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_best.pth"))

        # --- Physical Validation ---
        if (epoch + 1) % 5 == 0:
            print(f"Running Physical Validation (Sampling over {10} batches)...")
            model.eval()

            # Storage for aggregating metrics
            val_metrics = {"wd": [], "max_err": []}

            # Limit validation to N batches to prevent excessive training downtime
            NUM_PHYS_BATCHES = 10

            with torch.no_grad():
                for i, (X_val, Y_val, _) in enumerate(val_loader):
                    if i >= NUM_PHYS_BATCHES:
                        break

                    X_val, Y_val = X_val.to(device), Y_val.to(device)

                    # --- FILTER WET SAMPLES ---
                    # We only care about performance on rainy patches
                    input_precip = X_val[:, 0, :, :]
                    is_wet = input_precip.amax(dim=(1, 2)) > 1e-6
                    wet_indices = torch.where(is_wet)[0]

                    if len(wet_indices) == 0:
                        continue

                    X_wet = X_val[wet_indices]
                    Y_wet = Y_val[wet_indices]

                    # Sample from model using fast DDIM
                    gen_wet = diffusion.sample_ddim(
                        model, n=len(wet_indices), conditions=X_wet, ddim_steps=50
                    )

                    # --- PHYSICAL TRANSFORMATION ---
                    # Move to CPU, invert [-1, 1] shift, and Denormalize
                    Y_cpu = (Y_wet.cpu().numpy().squeeze() + 1.0) / 2.0
                    Gen_cpu = (gen_wet.cpu().numpy().squeeze() + 1.0) / 2.0

                    Y_phys = denormalizer.unnormalize(Y_cpu)
                    Gen_phys = denormalizer.unnormalize(Gen_cpu)

                    # Compute metrics for this batch
                    batch_metrics = compute_physical_metrics(Y_phys, Gen_phys)

                    val_metrics["wd"].append(batch_metrics["wasserstein_dist"])
                    val_metrics["max_err"].append(batch_metrics["max_intensity_err"])

            # --- AGGREGATE RESULTS ---
            if len(val_metrics["wd"]) > 0:
                mean_wd = np.mean(val_metrics["wd"])
                mean_max_err = np.mean(val_metrics["max_err"])

                print(
                    f"Epoch {epoch+1} Physical Metrics (Avg over {NUM_PHYS_BATCHES} batches):"
                )
                print(f"  > Wasserstein Dist: {mean_wd:.4f}")
                print(f"  > Max Intensity Err: {mean_max_err:.4f}")

                # Pass the meaningful average to early stopper
                early_stopper(mean_wd)
            else:
                print(
                    "Warning: No wet samples found in validation subset. Skipping metrics."
                )


if __name__ == "__main__":
    main()
