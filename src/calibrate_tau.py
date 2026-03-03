import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
from tqdm import tqdm
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Project imports
from data.dataset import SRDataset
from utils import load_emulator


class DataDenormalizer:
    def __init__(self, stats_path):
        try:
            data = np.load(stats_path)
            self.max_val = float(data.item())
        except FileNotFoundError:
            self.max_val = 1.0

    def unnormalize_torch(self, x_norm):
        max_val_tensor = torch.tensor(
            self.max_val, device=x_norm.device, dtype=x_norm.dtype
        )
        x_scaled = x_norm * max_val_tensor
        x_scaled = torch.clamp(x_scaled, max=50.0)
        x_phys = torch.expm1(x_scaled)
        return F.relu(x_phys)


def calibrate_tau_log_space():
    """
    Calibrates tau using physical inputs mapped from the [-1, 1] domain,
    evaluating emulator errors in log-space to match the Minkowski loss.
    """
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_path, "config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Denormalizer
    scaler_path = os.path.join(config["PREPROCESSED_DATA_DIR"], "precip_max_val.npy")
    denormalizer = DataDenormalizer(scaler_path)
    print(f"Loaded Physical Max (Log-Space) Scaling Factor: {denormalizer.max_val:.4f}")

    with open(config["DEM_STATS"], "r") as f:
        stats = json.load(f)
    dem_stats = (float(stats["dem_mean"]), float(stats["dem_std"]))

    print("Initializing Dataset...")
    val_dataset = SRDataset(
        preprocessed_data_dir=config["PREPROCESSED_DATA_DIR"],
        metadata_file=config["VAL_METADATA_FILE"],
        dem_patches_dir=config["DEM_DATA_DIR"],
        dem_stats=dem_stats,
        scaler_max_val=denormalizer.max_val,
        split="validation",
        wet_dry_ratio=None,  # Keep true climatology for evaluation
    )

    loader = DataLoader(
        val_dataset,
        batch_size=config.get("BATCH_SIZE", 32),
        num_workers=4,
        shuffle=False,
    )

    print("Loading Emulator...")
    emu_path = config.get("EMULATOR_CHECKPOINT_PATH")
    emulator = load_emulator(emu_path, config, device)
    emulator.eval()

    errors_mean_spatial = []
    errors_max_spatial = []

    print("Computing Emulator Errors...")
    with torch.no_grad():
        # SRDataset returns: input_stack, target_tensor, target_gamma_tensor
        for _, Y, Y_gamma_log in tqdm(loader):
            Y = Y.to(device)
            Y_gamma_log = Y_gamma_log.to(device)

            # 1. Invert the [-1, 1] domain shift to [0, 1]
            Y_norm = (Y + 1.0) / 2.0

            # 2. Unnormalize to Physical mm/h and enforce drizzle threshold
            Y_phys = denormalizer.unnormalize_torch(Y_norm)
            Y_phys = Y_phys * (Y_phys > 0.1).float()

            # 3. Emulator Prediction (Returns physical topology)
            gamma_pred_phys = emulator(Y_phys)

            # 4. Transform to Log Space to match Y_gamma_log
            gamma_pred_log = torch.log1p(gamma_pred_phys)

            # 5. Compute MSE
            mse_tensor = F.mse_loss(gamma_pred_log, Y_gamma_log, reduction="none")

            mse_mean = mse_tensor.view(mse_tensor.size(0), -1).mean(dim=1)
            mse_max, _ = mse_tensor.view(mse_tensor.size(0), -1).max(dim=1)

            errors_mean_spatial.extend(mse_mean.cpu().numpy())
            errors_max_spatial.extend(mse_max.cpu().numpy())

    errors_mean_spatial = np.array(errors_mean_spatial)
    errors_max_spatial = np.array(errors_max_spatial)

    print("\n" + "=" * 40)
    print("       ERROR STATISTICS (Final)      ")
    print("=" * 40)

    stats_mean = {
        "p50": np.percentile(errors_mean_spatial, 50),
        "p90": np.percentile(errors_mean_spatial, 90),
    }
    stats_max = {
        "p50": np.percentile(errors_max_spatial, 50),
        "p90": np.percentile(errors_max_spatial, 90),
    }

    print(f"{'Metric':<10} | {'Median (p50)':<12} | {'High (p90)':<12}")
    print("-" * 40)
    print(f"{'Mean MSE':<10} | {stats_mean['p50']:<12.5f} | {stats_mean['p90']:<12.5f}")
    print(f"{'Max MSE':<10} | {stats_max['p50']:<12.5f} | {stats_max['p90']:<12.5f}")

    # Calculate Tau: tau = -ln(Target_Weight) / Error_Metric
    tau_std_aggressive = -np.log(0.1) / (stats_mean["p90"] + 1e-8)
    tau_stability = -np.log(0.5) / (stats_max["p90"] + 1e-8)

    print("\n" + "=" * 40)
    print("         RECOMMENDED TAU VALUES          ")
    print("=" * 40)
    print(f"Aggressive (Weight=0.1 at Mean p90): {tau_std_aggressive:.6f}")
    print(f"Stability (Weight=0.5 at Max p90):   {tau_stability:.6f}")
    print("-" * 40)


if __name__ == "__main__":
    calibrate_tau_log_space()
