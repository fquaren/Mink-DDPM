import yaml
import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader

from model import UNetSR
from data.dataset import SRDataset
from loss import estimate_s_inv_from_dataset


def setup_evaluation(run_dir):
    """Loads config and sets up device."""
    print(f"Setting up evaluation for: {run_dir}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{run_dir}'")

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return config, device


def load_sr_model(config, device, run_dir, model_filename="best_sr_model.pth"):
    """
    Loads the UNetSR model and checkpoint.
    Modified to accept specific model filenames.
    """
    model = UNetSR(in_channels=2, out_channels=1).to(device)
    checkpoint_path = os.path.join(run_dir, model_filename)

    if not os.path.exists(checkpoint_path):
        # We return None here so the main loop can handle the missing file gracefully
        # rather than crashing immediately.
        print(f"Warning: Checkpoint file not found: '{checkpoint_path}'")
        return None

    print(f"Loading checkpoint: {model_filename}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("SR Model loaded successfully.")
    return model


def load_dem_stats(config):
    """Loads the DEM statistics file."""
    dem_stats_path = config["DEM_STATS"]
    with open(dem_stats_path, "r") as f:
        stats_dict = json.load(f)
    dem_stats_tuple = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    return dem_stats_tuple


def load_data(config, dem_stats, scaler_max_val):
    """Loads test data loader for SRDataset using the strict domain constraints."""
    from data.dataset import SRDataset  # Ensure proper import path

    test_dataset = SRDataset(
        preprocessed_data_dir=config["PREPROCESSED_DATA_DIR"],
        metadata_file=config["TEST_METADATA_FILE"],
        dem_patches_dir=config["DEM_DATA_DIR"],
        dem_stats=dem_stats,
        scaler_max_val=scaler_max_val,
        split="test",
        wet_dry_ratio=None,  # Keep validation true to climatology
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("SR_BATCH_SIZE", 16),
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4),
        pin_memory=True,
    )
    print(f"Loaded {len(test_dataset)} samples for evaluation.")
    return test_loader


def load_s_inv(config, dem_stats, device):
    """Loads train dataset to compute S_inv for geometric loss."""
    print("Loading train dataset to compute S_inv...")
    train_dataset_for_s = SRDataset(
        config["PREPROCESSED_DATA_DIR"],
        config["TRAIN_METADATA_FILE"],
        config["DEM_DATA_DIR"],
        dem_stats,
        split="train",
    )
    S_inv_tensors = estimate_s_inv_from_dataset(
        train_dataset_for_s, config.get("S_ESTIMATION_SAMPLES", 1000), device
    )
    return S_inv_tensors


def save_metrics_text(output_dir, group_metrics, per_feature_metrics):
    """Saves the key evaluation metrics to a text file."""
    file_path = os.path.join(output_dir, "evaluation_metrics.txt")
    print(f"\nSaving evaluation metrics to {file_path}...")

    try:
        with open(file_path, "w") as f:
            f.write("--- Metrics by Precipitation Group ---\n")
            f.write(group_metrics.to_string(float_format="%.6f"))
            f.write("\n\n")

            f.write("--- Per-Feature Analytical Gamma Metrics (Averaged) ---\n")
            f.write(
                per_feature_metrics["mean_by_component"].to_string(float_format="%.6f")
            )
            f.write("\n\n")

            f.write("--- Per-Feature Analytical Gamma R^2 (Full Matrix) ---\n")
            f.write(per_feature_metrics["r2_matrix"].to_string(float_format="%.4f"))
            f.write("\n\n")

            f.write("--- Per-Feature Analytical Gamma MSE (Full Matrix) ---\n")
            f.write(per_feature_metrics["mae_matrix"].to_string(float_format="%.4e"))
            f.write("\n\n")

            f.write("--- Per-Feature Analytical Gamma Variance (Full Matrix) ---\n")
            f.write(per_feature_metrics["var_matrix"].to_string(float_format="%.4e"))

        print("Metrics saved successfully.")
    except IOError as e:
        print(f"Error saving metrics file: {e}")


def save_metrics_npz(output_dir, metrics_df, per_feature_metrics):
    """Saves full metrics dataframe and per-feature gamma matrices."""
    # Save per-feature NPZ
    npz_save_path = os.path.join(output_dir, "per_feature_gamma_metrics.npz")
    np.savez_compressed(
        npz_save_path,
        r2_matrix=per_feature_metrics["r2_matrix"].values,
        mae_matrix=per_feature_metrics["mae_matrix"].values,
        var_matrix=per_feature_metrics["var_matrix"].values,
        quantiles=per_feature_metrics["quantiles"],
    )
    print(f"Per-feature analytical gamma metric matrices saved to: {npz_save_path}")
