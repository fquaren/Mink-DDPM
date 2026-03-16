import yaml
import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
import sys

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from data.dataset import DiffusionSRDataset


def setup_evaluation(run_dir):
    """Loads config and sets up device."""
    print(f"Setting up evaluation for: {run_dir}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{run_dir}'")

    # Prioritize the snapshot created during training for exact hyperparameter replication
    snapshot_path = os.path.join(run_dir, "config_snapshot.yaml")
    base_config_path = os.path.join(run_dir, "config.yaml")

    if os.path.exists(snapshot_path):
        config_path = snapshot_path
        print("Loading configuration from snapshot...")
    elif os.path.exists(base_config_path):
        config_path = base_config_path
        print("Loading configuration from base config.yaml...")
    else:
        raise FileNotFoundError(f"No configuration file found in {run_dir}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return config, device


def load_dem_stats(config):
    """Loads the DEM statistics file."""
    dem_stats_path = config["DEM_STATS"]
    with open(dem_stats_path, "r") as f:
        stats_dict = json.load(f)
    dem_stats_tuple = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    return dem_stats_tuple


def load_data(config, dem_stats, scaler_max_val):
    """Loads test data loader for DiffusionSRDataset using the strict domain constraints."""

    # Critical check to ensure TEST_METADATA_FILE is present, as training configs
    # sometimes omit it in favor of VAL_METADATA_FILE.
    if "TEST_METADATA_FILE" not in config:
        print(
            "Warning: TEST_METADATA_FILE missing from config. Defaulting to VAL_METADATA_FILE for evaluation."
        )
        metadata_target = config.get("VAL_METADATA_FILE")
    else:
        metadata_target = config["TEST_METADATA_FILE"]

    test_dataset = DiffusionSRDataset(
        preprocessed_data_dir=config["PREPROCESSED_DATA_DIR"],
        metadata_file=metadata_target,
        dem_patches_dir=config["DEM_DATA_DIR"],
        dem_stats=dem_stats,
        scaler_max_val=scaler_max_val,
        split="test",
        wet_dry_ratio=None,
    )

    # Fallback cascade for batch size keys between old and new architectures
    eval_batch_size = config.get("BATCH_SIZE", config.get("SR_BATCH_SIZE", 16))

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4),
        pin_memory=True,
    )
    print(
        f"Loaded {len(test_dataset)} samples for evaluation using batch size {eval_batch_size}."
    )
    return test_loader


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
    npz_save_path = os.path.join(output_dir, "per_feature_gamma_metrics.npz")
    np.savez_compressed(
        npz_save_path,
        r2_matrix=per_feature_metrics["r2_matrix"].values,
        mae_matrix=per_feature_metrics["mae_matrix"].values,
        var_matrix=per_feature_metrics["var_matrix"].values,
        quantiles=per_feature_metrics["quantiles"],
    )
    print(f"Per-feature analytical gamma metric matrices saved to: {npz_save_path}")
