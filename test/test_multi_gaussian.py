import numpy as np
import torch
import gudhi as gd
from skimage import measure
import matplotlib.pyplot as plt
import yaml
import sys
import os
import argparse

# --- Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)


def generate_multi_peak_gaussian(
    size=128,
    peak_centers=[(32, 32), (32, 96), (96, 32), (96, 96)],
    sigma=10.0,
    max_val=10.0,
):
    """Generates a 2D spatial field with symmetric Gaussian peaks."""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    field = np.zeros((size, size), dtype=np.float32)

    for cx, cy in peak_centers:
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        field += np.exp(-dist_sq / (2 * sigma**2))

    field = (field / field.max()) * max_val
    return field


def compute_tda_persistence(prec_2d_np_clean):
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    return cubical_complex.persistence()


def compute_gamma_matrix_single(
    prec_2d_data, physical_thresholds, pixel_size_km=2.0, thresh_b0=0.1, thresh_b1=0.1
):
    """In-memory adaptation of the geometric computation."""
    gamma_matrix = np.zeros((4, len(physical_thresholds)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    thresholds_broadcast_1d = physical_thresholds[np.newaxis, :]

    # Betti 0
    pairs_d0 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 0], dtype=np.float64
    )
    if pairs_d0.shape[0] > 0:
        births_0, deaths_0 = -pairs_d0[:, 0], -pairs_d0[:, 1]
        is_finite_0 = deaths_0 != -np.inf
        is_background_0 = ~is_finite_0
        deaths_0[is_background_0] = np.inf
        persistence_0 = births_0 - deaths_0
        persistence_0[is_background_0] = np.inf

        is_significant_0 = persistence_0 > thresh_b0
        pers_thresh_mask_0 = is_significant_0[:, np.newaxis]
        birth_thresh_mask_0 = births_0[:, np.newaxis] >= thresholds_broadcast_1d
        death_thresh_mask_0 = deaths_0[:, np.newaxis] < thresholds_broadcast_1d

        finite_pass_mask_0 = (
            pers_thresh_mask_0
            & birth_thresh_mask_0
            & death_thresh_mask_0
            & is_finite_0[:, np.newaxis]
        )
        background_pass_mask_0 = (
            pers_thresh_mask_0
            & birth_thresh_mask_0
            & is_background_0[:, np.newaxis]
            & (thresholds_broadcast_1d <= 0.01)
        )
        gamma_matrix[2, :] = np.sum(finite_pass_mask_0, axis=0) + np.sum(
            background_pass_mask_0, axis=0
        )

    # Betti 1
    pairs_d1 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 1], dtype=np.float64
    )
    if pairs_d1.shape[0] > 0:
        births_1, deaths_1 = -pairs_d1[:, 0], -pairs_d1[:, 1]
        persistence_1 = births_1 - deaths_1
        is_significant_1 = persistence_1 > thresh_b1
        pers_thresh_mask_1 = is_significant_1[:, np.newaxis]
        birth_thresh_mask_1 = births_1[:, np.newaxis] >= thresholds_broadcast_1d
        death_thresh_mask_1 = deaths_1[:, np.newaxis] < thresholds_broadcast_1d
        pass_mask_1 = pers_thresh_mask_1 & birth_thresh_mask_1 & death_thresh_mask_1
        gamma_matrix[3, :] = np.sum(pass_mask_1, axis=0)

    # Area and Perimeter
    pixel_area_km2 = pixel_size_km**2
    prec_broadcast = prec_2d_np_clean[..., np.newaxis]
    thresholds_broadcast_3d = physical_thresholds[np.newaxis, np.newaxis, :]
    masks_3d = prec_broadcast >= thresholds_broadcast_3d

    gamma_matrix[0, :] = np.sum(masks_3d, axis=(0, 1)) * pixel_area_km2

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, thresh in enumerate(physical_thresholds):
            mask_t = masks_3d[:, :, i]
            if not np.any(mask_t):
                continue
            contours = measure.find_contours(mask_t.astype(float), 0.5)
            perimeter_pixels = sum(
                np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in contours
            )
            gamma_matrix[1, i] = perimeter_pixels * pixel_size_km

    return gamma_matrix


def load_emulator(run_dir, arch, device):
    """Loads config, initializes model architecture, and loads weights."""
    if not run_dir or not os.path.exists(run_dir):
        return None, None

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    input_shape = (1, config.get("PATCH_SIZE", 128), config.get("PATCH_SIZE", 128))
    n_quantiles = len(config["QUANTILE_LEVELS"])
    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)

    if arch == "Baseline":
        model = BaselineCNN(n_quantiles=n_quantiles, input_shape=input_shape)
    elif arch == "Lipschitz":
        model = LipschitzCNN(n_quantiles=n_quantiles, input_shape=input_shape)
    elif arch == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=n_quantiles,
            input_shape=input_shape,
            quantile_levels=config["QUANTILE_LEVELS"],
            pixel_area_km2=pixel_size_km**2,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)
    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {arch} weights from {run_dir}")
    else:
        print(
            f"[WARNING] Checkpoint not found at {checkpoint_path}. Using random initialization."
        )

    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Test emulators on multi-peak Gaussian."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Run directory for Baseline model",
    )
    parser.add_argument(
        "--lipschitz_dir",
        type=str,
        default=None,
        help="Run directory for Lipschitz model",
    )
    parser.add_argument(
        "--constrained_dir",
        type=str,
        default=None,
        help="Run directory for Constrained model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the resulting plots",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Base Configuration Resolution
    run_dirs = [
        d
        for d in [args.baseline_dir, args.lipschitz_dir, args.constrained_dir]
        if d is not None
    ]
    if run_dirs:
        with open(os.path.join(run_dirs[0], "config.yaml"), "r") as f:
            base_config = yaml.safe_load(f)
    else:
        with open(os.path.join(parent_path, "config.yaml"), "r") as f:
            base_config = yaml.safe_load(f)

    physical_thresholds = np.array(base_config["QUANTILE_LEVELS"], dtype=np.float32)

    # 2. Data Generation and Normalization
    field_phys = generate_multi_peak_gaussian(
        size=base_config.get("PATCH_SIZE", 128), max_val=10.0
    )

    scaler_path = os.path.join(
        base_config.get("PREPROCESSED_DATA_DIR", ""), "log_precip_max_val.npy"
    )
    scaler_val = (
        float(np.load(scaler_path).item()) if os.path.exists(scaler_path) else 5.01
    )

    # Apply identical scaling to match ZarrMixupDataset processing
    field_normalized = np.clip(np.log1p(field_phys) / scaler_val, 0.0, 1.0)
    input_tensor = (
        torch.from_numpy(field_normalized)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(args.device)
    )

    # 3. Ground Truth Computation
    true_gamma = compute_gamma_matrix_single(field_phys, physical_thresholds)

    # 4. Model Instantiation & Inference
    models = {}
    if args.baseline_dir:
        models["Baseline"], _ = load_emulator(
            args.baseline_dir, "Baseline", args.device
        )
    if args.lipschitz_dir:
        models["Lipschitz"], _ = load_emulator(
            args.lipschitz_dir, "Lipschitz", args.device
        )
    if args.constrained_dir:
        models["Constrained"], _ = load_emulator(
            args.constrained_dir, "Constrained", args.device
        )

    predictions = {}
    with torch.no_grad():
        for name, model in models.items():
            if model is None:
                continue
            # Model outputs physical estimates of gamma
            pred_phys = model(input_tensor)
            predictions[name] = pred_phys.squeeze(0).cpu().numpy()

    # 5. Plot Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    components = ["Area", "Perimeter", "Betti 0 (Components)", "Betti 1 (Holes)"]

    for i, ax in enumerate(axes.flatten()):
        ax.step(
            physical_thresholds,
            true_gamma[i, :],
            label="True (Step/ReLU)",
            color="black",
            linewidth=2,
            where="post",
        )

        for name, pred_gamma in predictions.items():
            ax.plot(
                physical_thresholds,
                pred_gamma[i, :],
                label=f"{name} CNN",
                linestyle="--",
            )

        ax.set_title(components[i])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    comparison_path = os.path.join(args.save_dir, "synthetic_gaussian_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    print(f"\nSaved functional approximation plot to: {comparison_path}")
    plt.close(fig)

    # Plot Input Field
    plt.figure(figsize=(6, 5))
    plt.imshow(field_phys, cmap="viridis", origin="lower")
    plt.colorbar(label="Amplitude")
    plt.title("Synthetic Multi-Peak Gaussian Field")
    field_path = os.path.join(args.save_dir, "synthetic_gaussian_field.png")
    plt.savefig(field_path, dpi=300)
    print(f"Saved synthetic field plot to: {field_path}")
    plt.close()


if __name__ == "__main__":
    main()
