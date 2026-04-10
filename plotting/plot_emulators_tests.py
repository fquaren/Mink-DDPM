"""
plot_manuscript_figures.py
Execution: python plot_manuscript_figures.py --ckpt_baseline <path> --ckpt_lipschitz <path> --ckpt_constrained <path>
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import sys
import matplotlib.patches as patches


# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

from tests.test_emulators import (
    get_scaler_val,
    generate_multi_peak_gaussian,
    get_true_scalar_metric,
    compute_gamma_matrix_single,
)
from tests.test_emulators import (
    load_emulator,
    empirical_lipschitz_batched,
    LatentEvaluator,
)
from data.dataset import ZarrMixupDataset
from torch.utils.data import DataLoader

PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PIXEL_SIZE_KM = config["PIXEL_SIZE_KM"]

# ==========================================
# Shared Visualization Styles
# ==========================================

DISPLAY_NAMES = {
    "Baseline": "CNN (Unconstr.)",
    "Lipschitz": "Lip-CNN (Unconstr.)",
    "Constrained": "Lip-CNN (Constr.)",
}

MODEL_STYLES = {
    "Target": {"color": "black", "ls": "-", "marker": "o", "lw": 2, "alpha": 0.8},
    "CNN (Unconstr.)": {"color": "#648fff", "ls": "--", "marker": "x", "lw": 1},
    "Lip-CNN (Unconstr.)": {"color": "#ffb000", "ls": "-.", "marker": "^", "lw": 1},
    "Lip-CNN (Constr.)": {"color": "#dc2680", "ls": ":", "marker": "s", "lw": 1},
}


def plot_mechanistic_proof(model, device, scaler_val, save_dir):
    """Figure 1: Original Field, Saliency, and 1D Amplitude Perturbations"""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5))

    # --- Data Setup ---
    sample_phys = (
        torch.tensor(generate_multi_peak_gaussian(size=PATCH_SIZE, max_val=10.0))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Define perturbation region (Targeting the peak at 32, 32)
    x_start, x_end, y_start, y_end = 22, 42, 22, 42
    cell_mask = torch.zeros_like(sample_phys, dtype=torch.bool)
    cell_mask[0, 0, x_start:x_end, y_start:y_end] = True

    # --- CALIBRATION FIX ---
    # Scale the targeted peak so its baseline maximum is exactly 5.0
    sample_phys[cell_mask] *= 0.5

    # Set the threshold exactly to the targeted peak's baseline maximum
    physical_threshold = 5.0
    # -----------------------

    alphas = np.linspace(0.5, 1.5, 50)
    quantile_idx = 15  # Median

    # Dictionaries to store metrics for a single batched computation
    true_metrics = {i: [] for i in range(3)}
    pred_metrics = {i: [] for i in range(3)}

    gamma_labels = ["Area (km²)", "Perimeter (km)", "Betti 0 (CCs)"]

    print("Computing multi-metric amplitude landscape...")
    for alpha in alphas:
        p_perturbed = sample_phys.clone()
        p_perturbed[cell_mask] *= alpha

        # Exact TDA (Requires physical scale)
        for i in range(3):
            true_metrics[i].append(
                get_true_scalar_metric(p_perturbed, physical_threshold, i)
            )

        # Surrogate Prediction (Requires normalized scale)
        x_norm = torch.clamp(torch.log1p(p_perturbed) / scaler_val, 0.0, 1.0)
        with torch.no_grad():
            preds = model(x_norm)[0, :, quantile_idx].cpu().numpy()
            for i in range(3):
                pred_metrics[i].append(preds[i])

    # --- Panel A: Original Field ---
    ax_orig = axes[0, 0]
    orig_img = sample_phys.squeeze().cpu().numpy()
    im_orig = ax_orig.imshow(orig_img, cmap="Blues", origin="lower")

    # Draw a rectangle to indicate the perturbed region
    rect = patches.Rectangle(
        (y_start, x_start),
        y_end - y_start,
        x_end - x_start,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
    )
    ax_orig.add_patch(rect)
    ax_orig.set_title("Original Field & Mask")
    ax_orig.axis("off")
    fig.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04, label="mm/h")

    # --- Panel B: Spatial Saliency (Area) ---
    ax_sal = axes[0, 1]
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    x_norm = torch.clamp(
        torch.log1p(sample_phys) / scaler_val, 0.0, 1.0
    ).requires_grad_(True)
    # Using Area (index 0) as the representative spatial gradient
    target_pred = model(x_norm)[0, 0, quantile_idx]
    model.zero_grad()
    target_pred.backward()

    saliency_map = x_norm.grad.data.abs().squeeze().cpu().numpy()

    im_sal = ax_sal.imshow(saliency_map, cmap="hot", origin="lower")
    ax_sal.set_title(r"Gradient Envelope ($\nabla_x f_\theta$)")
    ax_sal.axis("off")
    fig.colorbar(im_sal, ax=ax_sal, fraction=0.046, pad=0.04)

    # --- Panels C-E: 1D Perturbations ---
    # Map axes to metric indices, omitting Betti 1 (index 3)
    ax_map = {0: axes[0, 2], 1: axes[1, 0], 2: axes[1, 1]}
    disp_name = DISPLAY_NAMES["Constrained"]

    # Iterate strictly through indices 0 to 2
    for i in range(3):
        ax = ax_map[i]

        # Plot Target style
        ax.plot(
            alphas,
            true_metrics[i],
            drawstyle="steps-post",
            label="Target",
            **MODEL_STYLES["Target"],
        )

        # Plot Constrained Model style
        ax.plot(alphas, pred_metrics[i], label=disp_name, **MODEL_STYLES[disp_name])

        ax.set_xlabel(r"Amplitude Scaling ($\alpha$)")
        ax.set_title(gamma_labels[i])
        ax.grid(True, linestyle="--", alpha=0.5)

        # Only put the legend on the Area plot to save space
        if i == 0:
            ax.legend(loc="best", fontsize=7)

    # Turn off the empty axis that was reserved for Betti 1
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_mechanistic_proof.pdf"), dpi=300)
    plt.close()


def plot_surrogate_quality(models_dict, device, scaler_val, save_dir, n_batches=5):
    """Figure 2: Gradient Alignment Distribution & Lipschitz Constants"""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3))

    # Setup Dataloader for statistical evaluation
    zarr_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_dataset.zarr")

    test_dataset = ZarrMixupDataset(
        zarr_path=zarr_path,
        split="test",
        scaler_val=scaler_val,
        augment=False,
        include_original=True,
        include_mixup=False,
        subset_fraction=0.1,
    )
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # --- Panel A: Cosine Similarity Distribution ---
    model = models_dict.get("Constrained")
    for param in model.parameters():
        param.requires_grad = True

    similarities = []
    epsilon = 1e-2
    metric_idx, quantile_idx, physical_threshold = 0, 15, 1.0

    iterator = iter(dataloader)
    for _ in tqdm(range(n_batches), desc="Computing Jacobian Alignments"):
        batch = next(iterator)[0].to(device)  # Normalized input
        for i in range(batch.shape[0]):
            x_norm = batch[i : i + 1].clone().requires_grad_(True)
            pred = model(x_norm)[0, metric_idx, quantile_idx]
            model.zero_grad()
            pred.backward()
            grad_analytical = x_norm.grad.view(-1).cpu().numpy()

            # Stochastic Directional Derivative
            v = torch.randn_like(x_norm)
            v = v / torch.norm(v)

            p_plus = torch.expm1((x_norm + epsilon * v) * scaler_val)
            p_minus = torch.expm1((x_norm - epsilon * v) * scaler_val)

            f_plus = get_true_scalar_metric(p_plus, physical_threshold, metric_idx)
            f_minus = get_true_scalar_metric(p_minus, physical_threshold, metric_idx)
            dir_deriv_true = (f_plus - f_minus) / (2 * epsilon)

            dir_deriv_analytical = np.dot(grad_analytical, v.view(-1).cpu().numpy())

            if dir_deriv_true != 0 or dir_deriv_analytical != 0:
                similarities.append((dir_deriv_true, dir_deriv_analytical))

    sim_arr = np.array(similarities)
    if len(sim_arr) > 1:
        corr = np.corrcoef(sim_arr[:, 0], sim_arr[:, 1])[0, 1]
        axes[0].scatter(
            sim_arr[:, 0],
            sim_arr[:, 1],
            alpha=0.5,
            s=10,
            color=MODEL_STYLES["Lip-CNN (Constr.)"]["color"],
        )
        axes[0].set_title(f"Gradient Alignment (r={corr:.2f})")
        axes[0].set_xlabel("Empirical Directional Derivative")
        axes[0].set_ylabel("Surrogate Directional Derivative")

    # --- Panel B: Lipschitz Constants ---
    latent_models = {
        name: LatentEvaluator(m, name).to(device) for name, m in models_dict.items()
    }
    lip_results = {name: [] for name in latent_models}

    iterator = iter(dataloader)
    for _ in tqdm(range(n_batches), desc="Computing Lipschitz Constraints"):
        batch = next(iterator)[0].to(device)
        for name, l_model in latent_models.items():
            for param in l_model.parameters():
                param.requires_grad = False
            batch_constants = empirical_lipschitz_batched(l_model, batch)

            lip_results[name].extend(batch_constants.detach().cpu().numpy())

    names = list(lip_results.keys())
    data = [lip_results[n] for n in names]

    # Apply standard display names to boxplot labels
    display_labels = [DISPLAY_NAMES[n] for n in names]
    axes[1].boxplot(data, labels=display_labels)
    axes[1].set_ylabel("Empirical Lipschitz Constant")
    axes[1].set_yscale("log")
    axes[1].set_title("Latent Space Regularization")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_surrogate_quality.pdf"), dpi=300)
    plt.close()


def plot_functional_approximation(models_dict, device, scaler_val, save_dir):
    """Figure 3: Functional Approximation across Thresholds (2x2 Grid)"""
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6))

    field_phys = generate_multi_peak_gaussian(size=PATCH_SIZE, max_val=10.0)
    physical_thresholds = np.array(QUANTILE_LEVELS, dtype=np.float32)
    input_tensor = (
        torch.clamp(
            torch.log1p(torch.from_numpy(field_phys).float()) / scaler_val, 0.0, 1.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    true_gamma = compute_gamma_matrix_single(
        field_phys, physical_thresholds, pixel_size_km=PIXEL_SIZE_KM
    )

    predictions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            predictions[name] = model(input_tensor).squeeze(0).cpu().numpy()

    gamma_labels = ["Area (km²)", "Perimeter (km)", "Betti 0 (CCs)", "Betti 1 (Holes)"]

    # Iterate over only the first 3 metrics, hiding the last subplot
    for i, ax in enumerate(axes.flatten()[:3]):
        # Plot Target Style
        ax.plot(
            physical_thresholds,
            true_gamma[i, :],
            drawstyle="steps-post",
            label="Target",
            **MODEL_STYLES["Target"],
        )

        # Plot Model Styles
        for name, pred_gamma in predictions.items():
            disp_name = DISPLAY_NAMES[name]
            ax.plot(
                physical_thresholds,
                pred_gamma[i, :],
                label=disp_name,
                **MODEL_STYLES[disp_name],
            )

        ax.set_title(gamma_labels[i])
        ax.set_xlabel("Precipitation Threshold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", frameon=True)

    # Hide the 4th subplot completely
    axes.flatten()[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_functional_approximation.pdf"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_dir", type=str, default="./manuscript_figures")
    parser.add_argument("--ckpt_baseline", type=str, required=True)
    parser.add_argument("--ckpt_lipschitz", type=str, required=True)
    parser.add_argument("--ckpt_constrained", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    scaler_val = get_scaler_val()

    print("Loading models...")
    models = {
        "Baseline": load_emulator("Baseline", args.ckpt_baseline, args.device),
        "Lipschitz": load_emulator("Lipschitz", args.ckpt_lipschitz, args.device),
        "Constrained": load_emulator("Constrained", args.ckpt_constrained, args.device),
    }

    print("Generating Figure 1: Mechanistic Proof...")
    plot_mechanistic_proof(
        models["Constrained"], args.device, scaler_val, args.save_dir
    )

    print("Generating Figure 2: Surrogate Quality...")
    plot_surrogate_quality(models, args.device, scaler_val, args.save_dir)

    print("Generating Figure 3: Functional Approximation...")
    plot_functional_approximation(models, args.device, scaler_val, args.save_dir)

    print(f"All figures saved to {args.save_dir} as PDFs.")
