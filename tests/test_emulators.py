import os
import sys
import yaml
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import gudhi as gd
from skimage import measure

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)
from data.dataset import ZarrMixupDataset

# --- Global Constants ---
PREPROCESSED_DATA_DIR = config.get("PREPROCESSED_DATA_DIR", "")
QUANTILE_LEVELS = config.get("QUANTILE_LEVELS", np.linspace(0.01, 0.99, 30))
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config.get("PATCH_SIZE", 128)
INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
BATCH_SIZE = config.get("BATCH_SIZE", 16)
NUM_WORKERS = config.get("NUM_WORKERS", 4)


# ==========================================
# 1. Exact Topological Computation Tools
# ==========================================
def compute_tda_persistence(prec_2d_np_clean):
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    return cubical_complex.persistence()


def compute_gamma_matrix_single(
    prec_2d_data, physical_thresholds, pixel_size_km=2.0, thresh_b0=0.1
):
    """In-memory exact topological computation adapted for A, P, and Betti 0."""
    gamma_matrix = np.zeros((3, len(physical_thresholds)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    thresholds_broadcast_1d = physical_thresholds[np.newaxis, :]

    b0 = np.zeros(len(physical_thresholds))

    # Betti 0 calculation
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
        mask_0 = (
            (is_significant_0[:, None])
            & (births_0[:, None] >= thresholds_broadcast_1d)
            & ((deaths_0[:, None] < thresholds_broadcast_1d) | is_background_0[:, None])
        )
        b0 = np.sum(mask_0, axis=0)

    # Minkowski functional 3 proxy: Betti 0 (Connected Components)
    gamma_matrix[2, :] = b0

    # Area and perimeter
    pixel_area_km2 = pixel_size_km**2
    prec_broadcast = prec_2d_np_clean[..., np.newaxis]
    thresholds_broadcast_3d = physical_thresholds[np.newaxis, np.newaxis, :]
    masks_3d = prec_broadcast >= thresholds_broadcast_3d

    gamma_matrix[0, :] = np.sum(masks_3d, axis=(0, 1)) * pixel_area_km2

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


def get_true_scalar_metric(p_phys_tensor, physical_threshold, metric_idx):
    """Wrapper to interface 4D PyTorch tensors with 2D exact topological solver."""
    p_2d = p_phys_tensor.detach().squeeze().cpu().numpy()
    gamma_mat = compute_gamma_matrix_single(
        p_2d,
        np.array([physical_threshold], dtype=np.float32),
        pixel_size_km=PIXEL_SIZE_KM,
    )
    return gamma_mat[metric_idx, 0]


# ==========================================
# 2. Model & Data Initialization
# ==========================================
def load_emulator(arch, checkpoint_path, device):
    """Initializes model architecture and loads weights."""
    if arch == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif arch == "Lipschitz":
        model = LipschitzCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif arch == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        new_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(new_state_dict)
        print(f"Loaded {arch} weights from {checkpoint_path}")
    else:
        print(
            f"[WARNING] Checkpoint not found at {checkpoint_path}. Using random initialization."
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def generate_multi_peak_gaussian(
    size=128,
    peak_centers=[(32, 32), (32, 96), (96, 32), (96, 96)],
    sigma=10.0,
    max_val=10.0,
):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    field = np.zeros((size, size), dtype=np.float32)
    for cx, cy in peak_centers:
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        field += np.exp(-dist_sq / (2 * sigma**2))
    field = (field / field.max()) * max_val
    return field


def get_scaler_val():
    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    if os.path.exists(scaler_path):
        return float(np.load(scaler_path).item())
    return 5.01


# ==========================================
# 3. Test Suites
# ==========================================


# --- A. Amplitude Perturbation ---
def test_amplitude_perturbation(
    model,
    p_phys,
    cell_mask,
    scaler_val,
    physical_threshold=1.0,
    metric_idx=0,
    quantile_idx=15,
    save_dir=None,
):
    model.eval()
    alphas = np.linspace(0.5, 1.5, 50)
    true_metrics, pred_metrics = [], []

    print("Computing amplitude perturbation landscape...")
    for alpha in tqdm(alphas):
        p_perturbed = p_phys.clone()
        p_perturbed[cell_mask] *= alpha

        true_val = get_true_scalar_metric(p_perturbed, physical_threshold, metric_idx)
        true_metrics.append(true_val)

        x_norm = torch.log1p(p_perturbed) / scaler_val
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        with torch.no_grad():
            preds = model(x_norm)
            pred_metrics.append(preds[0, metric_idx, quantile_idx].item())

    plt.figure(figsize=(8, 5))
    plt.plot(
        alphas,
        true_metrics,
        drawstyle="steps-post",
        label="True discrete metric",
        color="black",
    )
    plt.plot(
        alphas, pred_metrics, label="Emulator prediction", color="red", linewidth=2
    )
    plt.xlabel(r"Physical perturbation factor $\alpha$")
    plt.ylabel("Topological metric")
    plt.title("1D Amplitude Perturbation Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "amplitude_perturbation.png"), dpi=300)
    plt.close()


# --- B. Spatial Gradient Attribution ---
def test_spatial_attribution(
    model, p_phys, scaler_val, metric_idx=0, quantile_idx=15, save_dir=None
):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    x_norm = torch.log1p(p_phys) / scaler_val
    x_norm = torch.clamp(x_norm, 0.0, 1.0)
    x_norm.requires_grad_(True)

    preds = model(x_norm)
    target_pred = preds[0, metric_idx, quantile_idx]

    model.zero_grad()
    target_pred.backward()

    saliency_map = x_norm.grad.data.abs().squeeze().cpu().numpy()
    p_input = p_phys.detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(p_input, cmap="Blues")
    axes[0].set_title("Physical precipitation field")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(saliency_map, cmap="hot")
    axes[1].set_title(r"Surrogate receptive field ($\nabla_x f_\theta(x)$)")
    plt.colorbar(im2, ax=axes[1])
    if save_dir:
        plt.savefig(os.path.join(save_dir, "spatial_attribution.png"), dpi=300)
    plt.close()


# --- C. Surrogate Gradient Alignment ---
def test_gradient_alignment(
    model,
    p_phys,
    scaler_val,
    physical_threshold=1.0,
    metric_idx=0,
    quantile_idx=15,
    save_dir=None,
):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True

    x_norm = torch.log1p(p_phys) / scaler_val
    x_norm = torch.clamp(x_norm, 0.0, 1.0)
    x_norm.requires_grad_(True)

    pred = model(x_norm)[0, metric_idx, quantile_idx]
    model.zero_grad()
    pred.backward()
    grad_analytical = x_norm.grad.view(-1).cpu().numpy()
    x_norm.requires_grad_(False)

    epsilons = np.logspace(-4, -1, 15)
    n_samples = 50
    correlations = []

    print("Computing surrogate gradient alignment calibration...")

    directions = []
    for _ in range(n_samples):
        v = torch.randn_like(x_norm)
        v = v / torch.norm(v)
        directions.append(v)

    for epsilon in tqdm(epsilons, desc="Epsilon Calibration"):
        similarities = []
        for v in directions:
            x_plus = x_norm + epsilon * v
            x_minus = x_norm - epsilon * v

            p_plus = torch.expm1(x_plus * scaler_val)
            p_minus = torch.expm1(x_minus * scaler_val)

            f_plus = get_true_scalar_metric(p_plus, physical_threshold, metric_idx)
            f_minus = get_true_scalar_metric(p_minus, physical_threshold, metric_idx)
            dir_deriv_true = (f_plus - f_minus) / (2 * epsilon)

            v_np = v.view(-1).cpu().numpy()
            dir_deriv_analytical = np.dot(grad_analytical, v_np)
            similarities.append((dir_deriv_true, dir_deriv_analytical))

        sim_arr = np.array(similarities)
        active_mask = (sim_arr[:, 0] != 0) | (sim_arr[:, 1] != 0)

        if not np.any(active_mask):
            correlations.append(0.0)
        else:
            true_derivs = sim_arr[active_mask, 0]
            surrogate_derivs = sim_arr[active_mask, 1]

            if np.std(true_derivs) == 0 or np.std(surrogate_derivs) == 0:
                correlations.append(0.0)
            else:
                correlation = np.corrcoef(true_derivs, surrogate_derivs)[0, 1]
                if np.isnan(correlation):
                    correlations.append(0.0)
                else:
                    correlations.append(correlation)

    plt.figure(figsize=(7, 5))
    plt.plot(epsilons, correlations, marker="o", linestyle="-", color="purple")
    plt.xscale("log")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel(r"Finite difference step size ($\epsilon$)")
    plt.ylabel("Pearson correlation ($r$)")
    plt.title("Surrogate Gradient Alignment Calibration")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, "gradient_alignment_calibration.png"), dpi=300
        )
    plt.close()


# --- D. Trajectory Alignment ---
def test_trajectory_alignment(
    model,
    p_phys_init,
    scaler_val,
    y_target,
    physical_threshold=1.0,
    metric_idx=0,
    quantile_idx=15,
    n_steps=100,
    lr=0.05,
    save_dir=None,
    method="projected",
):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if method == "projected" or method == "regularized":
        x_norm = torch.log1p(p_phys_init) / scaler_val
        x_norm = torch.clamp(x_norm, 0.0, 1.0).detach().clone()
        x_norm.requires_grad_(True)
        optimizer = optim.Adam([x_norm], lr=lr)

    elif method == "reparameterized":
        x_norm_initial = torch.clamp(
            torch.log1p(p_phys_init) / scaler_val, 1e-4, 1 - 1e-4
        )
        z = torch.log(x_norm_initial / (1 - x_norm_initial)).detach().clone()
        z.requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    criterion = nn.MSELoss()
    target_tensor = torch.tensor(
        [y_target], dtype=torch.float32, device=p_phys_init.device
    )

    history_surrogate = []
    history_exact = []

    print(f"Starting trajectory alignment ({method} method)...")

    for _ in tqdm(range(n_steps), desc=f"Optimizing ({method})"):
        optimizer.zero_grad()

        if method == "reparameterized":
            x_current = torch.sigmoid(z)
        else:
            x_current = x_norm

        preds = model(x_current)
        pred_val = preds[0, metric_idx, quantile_idx]

        loss = criterion(pred_val, target_tensor)

        if method == "regularized":
            tv_loss = torch.sum(
                torch.abs(x_current[:, :, :, :-1] - x_current[:, :, :, 1:])
            ) + torch.sum(torch.abs(x_current[:, :, :-1, :] - x_current[:, :, 1:, :]))
            loss += 0.01 * tv_loss

        loss.backward()
        optimizer.step()

        if method == "projected" or method == "regularized":
            with torch.no_grad():
                x_norm.clamp_(0.0, 1.0)

        with torch.no_grad():
            if method == "reparameterized":
                x_eval = torch.sigmoid(z)
            else:
                x_eval = x_norm

            p_current = torch.expm1(x_eval * scaler_val)
            exact_val = get_true_scalar_metric(
                p_current, physical_threshold, metric_idx
            )

            history_surrogate.append(pred_val.item())
            history_exact.append(exact_val)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(n_steps),
        history_surrogate,
        label=r"Surrogate prediction $f_\theta(x_t)$",
        color="blue",
        alpha=0.8,
    )
    plt.plot(
        range(n_steps),
        history_exact,
        label="Exact physical metric $F(x_t)$",
        color="black",
        linewidth=2,
    )
    plt.axhline(
        y_target, color="red", linestyle="--", label=r"Target $y_{\text{target}}$"
    )

    plt.xlabel("Gradient descent steps")
    plt.ylabel("Topological metric")
    plt.title(f"Optimization Trajectory Alignment ({method})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"trajectory_alignment_{method}.png"), dpi=300
        )
    plt.close()

    return history_exact, history_surrogate


# --- E. Lipschitz Constants Evaluation ---
class LatentEvaluator(nn.Module):
    def __init__(self, model, arch_name):
        super().__init__()
        self.model = model
        self.latent = None
        target_layer = self.model.features if arch_name == "Baseline" else self.model.fc
        self.handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.latent = outputs.flatten(1) if outputs.dim() > 2 else outputs

    def forward(self, x):
        _ = self.model(x)
        return self.latent

    def __del__(self):
        if hasattr(self, "handle"):
            self.handle.remove()


def empirical_lipschitz_batched(model, x, epsilon=1e-2, steps=10, init_noise=1e-4):
    model.eval()
    x_orig = x.clone().detach()
    x_pert = x_orig.clone().detach() + (torch.rand_like(x_orig) * 2 - 1) * init_noise
    x_pert.clamp_(0, 1)
    x_pert.requires_grad = True

    y_orig = model(x_orig).detach()

    for _ in range(steps):
        model.zero_grad()
        y_pert = model(x_pert)
        dist = torch.linalg.vector_norm(
            y_pert - y_orig, dim=tuple(range(1, y_pert.ndim))
        )
        loss = -dist.sum()
        loss.backward()

        with torch.no_grad():
            x_pert += epsilon * torch.sign(x_pert.grad)
            x_pert.clamp_(0, 1)
            x_pert.grad.zero_()
            x_pert.requires_grad = True

    y_final = model(x_pert)
    dx = torch.linalg.vector_norm(x_pert - x_orig, dim=tuple(range(1, x_pert.ndim)))
    dy = torch.linalg.vector_norm(y_final - y_orig, dim=tuple(range(1, y_final.ndim)))
    return dy / (dx + 1e-8)


def test_lipschitz_constants(models_dict, scaler_val, device):
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
    dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    latent_models = {
        name: LatentEvaluator(model, name).to(device)
        for name, model in models_dict.items()
    }
    results = {name: [] for name in latent_models}

    for batch in tqdm(dataloader, desc="Evaluating Lipschitz constants"):
        x = batch[0].to(device)
        for name, model in latent_models.items():
            for param in model.parameters():
                param.requires_grad = False
            batch_constants = empirical_lipschitz_batched(model, x)
            results[name].append(batch_constants.max().item())

    print("\n--- Lipschitz Evaluation Results ---")
    for name, vals in results.items():
        print(f"Model: {name} | Max empirical Lipschitz (latent): {max(vals):.4f}")


# --- F. Synthetic Multi-Peak Gaussian Test ---
def test_synthetic_gaussian(models_dict, scaler_val, device, save_dir=None):
    field_phys = generate_multi_peak_gaussian(size=PATCH_SIZE, max_val=10.0)
    physical_thresholds = np.array(QUANTILE_LEVELS, dtype=np.float32)
    field_normalized = np.clip(np.log1p(field_phys) / scaler_val, 0.0, 1.0)
    input_tensor = (
        torch.from_numpy(field_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    )

    true_gamma = compute_gamma_matrix_single(
        field_phys, physical_thresholds, PIXEL_SIZE_KM
    )

    predictions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            predictions[name] = model(input_tensor).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = ["Area (A)", "Perimeter (P)", "Betti 0 (CCs)"]

    for i, ax in enumerate(axes):
        ax.step(
            physical_thresholds,
            true_gamma[i, :],
            label="True",
            color="black",
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
        ax.set_xlabel("Threshold (mm/h)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            os.path.join(save_dir, "synthetic_gaussian_comparison.png"), dpi=300
        )
    plt.close()


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Unified Emulator Testing Suite")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_dir", type=str, default="./test_results")
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "perturb", "gradients", "trajectory", "lipschitz", "synthetic"],
        default="all",
    )

    # Checkpoint Paths
    parser.add_argument("--ckpt_baseline", type=str, default=None)
    parser.add_argument("--ckpt_lipschitz", type=str, default=None)
    parser.add_argument("--ckpt_constrained", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    scaler_val = get_scaler_val()

    # Load active models
    models_dict = {}
    if args.ckpt_baseline:
        models_dict["Baseline"] = load_emulator(
            "Baseline", args.ckpt_baseline, args.device
        )
    if args.ckpt_lipschitz:
        models_dict["Lipschitz"] = load_emulator(
            "Lipschitz", args.ckpt_lipschitz, args.device
        )
    if args.ckpt_constrained:
        models_dict["Constrained"] = load_emulator(
            "Constrained", args.ckpt_constrained, args.device
        )

    if not models_dict:
        print(
            "[WARNING] No checkpoints provided. Initializing untrained Constrained model for tests."
        )
        models_dict["Constrained"] = load_emulator("Constrained", None, args.device)

    if args.test in ["all", "lipschitz"]:
        test_lipschitz_constants(models_dict, scaler_val, args.device)

    if args.test in ["all", "synthetic"]:
        test_synthetic_gaussian(models_dict, scaler_val, args.device, args.save_dir)

    if args.test in ["all", "perturb", "gradients", "trajectory"]:
        # Generate base field for local perturbation/gradient tests
        sample_phys = (
            torch.tensor(generate_multi_peak_gaussian(size=PATCH_SIZE, max_val=10.0))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(args.device)
        )
        convective_mask = torch.zeros_like(sample_phys, dtype=torch.bool)
        convective_mask[0, 0, 22:42, 22:42] = True  # Mask over one of the peaks

        test_model = next(iter(models_dict.values()))  # Use the first available model

        if args.test in ["all", "perturb"]:
            test_amplitude_perturbation(
                test_model,
                sample_phys,
                convective_mask,
                scaler_val,
                save_dir=args.save_dir,
            )

        if args.test in ["all", "gradients"]:
            test_spatial_attribution(
                test_model, sample_phys, scaler_val, save_dir=args.save_dir
            )
            test_gradient_alignment(
                test_model, sample_phys, scaler_val, save_dir=args.save_dir
            )

        if args.test in ["all", "trajectory"]:
            # Define an arbitrary optimization target to test trajectory convergence
            initial_metric = get_true_scalar_metric(
                sample_phys, physical_threshold=1.0, metric_idx=0
            )
            target_metric = initial_metric * 1.5  # Aim to increase the metric by 50%

            for method in ["projected", "reparameterized", "regularized"]:
                test_trajectory_alignment(
                    model=test_model,
                    p_phys_init=sample_phys,
                    scaler_val=scaler_val,
                    y_target=target_metric,
                    physical_threshold=1.0,
                    metric_idx=0,
                    quantile_idx=15,
                    n_steps=100,
                    lr=0.05,
                    save_dir=args.save_dir,
                    method=method,
                )


if __name__ == "__main__":
    main()
