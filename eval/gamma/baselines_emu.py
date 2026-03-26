import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from data.dataset import ZarrMixupDataset
from eval.sr.metrics_lib_sr import compute_isoperimetric_violation


def compute_analytical_approximation(
    field_phys, thresholds, init_factor=0.1, min_temp=1e-3
):
    """
    Computes the differentiable analytical approximation of Minkowski functionals.
    field_phys: Tensor [B, 1, H, W] in physical space.
    thresholds: Array or Tensor of physical thresholds.
    """
    device = field_phys.device
    thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)

    base_temps = np.maximum(np.array(thresholds) * init_factor, min_temp)
    base_temps_tensor = torch.tensor(base_temps, dtype=torch.float32, device=device)

    areas, perimeters, eulers = [], [], []

    for q_idx, thresh in enumerate(thresholds_tensor):
        current_temp = base_temps_tensor[q_idx]

        p = torch.sigmoid((field_phys - thresh) / current_temp)

        area = torch.sum(p, dim=(1, 2, 3))

        dx = p[:, :, :, 1:] - p[:, :, :, :-1]
        dy = p[:, :, 1:, :] - p[:, :, :-1, :]
        perimeter = torch.sum(
            torch.sqrt(dx[:, :, :-1, :] ** 2 + dy[:, :, :, :-1] ** 2 + 1e-8),
            dim=(1, 2, 3),
        )

        V = torch.sum(p, dim=(1, 2, 3))
        E_x = torch.sum(p[:, :, :, :-1] * p[:, :, :, 1:], dim=(1, 2, 3))
        E_y = torch.sum(p[:, :, :-1, :] * p[:, :, 1:, :], dim=(1, 2, 3))
        F_faces = torch.sum(
            p[:, :, :-1, :-1] * p[:, :, :-1, 1:] * p[:, :, 1:, :-1] * p[:, :, 1:, 1:],
            dim=(1, 2, 3),
        )
        euler = V - E_x - E_y + F_faces

        areas.append(area)
        perimeters.append(perimeter)
        eulers.append(euler)

    pred_gamma_phys = torch.stack(
        [
            torch.stack(areas, dim=1),
            torch.stack(perimeters, dim=1),
            torch.stack(eulers, dim=1),
        ],
        dim=1,
    )

    pred_gamma_log = torch.sign(pred_gamma_phys) * torch.log1p(
        torch.abs(pred_gamma_phys)
    )
    return pred_gamma_log


def compute_gkf_expectations(dataset, physical_thresholds, scaler_val):
    """
    Computes GKF analytical expectations.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    sum_val, sum_sq, count = 0.0, 0.0, 0
    grad_var_sum, grad_count = 0.0, 0

    for inputs, _, _, _ in tqdm(loader, desc="Computing GKF expectations"):
        field = inputs.squeeze(1).numpy()

        sum_val += field.sum()
        sum_sq += (field**2).sum()
        count += field.size

        dy, dx = np.gradient(field, axis=(1, 2))
        grad_var_sum += (dy**2 + dx**2).sum() / 2.0
        grad_count += dy.size

    mu = sum_val / count
    sigma = np.sqrt((sum_sq / count) - mu**2)
    lambda_2 = (grad_var_sum / grad_count) / (sigma**2)

    log_thresh = np.log1p(physical_thresholds) / scaler_val
    u = (log_thresh - mu) / sigma

    e_area = 1.0 - norm.cdf(u)
    e_perimeter = (np.sqrt(lambda_2) / 4.0) * np.exp(-(u**2) / 2.0)
    e_euler = (lambda_2 / ((2 * np.pi) ** (1.5))) * u * np.exp(-(u**2) / 2.0)

    return np.vstack([e_area, e_perimeter, e_euler])


def train_pcr_baseline(train_dataset, n_components=50):
    """
    Fits Incremental PCA and Ridge Regression.
    """
    loader = DataLoader(train_dataset, batch_size=200, shuffle=False, num_workers=4)

    ipca = IncrementalPCA(n_components=n_components)
    print("Fitting Incremental PCA...")
    for inputs, _, _, _ in loader:
        X_batch = inputs.view(inputs.size(0), -1).numpy()
        ipca.partial_fit(X_batch)

    X_pca_list = []
    y_list = []

    print("Transforming data and accumulating targets...")
    for inputs, log_target_gamma, _, _ in loader:
        X_batch = inputs.view(inputs.size(0), -1).numpy()
        X_pca_list.append(ipca.transform(X_batch))

        y_batch = log_target_gamma.view(log_target_gamma.size(0), -1).numpy()
        y_list.append(y_batch)

    X_train = np.vstack(X_pca_list)
    y_train = np.vstack(y_list)

    print("Fitting Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    return ipca, ridge


def evaluate_predictions(y_true_log, y_pred_log, quantiles, feature_names):
    """
    Dynamically evaluates metrics based on the provided feature names.
    Bypasses PyTorch module dependency to allow arbitrary channel dimensions.
    """
    y_true_flat = y_true_log.reshape(y_true_log.shape[0], -1)
    y_pred_flat = y_pred_log.reshape(y_pred_log.shape[0], -1)

    metrics = {
        "MSE_Total": mean_squared_error(y_true_flat, y_pred_flat),
        "R2_Total": r2_score(y_true_flat, y_pred_flat),
        "Isoperimetric_Violation_Pct": compute_isoperimetric_violation(y_pred_log),
    }

    # Vectorized 1-Wasserstein (L1 Minkowski) approximation via trapezoidal integration
    abs_diff = np.abs(y_pred_log - y_true_log)
    dist = np.trapezoid(abs_diff, x=quantiles, axis=2)
    metrics["Minkowski_Total"] = dist.sum(axis=1).mean()

    for i, name in enumerate(feature_names):
        metrics[f"R2_{name}"] = r2_score(
            y_true_log[:, i, :].flatten(), y_pred_log[:, i, :].flatten()
        )
        metrics[f"Minkowski_{name}"] = dist[:, i].mean()

    return metrics


def evaluate_baselines(zarr_path, thresholds_path, scaler_val, n_components, quantiles):
    print("--- Initializing datasets ---")
    train_dataset = ZarrMixupDataset(
        zarr_path,
        split="train",
        scaler_val=scaler_val,
        include_original=True,
        include_mixup=False,
        augment=False,
    )
    test_dataset = ZarrMixupDataset(
        zarr_path,
        split="test",
        scaler_val=scaler_val,
        include_original=True,
        include_mixup=False,
        augment=False,
    )

    physical_thresholds = np.load(thresholds_path)

    # 1. PCR Baseline Execution (4 Channels)
    print("\n--- Running principal component regression (PCR) baseline ---")
    ipca, ridge = train_pcr_baseline(train_dataset, n_components=n_components)

    # 2. Test Set Inference and Target Construction
    print("\n--- Accumulating test targets and predictions ---")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    y_true_4ch_list = []
    x_test_list = []
    y_pred_analytical_list = []

    for inputs, log_target_gamma, _, _ in test_loader:
        y_true_4ch_list.append(log_target_gamma.numpy())
        x_test_list.append(inputs.view(inputs.size(0), -1).numpy())

        scaled_inputs = torch.clamp(inputs[:, 0:1, :, :] * scaler_val, max=7.0)
        phys_inputs = F.relu(torch.expm1(scaled_inputs))

        with torch.no_grad():
            gamma_analytical = compute_analytical_approximation(
                phys_inputs, physical_thresholds
            )
            y_pred_analytical_list.append(gamma_analytical.numpy())

    y_true_log_4ch = np.concatenate(y_true_4ch_list, axis=0)
    x_test_flat = np.concatenate(x_test_list, axis=0)
    y_pred_analytical = np.concatenate(y_pred_analytical_list, axis=0)

    # Compute true Euler characteristic in physical space and re-transform
    y_true_raw = np.sign(y_true_log_4ch) * np.expm1(np.abs(y_true_log_4ch))
    true_euler_raw = y_true_raw[:, 2, :] - y_true_raw[:, 3, :]
    true_euler_log = np.sign(true_euler_raw) * np.log1p(np.abs(true_euler_raw))

    y_true_log_3ch = np.stack(
        [y_true_log_4ch[:, 0, :], y_true_log_4ch[:, 1, :], true_euler_log], axis=1
    )

    # Transform PCA and predict
    x_pca = ipca.transform(x_test_flat)
    y_pred_pcr_flat = ridge.predict(x_pca)
    y_pred_pcr = y_pred_pcr_flat.reshape(y_true_log_4ch.shape)

    # 4. Final Evaluation Printout
    print("\n=============================================")
    print("  Linear Statistical Baseline (PCR) Metrics  ")
    print("=============================================")
    pcr_metrics = evaluate_predictions(
        y_true_log_4ch, y_pred_pcr, quantiles, ["Area", "Perimeter", "Betti0", "Betti1"]
    )
    for k, v in pcr_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=============================================")
    print("      Analytical Approximation Metrics       ")
    print("=============================================")
    ana_metrics = evaluate_predictions(
        y_true_log_3ch, y_pred_analytical, quantiles, ["Area", "Perimeter", "Euler"]
    )
    for k, v in ana_metrics.items():
        print(f"{k}: {v:.4f}")

    # Dump results to a YAML file for later analysis
    results = {
        "PCR": pcr_metrics,
        "Analytical_Approximation": ana_metrics,
    }
    with open(os.path.join(parent_path, "baseline_evaluation_results.yaml"), "w") as f:
        yaml.dump(results, f)
    print("\nBaseline evaluation results saved to baseline_evaluation_results.yaml")


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_baselines.py config.yaml")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    thresholds_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy"
    )

    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path).item())
    else:
        print(f"Scaler file not found at {scaler_path}. Using default value of 5.01.")
        scaler_val = 5.01

    n_components = config.get("PCR_COMPONENTS", 50)
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)

    evaluate_baselines(zarr_path, thresholds_path, scaler_val, n_components, quantiles)


if __name__ == "__main__":
    main()
