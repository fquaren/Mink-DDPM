import os
import sys
import yaml
import torch
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
from eval.metrics import compute_isoperimetric_violation
from src.loss import MinkowskiLoss


def compute_gkf_expectations(dataset, physical_thresholds, scaler_val):
    """
    Computes GKF analytical expectations.
    Assumes the log-transformed, standardized field approximates a Gaussian random field.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    sum_val, sum_sq, count = 0.0, 0.0, 0
    grad_var_sum, grad_count = 0.0, 0

    for inputs, _, _, _ in tqdm(loader):
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
    Fits Incremental PCA and Ridge Regression to map flattened fields to gamma targets.
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


def compute_minkowski_loss(pred_log, target_log, quantiles):
    """Computes the 1-Wasserstein (L1 Minkowski) distance over quantiles."""
    pred_tensor = torch.from_numpy(pred_log).float()
    target_tensor = torch.from_numpy(target_log).float()
    quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)

    criterion = MinkowskiLoss(quantiles_tensor)

    weighted_total_loss, loss_A, loss_P, loss_CC = criterion(pred_tensor, target_tensor)

    return {
        "Minkowski_Total": weighted_total_loss.mean().item(),
        "Minkowski_Area": loss_A.mean().item(),
        "Minkowski_Perimeter": loss_P.mean().item(),
        "Minkowski_CC": loss_CC.mean().item(),
    }


def evaluate_predictions(y_true_log, y_pred_log, quantiles):
    """Computes R^2, MSE, Minkowski loss, and Isoperimetric violation."""
    y_true_flat = y_true_log.reshape(y_true_log.shape[0], -1)
    y_pred_flat = y_pred_log.reshape(y_pred_log.shape[0], -1)

    mse_total = mean_squared_error(y_true_flat, y_pred_flat)
    r2_total = r2_score(y_true_flat, y_pred_flat)

    r2_A = r2_score(y_true_log[:, 0, :].flatten(), y_pred_log[:, 0, :].flatten())
    r2_P = r2_score(y_true_log[:, 1, :].flatten(), y_pred_log[:, 1, :].flatten())
    r2_CC = r2_score(y_true_log[:, 2, :].flatten(), y_pred_log[:, 2, :].flatten())

    mink_metrics = compute_minkowski_loss(y_pred_log, y_true_log, quantiles)
    violation_rate = compute_isoperimetric_violation(y_pred_log)

    metrics = {
        "MSE_Total": mse_total,
        "R2_Total": r2_total,
        "R2_Area": r2_A,
        "R2_Perimeter": r2_P,
        "R2_CC": r2_CC,
        "Isoperimetric_Violation_Pct": violation_rate,
    }
    metrics.update(mink_metrics)
    return metrics


def evaluate_baselines(zarr_path, thresholds_path, scaler_val, n_components, quantiles):
    """Executes and orchestrates the metric computations for both baselines."""
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

    # 1. GKF Baseline Execution
    print("\n--- Running Gaussian kinematic formula (GKF) baseline ---")
    expected_gamma_gkf = compute_gkf_expectations(
        train_dataset, physical_thresholds, scaler_val
    )

    # 2. PCR Baseline Execution
    print("\n--- Running principal component regression (PCR) baseline ---")
    ipca, ridge = train_pcr_baseline(train_dataset, n_components=n_components)

    # 3. Test Set Inference and Metric Extraction
    print("\n--- Accumulating test targets and predictions ---")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    y_true_list = []
    x_test_list = []

    for inputs, log_target_gamma, _, _ in test_loader:
        y_true_list.append(log_target_gamma.numpy())
        x_test_list.append(inputs.view(inputs.size(0), -1).numpy())

    y_true_log = np.concatenate(y_true_list, axis=0)
    x_test_flat = np.concatenate(x_test_list, axis=0)

    # Broadcast GKF to match test set dimensions [N, 3, Q]
    y_pred_gkf = np.broadcast_to(expected_gamma_gkf, y_true_log.shape)

    # Transform PCA and predict
    x_pca = ipca.transform(x_test_flat)
    y_pred_pcr_flat = ridge.predict(x_pca)
    y_pred_pcr = y_pred_pcr_flat.reshape(y_true_log.shape)

    # 4. Final Evaluation Printout
    print("\n=============================================")
    print("      Analytical Baseline (GKF) Metrics      ")
    print("=============================================")
    gkf_metrics = evaluate_predictions(y_true_log, y_pred_gkf, quantiles)
    for k, v in gkf_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=============================================")
    print("  Linear Statistical Baseline (PCR) Metrics  ")
    print("=============================================")
    pcr_metrics = evaluate_predictions(y_true_log, y_pred_pcr, quantiles)
    for k, v in pcr_metrics.items():
        print(f"{k}: {v:.4f}")


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
        # Extract scalar using .item() to avoid deprecation warnings
        scaler_val = float(np.load(scaler_path).item())
    else:
        print(f"Scaler file not found at {scaler_path}. Using default value of 5.01.")
        scaler_val = 5.01

    n_components = config.get("PCR_COMPONENTS", 50)
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)

    evaluate_baselines(zarr_path, thresholds_path, scaler_val, n_components, quantiles)


if __name__ == "__main__":
    main()
