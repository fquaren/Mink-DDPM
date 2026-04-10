import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path Setup ---
parent_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
print(f"Adding {parent_path} to sys.path for module imports.")
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from data.dataset import ZarrMixupDataset
from eval.SR.metrics_lib_sr import compute_isoperimetric_violation


def compute_analytical_approximation(
    field_phys,
    thresholds,
    pixel_size_km=2.0,
    init_factor=1e-5,
    min_temp=1e-6,
    persistence_thresh=1.8699999839067458,
):
    device = field_phys.device
    thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)
    base_temps_tensor = torch.tensor(
        np.maximum(np.array(thresholds) * init_factor, min_temp),
        dtype=torch.float32,
        device=device,
    )

    pixel_area = pixel_size_km**2
    areas, perimeters, eulers = [], [], []

    # --- TOPOLOGY PRE-PROCESSING ---
    # 1. Morphological Closing (Fills small false holes / Betti-1)
    dilated = F.max_pool2d(field_phys, kernel_size=3, stride=1, padding=1)
    closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

    # 2. Morphological Opening (Removes small false spikes)
    eroded = -F.max_pool2d(-closed, kernel_size=3, stride=1, padding=1)
    field_phys_topo = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

    # 3. Local Maximum for Persistence
    local_max = F.max_pool2d(field_phys_topo, kernel_size=15, stride=1, padding=7)

    for q_idx, thresh in enumerate(thresholds_tensor):
        current_temp = base_temps_tensor[q_idx]

        # --- PATH 1: EXACT GEOMETRY (Area & Perimeter) ---
        p_raw = torch.sigmoid((field_phys - thresh) / current_temp)

        area = torch.sum(p_raw, dim=(1, 2, 3)) * pixel_area

        p_pad = F.pad(p_raw, (0, 1, 0, 1), mode="replicate")
        # Symmetric central differences (x[i+1] - x[i-1]) / 2
        dx = (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) / 2.0
        dy = (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]) / 2.0

        perimeter = (
            torch.sum(
                torch.sqrt(dx**2 + dy**2 + 1e-8),
                dim=(2, 3),
            )
            * pixel_size_km
        )

        # --- PATH 2: AMPLITUDE-AWARE TOPOLOGY (Euler) ---
        p_base = torch.sigmoid((field_phys_topo - thresh) / current_temp)

        # Binary mask: 1 if neighborhood peak > thresh + persistence, else 0
        persistence_mask = torch.sigmoid(
            (local_max - (thresh + persistence_thresh)) / current_temp
        )

        # Apply mask using Gödel T-norm (min) to prevent fractional distortion
        p_topo = torch.min(p_base, persistence_mask)

        # Gödel T-norm Expected Euler Characteristic
        V = torch.sum(p_topo, dim=(1, 2, 3))
        E_x = torch.sum(
            torch.min(p_topo[:, :, :, :-1], p_topo[:, :, :, 1:]), dim=(1, 2, 3)
        )
        E_y = torch.sum(
            torch.min(p_topo[:, :, :-1, :], p_topo[:, :, 1:, :]), dim=(1, 2, 3)
        )
        F_faces = torch.sum(
            torch.min(
                torch.min(p_topo[:, :, :-1, :-1], p_topo[:, :, :-1, 1:]),
                torch.min(p_topo[:, :, 1:, :-1], p_topo[:, :, 1:, 1:]),
            ),
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

    return torch.sign(pred_gamma_phys) * torch.log1p(torch.abs(pred_gamma_phys))


def train_pcr_baseline(train_dataset, n_components=50):
    """
    Fits Incremental PCA and Ridge Regression.
    """
    loader = DataLoader(train_dataset, batch_size=200, shuffle=False, num_workers=4)

    ipca = IncrementalPCA(n_components=n_components)
    print("Fitting Incremental PCA...")
    for inputs, _, _, _ in tqdm(loader, desc="Fitting Incremental PCA"):
        X_batch = inputs.view(inputs.size(0), -1).numpy()
        ipca.partial_fit(X_batch)

    X_pca_list = []
    y_list = []

    print("Transforming data and accumulating targets...")
    for inputs, log_target_gamma, _, _ in tqdm(loader, desc="Transforming Data"):
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
    # train_dataset = ZarrMixupDataset(
    #     zarr_path,
    #     split="train",
    #     scaler_val=scaler_val,
    #     include_original=True,
    #     include_mixup=False,
    #     augment=False,
    #     subset_fraction=0.01,
    # )
    test_dataset = ZarrMixupDataset(
        zarr_path,
        split="test",
        scaler_val=scaler_val,
        include_original=True,
        include_mixup=False,
        augment=False,
        subset_fraction=0.01,
    )

    physical_thresholds = np.load(thresholds_path)

    # # 1. PCR Baseline Execution (4 Channels)
    # print("\n--- Running principal component regression (PCR) baseline ---")
    # ipca, ridge = train_pcr_baseline(train_dataset, n_components=n_components)

    # 2. Test Set Inference and Target Construction
    print("\n--- Accumulating test targets and predictions ---")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    y_true_4ch_list = []
    x_test_list = []
    y_pred_analytical_list = []

    for inputs, log_target_gamma, _, _ in tqdm(test_loader):
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
    # x_test_flat = np.concatenate(x_test_list, axis=0)
    y_pred_analytical = np.concatenate(y_pred_analytical_list, axis=0)

    # Compute true Euler characteristic in physical space and re-transform
    y_true_raw = np.sign(y_true_log_4ch) * np.expm1(np.abs(y_true_log_4ch))
    true_euler_raw = y_true_raw[:, 2, :] - y_true_raw[:, 3, :]
    true_euler_log = np.sign(true_euler_raw) * np.log1p(np.abs(true_euler_raw))

    y_true_log_3ch = np.stack(
        [y_true_log_4ch[:, 0, :], y_true_log_4ch[:, 1, :], true_euler_log], axis=1
    )

    # # Transform PCA and predict
    # x_pca = ipca.transform(x_test_flat)
    # y_pred_pcr_flat = ridge.predict(x_pca)
    # y_pred_pcr = y_pred_pcr_flat.reshape(y_true_log_4ch.shape)

    # # 4. Final Evaluation Printout
    # print("\n=============================================")
    # print("  Linear Statistical Baseline (PCR) Metrics  ")
    # print("=============================================")
    # pcr_metrics = evaluate_predictions(
    #     y_true_log_4ch, y_pred_pcr, quantiles, ["Area", "Perimeter", "Betti0", "Betti1"]
    # )
    # for k, v in pcr_metrics.items():
    #     print(f"{k}: {v:.4f}")

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
        # "PCR": pcr_metrics,
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
