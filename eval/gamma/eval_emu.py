import torch
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader

# --- Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import eval.gamma.metrics_lib_emu as metrics_lib
import eval.gamma.plotting_lib_emu as plotting_lib

from src.loss import MinkowskiLoss
from data.dataset import ZarrMixupDataset
from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)


def generate_saliency_samples(model, loader, device, n_examples=10):
    """
    Selects specific examples and computes gradients.
    """
    print("\nGenerating Saliency Samples...")
    model.eval()
    samples = []
    found_dry = False
    found_wet = False
    count = 0

    for input_data, _, _, _ in loader:
        if count >= n_examples:
            break

        input_data = input_data.to(device)
        input_data.requires_grad_(True)

        # Forward
        output = model(input_data)

        # Target: Sum of predicted Area (Index 0)
        target_scalar = output[:, 0, :].sum()

        grad = torch.autograd.grad(target_scalar, input_data, create_graph=False)[0]

        inp_np = input_data.detach().cpu().numpy()
        grad_np = grad.detach().cpu().numpy()

        for b in range(input_data.shape[0]):
            if len(samples) >= n_examples:
                break

            total_rain = inp_np[b].sum()
            img = inp_np[b, 0]
            g = grad_np[b, 0]

            label = ""
            if total_rain < 0.1 and not found_dry:
                label = "Dry Input"
                found_dry = True
            elif total_rain > 1000 and not found_wet:
                label = "Large Storm"
                found_wet = True
            elif len(samples) < n_examples:
                label = f"Sample {len(samples)}"

            if label:
                samples.append((img, g, label))

        model.zero_grad()
        count += 1

    return samples


def compute_jacobian_stats(model, loader, device, n_samples=100):
    """
    Computes Gradient Norms.
    """
    print(f"\nComputing Jacobian Spectrum on subset ({n_samples} samples)...")
    model.eval()
    norms = {"Area": [], "Perimeter": [], "B0": [], "B1": []}
    count = 0

    for input_data, _, _, _ in loader:
        if count >= n_samples:
            break

        input_data = input_data.to(device)
        input_data.requires_grad_(True)

        output = model(input_data)

        # [HANDLE FNO TUPLE]
        if isinstance(output, tuple):
            output = output[0]  # Use Mean for gradient stability check

        n_quantiles = output.shape[2]
        mid_idx = n_quantiles // 2

        # 1. Area Gradient
        target_A = output[:, 0, mid_idx].sum()
        grad_A = torch.autograd.grad(
            target_A, input_data, retain_graph=True, create_graph=False
        )[0]
        norm_A = grad_A.view(grad_A.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        norms["Area"].extend(norm_A)

        # 2. Perimeter Gradient
        target_P = output[:, 1, mid_idx].sum()
        grad_P = torch.autograd.grad(
            target_P, input_data, retain_graph=True, create_graph=False
        )[0]
        norm_P = grad_P.view(grad_P.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        norms["Perimeter"].extend(norm_P)

        # 3. B0 Gradient
        target_B0 = output[:, 2, mid_idx].sum()
        grad_B0 = torch.autograd.grad(
            target_B0, input_data, retain_graph=False, create_graph=False
        )[0]
        norm_B0 = (
            grad_B0.view(grad_B0.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        )
        norms["B0"].extend(norm_B0)

        # 3. B1 Gradient
        target_B1 = output[:, 2, mid_idx].sum()
        grad_B1 = torch.autograd.grad(
            target_B1, input_data, retain_graph=False, create_graph=False
        )[0]
        norm_B1 = (
            grad_B1.view(grad_B1.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        )
        norms["B1"].extend(norm_B1)

        model.zero_grad()
        input_data.grad = None
        count += input_data.size(0)

    return norms


def setup_evaluation(run_dir):
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy"
    )
    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path))
    else:
        scaler_val = 5.01

    return config, device, scaler_val


def load_model_refactored(config, device, run_dir, architecture_type):
    PATCH_SIZE = config["PATCH_SIZE"]
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    N_QUANTILES = len(config["QUANTILE_LEVELS"])
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

    if architecture_type == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Lipschitz":
        model = LipschitzCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")

    model = model.to(device)
    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def get_test_loader(config, scaler_val):
    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    test_dataset = ZarrMixupDataset(
        zarr_path=zarr_path,
        split="test",
        scaler_val=scaler_val,
        augment=False,
        include_original=True,
        include_mixup=False,
    )
    return DataLoader(
        test_dataset,
        batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4),
        pin_memory=True,
    )


def run_prediction_loop(model, loader, criterion, device):
    model.eval()
    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_total_losses = [], []

    # Weights for the evaluation metric integration
    w_a, w_p, w_b0, w_b1 = 1.0, 1.0, 1.0, 1.0

    with torch.no_grad():
        for input_data, log_target_gamma, _, target_phys_tensor in tqdm(
            loader, desc="Inference"
        ):
            input_data = input_data.to(device)
            log_target_gamma = log_target_gamma.to(device)

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            weighted_total_loss, loss_a, loss_p, loss_b0, loss_b1 = criterion(
                predicted_gamma_log, log_target_gamma, w_a, w_p, w_b0, w_b1
            )

            # Revert normalized input to physical scale for visualization
            input_phys = torch.expm1(input_data * loader.dataset.scaler_val)

            all_total_losses.append(weighted_total_loss.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(target_phys_tensor.numpy())
            all_original_images.append(input_phys.squeeze(1).cpu().numpy())

    return (
        np.concatenate(all_preds_phys, axis=0),
        np.concatenate(all_targets_phys, axis=0),
        np.concatenate(all_original_images, axis=0),
        np.concatenate(all_total_losses, axis=0),
    )


def main(run_dir, architecture_type):
    config, device, scaler_val = setup_evaluation(run_dir)
    model = load_model_refactored(config, device, run_dir, architecture_type)
    test_loader = get_test_loader(config, scaler_val)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    criterion = MinkowskiLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    results = run_prediction_loop(model, test_loader, criterion, device)
    all_preds_phys, all_targets_phys, all_original_images, all_total_losses = results

    # The metrics_df and subsequent library calls must be updated to expect 4 channels
    # instead of 4 (A, P, B0, B1).
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_phys, all_targets_phys, all_original_images, all_total_losses
    )

    jacobian_norms = compute_jacobian_stats(model, test_loader, device, n_samples=200)
    # saliency_data = generate_saliency_samples(model, test_loader, device, n_examples=5)

    group_metrics_sample_wise = metrics_lib.calculate_grouped_metrics(metrics_df)
    per_feature_metrics = metrics_lib.calculate_per_feature_metrics(
        all_preds_phys, all_targets_phys, QUANTILE_LEVELS
    )

    # Plotting calls
    plotting_lib.plot_isoperimetric_check(all_preds_phys, run_dir)
    # plotting_lib.plot_dry_input_error(all_preds_phys, all_original_images, run_dir)
    # plotting_lib.plot_saliency_maps(saliency_data, run_dir)
    plotting_lib.plot_jacobian_spectrum(
        jacobian_data=jacobian_norms, output_dir=run_dir
    )
    plotting_lib.plot_per_feature_matrices(
        per_feature_metrics=per_feature_metrics, output_dir=run_dir
    )
    plotting_lib.plot_sample_comparisons(
        metrics_df=metrics_df,
        quantiles=QUANTILE_LEVELS,
        output_dir=run_dir,
        n_samples=15,
    )
    plotting_lib.plot_metric_distributions(metrics_df=metrics_df, output_dir=run_dir)
    plotting_lib.plot_qq_summary(metrics_df=metrics_df, output_dir=run_dir)
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df=metrics_df,
        group_metrics=group_metrics_sample_wise,
        quantiles=QUANTILE_LEVELS,
        output_dir=run_dir,
    )
    log_file_path = os.path.join(run_dir, "training_log.csv")
    plotting_lib.plot_training_log(log_path=log_file_path, output_dir=run_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["Baseline", "Lipschitz", "Constrained"],
    )
    args = parser.parse_args()
    main(args.run_dir, args.arch)
