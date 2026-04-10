import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
import yaml
import sys
from torch.utils.data import DataLoader

# --- Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from data.dataset import ZarrMixupDataset
from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)


# ==========================================
# Loader Helpers (from evaluation script)
# ==========================================


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


# ==========================================
# Plotting Logic
# ==========================================


def plot_comparative_panel(
    sample_best,
    sample_worst,
    predictions_best,
    predictions_worst,
    quantiles,
    best_idx,
    worst_idx,
    output_dir,
    scaler_val,
):
    """
    Generates a 2x5 grid using GridSpec:
    Row 0: Best Sample (Image + A/P/Betti 0/Betti 1 curves)
    Row 1: Worst Sample (Image + A/P/Betti 0/Betti 1 curves)
    Width Ratios: [0.5, 1, 1, 1, 1]
    """
    print(f"\nGenerating comparative plot for indices: {best_idx} and {worst_idx}")

    rows_data = [
        (sample_best, predictions_best, best_idx),
        (sample_worst, predictions_worst, worst_idx),
    ]

    gamma_labels = ["Area (km²)", "Perimeter (km)", "Betti 0 (CCs)", "Betti 1 (Holes)"]

    model_styles = {
        "Target": {"color": "black", "ls": "-", "marker": "o", "lw": 2, "alpha": 0.8},
        "CNN (Unconstr.)": {
            "color": "#648fff",
            "ls": "--",
            "marker": "x",
            "lw": 1,
        },
        "Lip-CNN (Unconstr.)": {
            "color": "#ffb000",
            "ls": "-.",
            "marker": "^",
            "lw": 1,
        },
        "Lip-CNN (Constr.)": {
            "color": "#dc2680",
            "ls": ":",
            "marker": "s",
            "lw": 1,
        },
    }

    fig = plt.figure(figsize=(20, 5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 5, width_ratios=[0.5, 1, 1, 1, 1], figure=fig)

    for row_idx, (sample, preds, sample_idx) in enumerate(rows_data):

        # Unpack sample exactly as in ZarrMixupDataset: (input, log_target, input, target_phys)
        input_tensor = sample[0]
        target_phys_tensor = sample[3]

        # Revert normalized input to physical scale
        precip_img = np.expm1(input_tensor.squeeze().cpu().numpy() * scaler_val)
        target_gamma = target_phys_tensor.cpu().numpy()

        # --- Col 0: Precipitation Image ---
        ax_img = fig.add_subplot(gs[row_idx, 0])

        plot_data = precip_img.copy()
        plot_data[plot_data <= 0] = np.nan

        cmap = copy.copy(plt.get_cmap("Blues"))
        cmap.set_bad(color="lightgrey", alpha=1.0)

        im = ax_img.imshow(plot_data, cmap=cmap, origin="lower")
        fig.colorbar(im, ax=ax_img, shrink=0.8, label="Precip. (mm/hr)")

        # --- Cols 1-4: Gamma Components ---
        for col_idx in range(4):
            ax_gamma = fig.add_subplot(gs[row_idx, col_idx + 1])
            component_idx = col_idx

            ax_gamma.plot(
                quantiles,
                target_gamma[component_idx],
                label="Target",
                **model_styles["Target"],
            )

            for model_name, model_pred in preds.items():
                pred_curve = model_pred[component_idx]
                ax_gamma.plot(
                    quantiles, pred_curve, label=model_name, **model_styles[model_name]
                )

            ax_gamma.grid(True, linestyle="--", alpha=0.5)

            if row_idx == 0:
                ax_gamma.set_title(gamma_labels[component_idx], fontsize=14)
            if row_idx == 1:
                ax_gamma.set_xlabel("Precip. Threshold (mm/hr)", fontsize=12)
            else:
                ax_gamma.tick_params(labelbottom=False)

            if row_idx == 0 and col_idx == 0:
                ax_gamma.legend()

    filename = f"comparison_best_{best_idx}_worst_{worst_idx}.pdf"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


# ==========================================
# Helpers
# ==========================================


def get_prediction(model, input_tensor):
    """Run inference for a single sample."""
    model.eval()
    with torch.no_grad():
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)
        if isinstance(output, tuple):
            output = output[0]
    return output.squeeze(0).cpu().numpy()


# ==========================================
# Main Execution
# ==========================================


def main(args):
    print("--- Setting up environment based on Baseline run ---")
    config, device, scaler_val = setup_evaluation(args.dir_baseline)

    test_loader = get_test_loader(config, scaler_val)
    test_dataset = test_loader.dataset
    quantiles = config["QUANTILE_LEVELS"]

    print("\n--- Loading Models ---")
    models = {}

    models["CNN (Unconstr.)"] = load_model_refactored(
        config, device, args.dir_baseline, "Baseline"
    )
    models["Lip-CNN (Unconstr.)"] = load_model_refactored(
        config, device, args.dir_lipschitz, "Lipschitz"
    )
    models["Lip-CNN (Constr.)"] = load_model_refactored(
        config, device, args.dir_constrained, "Constrained"
    )

    print(f"\n--- Retrieving Samples {args.best_idx} and {args.worst_idx} ---")

    if args.best_idx >= len(test_dataset) or args.worst_idx >= len(test_dataset):
        raise IndexError(f"Indices must be < {len(test_dataset)}")

    sample_best = test_dataset[args.best_idx]
    sample_worst = test_dataset[args.worst_idx]

    input_best = sample_best[0].to(device)
    input_worst = sample_worst[0].to(device)

    preds_best = {}
    preds_worst = {}

    for name, model in models.items():
        preds_best[name] = get_prediction(model, input_best)
        preds_worst[name] = get_prediction(model, input_worst)

    output_dir = "/home/fquareng/work/figures/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_comparative_panel(
        sample_best,
        sample_worst,
        preds_best,
        preds_worst,
        quantiles,
        args.best_idx,
        args.worst_idx,
        output_dir,
        scaler_val,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Baseline, Lipschitz, and Constrained emulators."
    )
    parser.add_argument(
        "--dir_baseline",
        type=str,
        required=True,
        help="Run directory for Baseline model",
    )
    parser.add_argument(
        "--dir_lipschitz",
        type=str,
        required=True,
        help="Run directory for Lipschitz model",
    )
    parser.add_argument(
        "--dir_constrained",
        type=str,
        required=True,
        help="Run directory for Constrained model",
    )
    parser.add_argument(
        "--best_idx", type=int, required=True, help="Index of the 'Best' sample"
    )
    parser.add_argument(
        "--worst_idx", type=int, required=True, help="Index of the 'Worst' sample"
    )

    args = parser.parse_args()
    main(args)
