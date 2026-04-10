import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import sys
import argparse
import json
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.ndimage import label, center_of_mass

# --- Project Imports ---
parent_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

print(f"Project root added to sys.path: {parent_path}")

from models.SR.deterministic.unet import LogSpaceResidualUNet
from data.dataset import DeterministicSRDataset
from src.loss import AnalyticalMinkowskiLoss

# --- Metric Implementations ---


class DataDenormalizer:
    def __init__(self, max_val):
        self.max_val = float(max_val)

    def unnormalize(self, x_norm):
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.detach().cpu().numpy()
        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        return np.maximum(x_phys, 0.0)

    def unnormalize_torch(self, x_norm):
        max_val_tensor = torch.tensor(
            self.max_val, device=x_norm.device, dtype=x_norm.dtype
        )
        x_scaled = x_norm * max_val_tensor
        x_scaled = torch.clamp(x_scaled, max=50.0)
        x_phys = torch.expm1(x_scaled)
        return F.relu(x_phys)


def compute_fss(pred, target, threshold, window_size):
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    pad = window_size // 2
    pool = nn.AvgPool2d(
        kernel_size=window_size, stride=1, padding=pad, count_include_pad=False
    )

    pred_frac = pool(pred_bin)
    target_frac = pool(target_bin)

    mse = ((pred_frac - target_frac) ** 2).mean(dim=(1, 2, 3))
    ref = (pred_frac**2 + target_frac**2).mean(dim=(1, 2, 3))

    fss = 1.0 - (mse / (ref + 1e-8))
    return fss.mean().item()


def compute_wasserstein(pred_phys, target_phys, drizzle_threshold=0.1):
    p_flat = pred_phys[pred_phys > drizzle_threshold].flatten()
    t_flat = target_phys[target_phys > drizzle_threshold].flatten()
    if len(p_flat) == 0 or len(t_flat) == 0:
        return 0.0
    return wasserstein_distance(t_flat, p_flat)


def compute_spectral_loss(pred, target):
    pred_fft = torch.fft.fft2(pred.float())
    target_fft = torch.fft.fft2(target.float())

    pred_mag = torch.log(torch.abs(torch.fft.fftshift(pred_fft)) + 1e-8)
    target_mag = torch.log(torch.abs(torch.fft.fftshift(target_fft)) + 1e-8)
    return F.l1_loss(pred_mag, target_mag).item()


def _get_sal_features(field, thr=0.1):
    R = np.mean(field)
    if R <= 1e-8:
        return 0.0, np.array([0.0, 0.0]), 0.0

    mask = field >= thr
    labeled_array, num_obj = label(mask)
    if num_obj == 0:
        return R, np.array(field.shape) / 2.0, 0.0

    V_sum = 0.0
    R_sum = np.sum(field[mask])

    for i in range(1, num_obj + 1):
        obj_mask = labeled_array == i
        R_n = np.sum(field[obj_mask])
        R_n_max = np.max(field[obj_mask])
        V_n = R_n / (R_n_max + 1e-8)
        V_sum += V_n * R_n

    V = V_sum / (R_sum + 1e-8)
    com = np.array(center_of_mass(field))
    return R, com, V


def compute_sal(pred, target, thr=0.1):
    R_p, com_p, V_p = _get_sal_features(pred, thr)
    R_t, com_t, V_t = _get_sal_features(target, thr)

    den_A = 0.5 * (R_p + R_t)
    A = (R_p - R_t) / den_A if den_A > 1e-8 else 0.0

    den_S = 0.5 * (V_p + V_t)
    S = (V_p - V_t) / den_S if den_S > 1e-8 else 0.0

    d = np.hypot(pred.shape[0], pred.shape[1])
    L = np.linalg.norm(com_p - com_t) / d if den_A > 1e-8 else 0.0

    return S, A, L


def compute_raps(image):
    image = np.squeeze(image)
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = np.round(knrm).astype(int)
    bins = np.bincount(knrm.ravel(), fourier_amplitudes.ravel())
    return bins[: npix // 2]


def compute_raps_error(pred, target):
    raps_p = compute_raps(pred)
    raps_t = compute_raps(target)
    return np.mean(np.abs(np.log10(raps_p + 1e-8) - np.log10(raps_t + 1e-8)))


# --- Main Evaluation Loop ---


def evaluate(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with open(os.path.join(parent_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)

    with open(config["DEM_STATS"], "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    max_val_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy"
    )
    max_val = float(np.load(max_val_path).item())
    denormalizer = DataDenormalizer(max_val)

    test_dataset = DeterministicSRDataset(
        config["PREPROCESSED_DATA_DIR"],
        config.get("TEST_METADATA_FILE", config["VAL_METADATA_FILE"]),
        config["DEM_DATA_DIR"],
        dem_stats,
        scaler_max_val=max_val,
        split="test",
        load_in_ram=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
    )

    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion_mae = nn.L1Loss()
    criterion_minkowski = AnalyticalMinkowskiLoss(
        thresholds=config["QUANTILE_LEVELS"]
    ).to(device)

    sal_thresholds = [0.1, 1.0, 5.0]

    metrics = {
        "mae_phys": [],
        "minkowski": [],
        "wasserstein": [],
        "spectral_mae": [],
        "sal_S": {t: [] for t in sal_thresholds},
        "sal_A": {t: [] for t in sal_thresholds},
        "sal_L": {t: [] for t in sal_thresholds},
        "raps_mae": [],
        "fss": {
            (1.0, 3): [],
            (1.0, 9): [],
            (1.0, 27): [],
            (5.0, 3): [],
            (5.0, 9): [],
            (5.0, 27): [],
            (10.0, 3): [],
            (10.0, 9): [],
            (10.0, 27): [],
        },
    }

    print(f"Starting evaluation on {len(test_dataset)} samples using device: {device}")

    with torch.no_grad():
        for X, Y_norm, Y_gamma_log in tqdm(test_loader, desc="Evaluating"):
            X, Y_norm, Y_gamma_log = (
                X.to(device),
                Y_norm.to(device),
                Y_gamma_log.to(device),
            )
            Y_pred_norm = model(X)

            pred_phys = denormalizer.unnormalize_torch(Y_pred_norm[:, 0:1])
            target_phys = denormalizer.unnormalize_torch(Y_norm[:, 0:1])

            metrics["mae_phys"].append(criterion_mae(pred_phys, target_phys).item())
            metrics["minkowski"].append(
                criterion_minkowski(pred_phys, Y_gamma_log).item()
            )
            metrics["spectral_mae"].append(
                compute_spectral_loss(pred_phys, target_phys)
            )

            for thresh, window in metrics["fss"].keys():
                fss_val = compute_fss(
                    pred_phys, target_phys, threshold=thresh, window_size=window
                )
                metrics["fss"][(thresh, window)].append(fss_val)

            p_np = pred_phys.cpu().numpy()
            t_np = target_phys.cpu().numpy()
            for b in range(p_np.shape[0]):
                metrics["wasserstein"].append(compute_wasserstein(p_np[b], t_np[b]))
                metrics["raps_mae"].append(compute_raps_error(p_np[b, 0], t_np[b, 0]))

                for thr in sal_thresholds:
                    S, A, L = compute_sal(p_np[b, 0], t_np[b, 0], thr)
                    metrics["sal_S"][thr].append(S)
                    metrics["sal_A"][thr].append(A)
                    metrics["sal_L"][thr].append(L)

    results = {
        "MAE (Physical)": np.mean(metrics["mae_phys"]),
        "Minkowski Loss": np.mean(metrics["minkowski"]),
        "Wasserstein Dist": np.mean(metrics["wasserstein"]),
        "Spectral Log-Mag MAE": np.mean(metrics["spectral_mae"]),
        "RAPS Log10-MAE": np.mean(metrics["raps_mae"]),
    }

    for thr in sal_thresholds:
        results[f"SAL Structure (S) > {thr}mm"] = np.mean(metrics["sal_S"][thr])
        results[f"SAL Amplitude (A) > {thr}mm"] = np.mean(metrics["sal_A"][thr])
        results[f"SAL Location (L) > {thr}mm"] = np.mean(metrics["sal_L"][thr])

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n--- Fractions Skill Score (FSS) ---")
    for (thresh, window), vals in metrics["fss"].items():
        print(
            f"Threshold: {thresh:04.1f}mm | Window: {window:02d}x{window:02d} -> FSS: {np.mean(vals):.4f}"
        )

    if args.save_json:
        fss_str_keys = {
            f"thresh_{t}_win_{w}": np.mean(vals)
            for (t, w), vals in metrics["fss"].items()
        }
        results["FSS"] = fss_str_keys
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SR UNet model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="eval_results.json",
        help="Path to save output metrics",
    )
    args = parser.parse_args()
    evaluate(args)
