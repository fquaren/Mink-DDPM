import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project Imports ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
print(f"Project root added to sys.path: {parent_path}")


from models.SR.deterministic.unet import LogSpaceResidualUNet
from models.SR.ddpm.ddpm import ContextUnet
from models.SR.ddpm.diffusion import Diffusion
from data.dataset import DeterministicSRDataset, DiffusionSRDataset


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


def load_unet(checkpoint_path, device):
    model = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_ddpm(checkpoint_path, device):
    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    diffusion = Diffusion(img_size=128, device=device)
    return model, diffusion


def compute_raps(image_batch):
    """
    Computes Radially Averaged Power Spectrum for a batch of images.
    Returns: Binned amplitudes (power) for each spatial frequency.
    """
    if image_batch.ndim == 4:
        image_batch = image_batch.squeeze(1)

    B, H, W = image_batch.shape
    assert H == W, "Image must be square for strict RAPS computation."
    npix = H

    fourier_image = np.fft.fftn(image_batch, axes=(-2, -1))
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = np.round(knrm).astype(int)

    max_bin = npix // 2
    binned_amplitudes = np.zeros((B, max_bin))

    for b in range(B):
        bins = np.bincount(knrm.ravel(), fourier_amplitudes[b].ravel())
        pad_len = max_bin - len(bins)
        if pad_len > 0:
            bins = np.pad(bins, (0, pad_len), "constant")
        binned_amplitudes[b] = bins[:max_bin]

    return binned_amplitudes


def generate_spectra_plot(args):
    torch.manual_seed(42)
    np.random.seed(42)

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

    metadata_file = config.get("TEST_METADATA_FILE", config["VAL_METADATA_FILE"])

    print("Initializing Datasets and Dataloaders...")

    test_dataset = DeterministicSRDataset(
        config["PREPROCESSED_DATA_DIR"],
        metadata_file,
        config["DEM_DATA_DIR"],
        dem_stats,
        scaler_max_val=max_val,
        split="test",
        load_in_ram=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4),
    )

    ddpm_dataset = DiffusionSRDataset(
        config["PREPROCESSED_DATA_DIR"],
        metadata_file,
        config["DEM_DATA_DIR"],
        dem_stats,
        scaler_max_val=max_val,
        split="test",
        data_percentage=100.0,
    )
    ddpm_loader = DataLoader(
        ddpm_dataset,
        batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 4),
    )

    print("Loading Models...")
    unet_baseline = load_unet(args.unet_baseline, device)
    unet_analytical = load_unet(args.unet_analytical, device)
    unet_lipcnn = load_unet(args.unet_lipcnn, device)
    ddpm_model, diffusion = load_ddpm(args.ddpm_ckpt, device)

    spectra = {
        "Target": [],
        "Bicubic": [],
        "UNet Baseline": [],
        "UNet Analytical": [],
        "UNet Lip-CNN": [],
        "DDIM": [],
    }

    print("Evaluating Deterministic Models...")
    with torch.no_grad():
        for i, (X, Y_norm, _) in enumerate(
            tqdm(test_loader, desc="UNet & Interpolation")
        ):
            if args.max_batches and i >= args.max_batches:
                break

            X = X.to(device)
            Y_norm = Y_norm.to(device)

            # Target
            y_phys = denormalizer.unnormalize_torch(Y_norm[:, 0:1]).cpu().numpy()
            spectra["Target"].append(compute_raps(y_phys))

            # Bicubic
            target_size = (Y_norm.size(2), Y_norm.size(3))
            interp_norm = F.interpolate(
                X[:, 0:1, :, :], size=target_size, mode="bicubic", align_corners=False
            )
            interp_phys = denormalizer.unnormalize_torch(interp_norm).cpu().numpy()
            spectra["Bicubic"].append(compute_raps(interp_phys))

            # UNet Baseline
            pred_base = unet_baseline(X)
            phys_base = denormalizer.unnormalize_torch(pred_base[:, 0:1]).cpu().numpy()
            spectra["UNet Baseline"].append(compute_raps(phys_base))

            # UNet Analytical
            pred_ana = unet_analytical(X)
            phys_ana = denormalizer.unnormalize_torch(pred_ana[:, 0:1]).cpu().numpy()
            spectra["UNet Analytical"].append(compute_raps(phys_ana))

            # UNet Lip-CNN
            pred_lip = unet_lipcnn(X)
            phys_lip = denormalizer.unnormalize_torch(pred_lip[:, 0:1]).cpu().numpy()
            spectra["UNet Lip-CNN"].append(compute_raps(phys_lip))

    print("Evaluating Diffusion Model...")
    with torch.no_grad():
        for i, (X, _, _, *_) in enumerate(tqdm(ddpm_loader, desc="DDIM")):
            if args.max_batches and i >= args.max_batches:
                break

            X = X.to(device)
            with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu"):
                pred_ddpm = diffusion.sample_ddim(
                    ddpm_model, n=X.shape[0], conditions=X, ddim_steps=args.ddim_steps
                )

            pred_ddpm_shifted = (pred_ddpm[:, 0:1].clamp(-1.0, 1.0) + 1.0) / 2.0
            ddpm_phys = denormalizer.unnormalize_torch(pred_ddpm_shifted).cpu().numpy()
            spectra["DDIM"].append(compute_raps(ddpm_phys))

    print("Aggregating Spectra and Plotting...")
    for key in spectra.keys():
        if len(spectra[key]) > 0:
            spectra[key] = np.mean(np.concatenate(spectra[key], axis=0), axis=0)

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(10, 8))

    wavenumbers = np.arange(1, spectra["Target"].shape[0] + 1)

    colors = {
        "Target": "black",
        "Bicubic": "grey",
        "UNet Baseline": "red",
        "UNet Analytical": "blue",
        "UNet Lip-CNN": "green",
        "DDIM": "purple",
    }

    linestyles = {
        "Target": "-",
        "Bicubic": ":",
        "UNet Baseline": "--",
        "UNet Analytical": "-",
        "UNet Lip-CNN": "-.",
        "DDIM": "-",
    }

    for key, spec in spectra.items():
        if len(spec) > 0:
            ax.loglog(
                wavenumbers,
                spec,
                label=key,
                color=colors[key],
                linestyle=linestyles[key],
                linewidth=2.5,
            )

    ax.set_xlabel("Wavenumber (km$^{-1}$)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Radially Averaged Power Spectral Density (Test Set)")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(loc="lower left")

    out_dir = os.path.join(parent_path, "ci26_revision_runs", "sr_experiment_runs")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, "mean_spectra_comparison.pdf")

    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"Saved spectra figure to {out_name}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate full dataset Radially Averaged Power Spectral Density (RAPS) plots."
    )
    parser.add_argument(
        "--unet_baseline",
        type=str,
        required=True,
        help="Path to UNet Baseline checkpoint",
    )
    parser.add_argument(
        "--unet_analytical",
        type=str,
        required=True,
        help="Path to UNet Analytical checkpoint",
    )
    parser.add_argument(
        "--unet_lipcnn", type=str, required=True, help="Path to UNet Lip-CNN checkpoint"
    )
    parser.add_argument(
        "--ddpm_ckpt", type=str, required=True, help="Path to DDPM checkpoint"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="Number of sampling steps for DDIM"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit number of batches evaluated (useful for debugging)",
    )
    args = parser.parse_args()

    generate_spectra_plot(args)
