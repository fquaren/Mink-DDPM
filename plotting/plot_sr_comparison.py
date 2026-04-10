import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import sys
import argparse
import json
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- project imports ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

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


def mask_low_values(img, threshold=0.1):
    masked = img.copy()
    masked[masked <= threshold] = np.nan
    return masked


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


def generate_comparison_plot(args):
    # enforce strict determinism
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

    # instantiate both datasets to preserve distinct input scaling mechanics
    unet_dataset = DeterministicSRDataset(
        config["PREPROCESSED_DATA_DIR"],
        metadata_file,
        config["DEM_DATA_DIR"],
        dem_stats,
        scaler_max_val=max_val,
        split="test",
        load_in_ram=False,
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

    # retrieve specific index
    x_u, y_u, _ = unet_dataset[args.index]
    x_d, y_d, _ = ddpm_dataset[args.index]

    x_u = x_u.unsqueeze(0).to(device)
    x_d = x_d.unsqueeze(0).to(device)
    y_target = y_d.unsqueeze(0).to(device)

    # 1. u-net inference
    unet_models = [load_unet(ckpt, device) for ckpt in args.unet_ckpts]
    unet_preds_phys = []
    with torch.no_grad():
        for model in unet_models:
            pred = model(x_u)
            pred_phys = denormalizer.unnormalize_torch(pred[:, 0:1])
            unet_preds_phys.append(pred_phys.cpu().numpy()[0, 0])

    # 2. ddpm inference
    ddpm_model, diffusion = load_ddpm(args.ddpm_ckpt, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in str(device) else "cpu"):
            pred_ddpm = diffusion.sample_ddim(
                ddpm_model, n=1, conditions=x_d, ddim_steps=50
            )

        # shift diffusion latent space [-1, 1] -> [0, 1] log space before physical conversion
        pred_ddpm_shifted = (pred_ddpm[:, 0:1].clamp(-1.0, 1.0) + 1.0) / 2.0
        ddpm_phys = (
            denormalizer.unnormalize_torch(pred_ddpm_shifted).cpu().numpy()[0, 0]
        )

    # 3. reference field processing
    with torch.no_grad():
        # original low resolution via ddpm scaling shift
        x_shifted = (x_d[:, 0:1] + 1.0) / 2.0
        lr_phys = denormalizer.unnormalize_torch(x_shifted)

        # bilinear interpolation
        interp_phys = F.interpolate(
            lr_phys, size=y_target.shape[-2:], mode="bicubic", align_corners=False
        )

        # target hr
        y_shifted = (y_target[:, 0:1] + 1.0) / 2.0
        target_phys = denormalizer.unnormalize_torch(y_shifted)

    lr_phys_np = lr_phys.cpu().numpy()[0, 0]
    interp_phys_np = interp_phys.cpu().numpy()[0, 0]
    target_phys_np = target_phys.cpu().numpy()[0, 0]

    # 4. dem un-standardization
    dem_patch = x_u[0, 1].cpu().numpy()
    dem_patch = (dem_patch * dem_stats[1]) + dem_stats[0]

    # 5. topology definitions
    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    all_precip = [
        lr_phys_np,
        interp_phys_np,
        target_phys_np,
        ddpm_phys,
    ] + unet_preds_phys
    vmax = max(np.nanmax(img) for img in all_precip)
    vmax = max(vmax, 1.0)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # set text size globally before figure initialization
    plt.rcParams.update({"font.size": 21})

    # strict geometry initialization replacing tight_layout
    fig = plt.figure(figsize=(20, 10), layout="constrained")

    # 6 columns: 5 for data, 1 narrow column strictly for the colorbar
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.05])

    axs = np.empty((2, 4), dtype=object)
    for i in range(2):
        for j in range(4):
            axs[i, j] = fig.add_subplot(gs[i, j])
            axs[i, j].axis(
                "off"
            )  # remove the axis dynamically during grid initialization

    # row 1 mappings
    im_dem = axs[0, 0].imshow(dem_patch, cmap="terrain", origin="lower")
    axs[0, 0].set_title("DEM")
    fig.colorbar(
        im_dem,
        ax=axs[0, 0],
        fraction=0.046,
        pad=0.04,
        label="Elevation (m)",
        location="left",
    )

    axs[0, 1].imshow(
        mask_low_values(lr_phys_np), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[0, 1].set_title("Original (low res)")

    axs[0, 2].imshow(
        mask_low_values(interp_phys_np), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[0, 2].set_title("Bicubic interp.")

    axs[0, 3].imshow(
        mask_low_values(unet_preds_phys[0]), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[0, 3].set_title("UNet baseline")

    # row 2 mappings
    axs[1, 0].imshow(
        mask_low_values(unet_preds_phys[1]), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[1, 0].set_title("UNet Mink. analytical")

    axs[1, 1].imshow(
        mask_low_values(unet_preds_phys[3]), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[1, 1].set_title("UNet Mink. Lip-CNN")

    axs[1, 2].imshow(
        mask_low_values(ddpm_phys), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[1, 2].set_title("DDIM baseline")

    im_precip = axs[1, 3].imshow(
        mask_low_values(target_phys_np), cmap=precip_cmap, norm=norm, origin="lower"
    )
    axs[1, 3].set_title("Target (high res)")

    # explicit colorbar allocation spanning both rows in column index 4
    cbar_ax = fig.add_subplot(gs[:, 4])
    fig.colorbar(im_precip, cax=cbar_ax, label="Precipitation (mm/hr)")

    out_name = f"/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/comparison_patch_{args.index}.pdf"
    plt.savefig(out_name, bbox_inches="tight", dpi=150)
    print(f"saved figure to {out_name}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate cross-architecture multi-patch visualization"
    )
    parser.add_argument(
        "--unet_ckpts",
        nargs=4,
        required=True,
        help="paths to 4 independent u-net checkpoints",
    )
    parser.add_argument(
        "--ddpm_ckpt", type=str, required=True, help="path to ddpm checkpoint"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="target patch index within the test dataset",
    )
    args = parser.parse_args()

    generate_comparison_plot(args)
