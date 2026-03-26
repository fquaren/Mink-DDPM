import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import yaml
import numpy as np


# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

import models.emulators.gamma_predictors as gamma_predictors
from data.dataset import ZarrMixupDataset


PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(config["QUANTILE_LEVELS"])
INPUT_SHAPE = (1, config["PATCH_SIZE"], config["PATCH_SIZE"])
BATCH_SIZE = config.get("BATCH_SIZE", 16)
NUM_WORKERS = config.get("NUM_WORKERS", 4)


class LatentEvaluator(nn.Module):
    """Wraps a model to return its latent representation using hooks."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.latent = None
        # Intercept the input arguments to head_CC
        self.handle = self.model.head_CC.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.latent = args[0]

    def forward(self, x):
        _ = self.model(x)  # Trigger the forward pass
        # The hook populates self.latent during the forward pass
        return self.latent

    def __del__(self):
        if hasattr(self, "handle"):
            self.handle.remove()


def load_emulator(model, checkpoint_path, device):
    """
    Loads weights into a pre-instantiated emulator model.
    """
    print(f"--- Loading weights from: {checkpoint_path} ---")

    model = model.to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Clean state_dict if it was saved with DataParallel ('module.' prefix)
        new_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }

        model.load_state_dict(new_state_dict)

    except Exception as e:
        print(f"\nCRITICAL ERROR loading emulator weights from {checkpoint_path}: {e}")
        raise e

    # Freeze and Eval
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Emulator successfully loaded, set to eval mode, and weights frozen.")
    return model


def empirical_lipschitz_batched(model, x, epsilon=1e-2, steps=10, init_noise=1e-4):
    """
    Corrected empirical Lipschitz estimation with initial symmetry breaking.
    """
    model.eval()
    x_orig = x.clone().detach()

    # Break symmetry: initialize x_pert slightly away from x_orig
    # Noise is uniform in [-init_noise, init_noise]
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


def run_model_comparison(device="cuda"):
    # Initialize dataset
    zarr_path = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_dataset.zarr")
    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")
    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path))
    else:
        print(f"Scaler file not found at {scaler_path}. Using default value of 5.01.")
        scaler_val = 5.01  # Default value based on training data max log(precip + 1)
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
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize models
    models = {
        "Baseline": gamma_predictors.BaselineCNN(
            n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE
        ).to(device),
        "Lipschitz": gamma_predictors.LipschitzCNN(
            n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE
        ).to(device),
        "Constrained": gamma_predictors.ConstrainedLipschitzCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
        ).to(device),
    }

    checkpoints = {
        "Baseline": "/home/fquareng/work/ch2/final_experiment_runs/GammaEmulator_v6_Baseline_T1_2026-03-13_08-20-20/best_model_checkpoint.pth",
        "Lipschitz": "/home/fquareng/work/ch2/final_experiment_runs/GammaEmulator_v6_Lipschitz_T1_2026-03-13_06-56-28/best_model_checkpoint.pth",
        "Constrained": "/home/fquareng/work/ch2/final_experiment_runs/GammaEmulator_v6_Constrained_T1_2026-03-13_08-22-37/best_model_checkpoint.pth",
    }

    results = {name: [] for name in models}

    for name in models.keys():
        if name not in checkpoints:
            raise KeyError(f"Checkpoint path missing for model: {name}")

        # Enforce strict keyword arguments for safety
        models[name] = load_emulator(
            model=models[name], checkpoint_path=checkpoints[name], device=device
        )

        # Wrap the initialized and loaded model to extract the latent representation
        models[name] = LatentEvaluator(models[name])

    for batch in tqdm(dataloader, desc="Evaluating Lipschitz constants"):
        x = batch[0].to(device)
        for name, model in models.items():
            batch_constants = empirical_lipschitz_batched(model, x)
            results[name].append(batch_constants.max().item())

    for name, vals in results.items():
        print(f"Model: {name} | Max Empirical Lipschitz (Latent): {max(vals):.4f}")


if __name__ == "__main__":
    run_model_comparison()
