import yaml
import torch
import numpy as np
import os


from models.emulators.gamma_predictors import (
    BaselineCNN,
    LipschitzCNN,
    ConstrainedLipschitzCNN,
)

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Constants ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
PREPROCESSED_DATA_DIR = config.get("PREPROCESSED_DATA_DIR", "./data")


def load_emulator(checkpoint_path, config, device):
    """
    Loads a trained Gamma Emulator model (Baseline, Lipschitz, or Constrained).

    Args:
        checkpoint_path (str): Path to the .pth file.
        config (dict): Configuration dictionary.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        nn.Module: The loaded, frozen emulator in eval mode.
    """
    print(f"--- Loading Gamma Emulator from: {checkpoint_path} ---")

    # 1. Load Physical Scaler (Critical for Normalization)
    # The new models normalize inputs internally using this value.
    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "log_precip_max_val.npy")

    if os.path.exists(scaler_path):
        max_input_val = float(np.load(scaler_path))
        print(f"Loaded max_input_val for emulator normalization: {max_input_val:.4f}")
    else:
        print("Warning: Scaler file not found. Defaulting to 5.5")
        max_input_val = 5.5

    # 2. Extract Config Parameters
    arch = config.get("ARCHITECTURE", "Baseline")  # Default to Baseline if missing
    patch_size = config["PATCH_SIZE"]
    n_quantiles = len(config["QUANTILE_LEVELS"])
    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)
    input_shape = (1, patch_size, patch_size)

    # 3. Instantiate Model based on Architecture
    print(f"Initializing Architecture: {arch}")

    if arch == "Baseline":
        model = BaselineCNN(n_quantiles=n_quantiles, input_shape=input_shape)

    elif arch == "Lipschitz":
        model = LipschitzCNN(
            n_quantiles=n_quantiles,
            input_shape=input_shape,
            max_input_val=max_input_val,
        )

    elif arch == "Constrained":
        model = ConstrainedLipschitzCNN(
            n_quantiles=n_quantiles,
            input_shape=input_shape,
            quantile_levels=config["QUANTILE_LEVELS"],
            pixel_area_km2=pixel_size_km**2,  # Pass Area (km^2), not length
        )

    else:
        raise ValueError(
            f"Unknown Architecture: '{arch}'. "
            "Supported: 'Baseline', 'Lipschitz', 'Constrained'."
        )

    model = model.to(device)

    # 4. Load Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Checkpoint might be the full dict (with optimizer, epoch) or just state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Clean state_dict if it was saved with DataParallel ('module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    except Exception as e:
        print(f"\nCRITICAL ERROR loading emulator weights: {e}")
        print(f"Expected Architecture: {arch}")
        print("Ensure the config['ARCHITECTURE'] matches the checkpoint file.")
        raise e

    # 5. Freeze and Eval
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Emulator successfully loaded, set to eval mode, and weights frozen.")
    return model
