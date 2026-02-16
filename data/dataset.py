import os
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRDataset(Dataset):
    """
    Super-Resolution Dataset for DDPM Training.
    Integrates Zarr-based I/O, physical wet/dry class balancing, and domain shifting [-1, 1].
    """

    def __init__(
        self,
        preprocessed_data_dir,
        metadata_file,
        dem_patches_dir,
        dem_stats,
        scaler_max_val,
        split="train",
        subset_fraction=1.0,
        wet_dry_ratio=1.0,  # 1.0 implies 1:1 ratio of wet to dry patches
    ):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats
        self.scaler_max_val = scaler_max_val
        self.split = split
        self.is_train = split == "train"

        # Lazy Zarr initialization (Required for PyTorch multiprocessing safety)
        self.zarr_path = os.path.join(
            self.preprocessed_data_dir, "preprocessed_dataset.zarr"
        )
        self.store = None
        self.group = None

        print(f"Loading and profiling {split} metadata...")

        # 1. Parse Metadata and Profile Physical Regimes
        self.metadata = []
        wet_indices = []
        dry_indices = []

        with open(metadata_file, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) == 4:
                    ts, y, x, p_max = parts
                    p_max = float(p_max)
                    self.metadata.append((ts, int(y), int(x), p_max))

                    # Strictly define physical domains
                    if p_max > 0.1:
                        wet_indices.append(i)
                    else:
                        dry_indices.append(i)

        # 2. OPTIMIZATION: Zero-Filtering & Class Balancing (Method A)
        if self.is_train and wet_dry_ratio is not None:
            n_dry_to_keep = int(len(wet_indices) * wet_dry_ratio)

            # Use fixed seed to ensure reproducible dataset epochs
            rng = np.random.default_rng(seed=42)

            if len(dry_indices) > n_dry_to_keep:
                keep_dry = rng.choice(dry_indices, size=n_dry_to_keep, replace=False)
            else:
                keep_dry = dry_indices

            self.valid_indices = np.concatenate([wet_indices, keep_dry])
            rng.shuffle(self.valid_indices)

            print(f"Dataset Balanced: {len(wet_indices)} Wet, {len(keep_dry)} Dry.")
        else:
            # Validation/Test must preserve the true climatological distribution
            self.valid_indices = np.arange(len(self.metadata))

        # 3. Apply Optional Subsetting
        if 0.0 < subset_fraction < 1.0:
            total_samples = len(self.valid_indices)
            subset_size = max(int(total_samples * subset_fraction), 1)
            print(
                f"Subsetting {split} dataset to {subset_fraction*100:.1f}% ({subset_size}/{total_samples})."
            )

            rng = np.random.default_rng(seed=42)
            self.valid_indices = rng.choice(
                self.valid_indices, size=subset_size, replace=False
            )
        elif subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(
                f"subset_fraction must be in (0, 1]. Received {subset_fraction}"
            )

        # 4. Handle Legacy Gamma Targets (Assuming they remain in .npz format)
        gamma_path = os.path.join(
            self.preprocessed_data_dir, split, "gamma_targets_persistence.npz"
        )
        if not os.path.exists(gamma_path):
            print(f"Warning: Gamma targets not found at {gamma_path}. Using zeros.")
            # Shape matches metadata size
            self.gamma_targets = np.zeros((len(self.metadata), 3), dtype=np.float32)
        else:
            self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        # 5. Geometrical Transforms
        self.geom_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ]
        )

    def _init_zarr(self):
        """Initializes the Zarr store lazily in the worker process."""
        if self.store is None:
            self.store = zarr.open(self.zarr_path, mode="r")
            self.group = self.store[self.split]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Ensure Zarr is mounted for this specific CPU worker
        self._init_zarr()

        # Map virtual index to physical array index
        real_idx = self.valid_indices[idx]

        # 1. Fetch Physical Tensors from Zarr
        target_phys = self.group["original_precip"][real_idx]
        interp_phys = self.group["interpolated_precip"][real_idx]

        # Fetch Gamma Targets directly from Zarr
        gamma_phys = self.group["gamma_targets"][real_idx]

        # 2. Dynamic Normalization and Domain Shift
        # Math: x_norm = clip(log1p(x_phys) / S, 0, 1)
        # Math: x_scaled = x_norm * 2.0 - 1.0 -> Domain [-1, 1]
        target_log = np.log1p(target_phys)
        interp_log = np.log1p(interp_phys)

        target_norm = np.clip(target_log / self.scaler_max_val, 0.0, 1.0)
        interp_norm = np.clip(interp_log / self.scaler_max_val, 0.0, 1.0)

        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0) * 2.0 - 1.0
        interp_tensor = torch.from_numpy(interp_norm).float().unsqueeze(0) * 2.0 - 1.0

        # 3. Fetch DEM Context
        ts, y_coord, x_coord, _ = self.metadata[real_idx]
        dem_filename = f"dem_patch_y{y_coord:04d}_x{x_coord:04d}.npy"
        dem_path = os.path.join(self.dem_patches_dir, dem_filename)

        try:
            dem_patch = np.load(dem_path)
            dem_patch = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        except FileNotFoundError:
            # Fallback to zero-elevation if missing
            dem_patch = np.zeros_like(target_phys)

        # Assuming DEM is standardized (N(0,1)), we do NOT shift it to [-1, 1]
        dem_tensor = torch.from_numpy(dem_patch).float().unsqueeze(0)

        # 4. Construct Conditioning Stack
        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        # 5. Geometrical Augmentations
        if self.is_train:
            state = torch.get_rng_state()
            input_stack = self.geom_transform(input_stack)
            torch.set_rng_state(state)
            target_tensor = self.geom_transform(target_tensor)

        # 6. Format Gamma Targets (Log Transform)
        target_gamma_tensor = torch.from_numpy(np.log1p(gamma_phys)).float()

        return input_stack, target_tensor, target_gamma_tensor
