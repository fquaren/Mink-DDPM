import os
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2


class ZarrMixupDataset(Dataset):
    def __init__(
        self,
        zarr_path,
        split="train",
        scaler_val=1.0,
        augment=True,
        include_original=False,
        include_mixup=True,
        subset_fraction=1.0,
    ):
        self.zarr_path = zarr_path
        self.split = split
        self.scaler_val = float(scaler_val)
        self.augment = augment

        self.store = zarr.open(zarr_path, mode="r")
        if split not in self.store:
            raise KeyError(f"Split '{split}' not found in zarr store.")

        self.group = self.store[split]
        self.data_keys = []
        self.target_keys = []
        self.lengths = []

        if include_original and "original_precip" in self.group:
            self.data_keys.append("original_precip")
            self.target_keys.append("gamma_targets")
            self.lengths.append(self.group["original_precip"].shape[0])

        if include_mixup and "mixup_precip" in self.group:
            self.data_keys.append("mixup_precip")
            self.target_keys.append("mixup_gamma_targets")
            self.lengths.append(self.group["mixup_precip"].shape[0])

        if not self.data_keys:
            raise ValueError("No valid data arrays found based on include parameters.")

        self.cumulative_sizes = np.cumsum(self.lengths)
        self.total_len = self.cumulative_sizes[-1]
        self.indices_map = np.arange(self.total_len)

        if 0.0 < subset_fraction < 1.0:
            subset_size = int(self.total_len * subset_fraction)
            rng = np.random.default_rng(seed=42)
            self.indices_map = rng.choice(
                self.indices_map, size=max(1, subset_size), replace=False
            )

        if self.augment:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomChoice(
                        [v2.RandomRotation([d, d]) for d in [0, 90, 180, 270]]
                    ),
                ]
            )

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, idx):
        real_idx = self.indices_map[idx]

        source_idx = np.searchsorted(self.cumulative_sizes, real_idx, side="right")
        local_idx = (
            real_idx
            if source_idx == 0
            else real_idx - self.cumulative_sizes[source_idx - 1]
        )

        d_key = self.data_keys[source_idx]
        t_key = self.target_keys[source_idx]

        # Read specific index from zarr (Physical Units)
        patch = self.group[d_key][local_idx]
        target_phys = self.group[t_key][local_idx]

        # Log transformation and [0, 1] normalization
        # Note: the [0,1] scaling is done using the global max log-precip value computed during
        # preprocessing on the original data, not the mixup data, to maintain consistency.
        patch_log = np.log1p(patch)
        patch_normalized = patch_log / self.scaler_val
        patch_normalized = np.clip(patch_normalized, 0.0, 1.0)

        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(patch_normalized).float().unsqueeze(0)
        target_phys_tensor = torch.from_numpy(target_phys).float()

        if self.augment:
            input_tensor = self.transform(input_tensor)

        # Gamma targets are log-transformed for the loss function evaluation
        log_target_gamma = torch.log1p(target_phys_tensor)

        return input_tensor, log_target_gamma, input_tensor, target_phys_tensor


class DeterministicSRDataset(Dataset):
    """
    Super-resolution dataset for UNet training.
    Integrates zarr-based I/O, dynamic sample weighting for stratified sampling, and normalization [0, 1].
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
        wet_dry_ratio=1.0,
    ):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats
        self.scaler_max_val = scaler_max_val
        self.split = split
        self.is_train = split == "train"

        self.zarr_path = os.path.join(
            self.preprocessed_data_dir, "preprocessed_dataset.zarr"
        )
        self.store = None
        self.group = None

        print(f"Loading and profiling {split} metadata...")

        self.metadata = []
        is_wet_list = []

        # 1. Parse metadata
        with open(metadata_file, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(",")
                if len(parts) == 4:
                    ts, y, x, p_max = parts
                    p_max = float(p_max)
                    self.metadata.append((ts, int(y), int(x), p_max))
                    is_wet_list.append(p_max > 0.1)

        self.valid_indices = np.arange(len(self.metadata))
        is_wet_array = np.array(is_wet_list)

        # 2. Apply optional subsetting first
        if 0.0 < subset_fraction < 1.0:
            total_samples = len(self.valid_indices)
            subset_size = max(int(total_samples * subset_fraction), 1)
            rng = np.random.default_rng(seed=42)
            self.valid_indices = rng.choice(
                self.valid_indices, size=subset_size, replace=False
            )
            # Filter the wet/dry boolean array to match the subset
            is_wet_array = is_wet_array[self.valid_indices]
        elif subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(
                f"subset_fraction must be in (0, 1]. Received {subset_fraction}"
            )

        # 3. Compute sample weights for stratified sampling
        self.sample_weights = None
        if self.is_train and wet_dry_ratio is not None:
            n_wet = np.sum(is_wet_array)
            n_dry = len(is_wet_array) - n_wet

            # Avoid division by zero if a class is entirely missing
            weight_wet = 1.0 / n_wet if n_wet > 0 else 0.0
            weight_dry = (1.0 / n_dry) * wet_dry_ratio if n_dry > 0 else 0.0

            self.sample_weights = np.where(is_wet_array, weight_wet, weight_dry)
            print(f"Dataset weighted: {n_wet} wet, {n_dry} dry instances available.")

        # 4. Geometrical transforms
        self.geom_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

    def _init_zarr(self):
        if self.store is None:
            self.store = zarr.open(self.zarr_path, mode="r")
            self.group = self.store[self.split]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        self._init_zarr()
        real_idx = self.valid_indices[idx]

        target_phys = self.group["original_precip"][real_idx]
        interp_phys = self.group["interpolated_precip"][real_idx]
        gamma_phys = self.group["gamma_targets"][real_idx]

        target_log = np.log1p(target_phys)
        interp_log = np.log1p(interp_phys)
        target_norm = np.clip(target_log / self.scaler_max_val, 0.0, 1.0)
        interp_norm = np.clip(interp_log / self.scaler_max_val, 0.0, 1.0)

        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)
        interp_tensor = torch.from_numpy(interp_norm).float().unsqueeze(0)

        dem_patch = self.group["dem"][real_idx]
        dem_patch = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        dem_tensor = torch.from_numpy(dem_patch).float().unsqueeze(0)

        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        if self.is_train:
            input_stack, target_tensor = self.geom_transform(input_stack, target_tensor)

        target_gamma_tensor = torch.from_numpy(np.log1p(gamma_phys)).float()

        return input_stack, target_tensor, target_gamma_tensor


class DiffusionSRDataset(Dataset):
    """
    Super-resolution dataset for diffusion training.
    Integrates zarr-based I/O, STRICT natural physical distribution, and domain shifting [-1, 1].
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
    ):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.dem_patches_dir = dem_patches_dir
        self.dem_mean, self.dem_std = dem_stats
        self.scaler_max_val = scaler_max_val
        self.split = split
        self.is_train = split == "train"

        self.zarr_path = os.path.join(
            self.preprocessed_data_dir, "preprocessed_dataset.zarr"
        )
        self.store = None
        self.group = None

        print(f"Loading {split} metadata...")

        # 1. Parse metadata (no wet/dry splitting required)
        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    ts, y, x, p_max = parts
                    self.metadata.append((ts, int(y), int(x), float(p_max)))

        self.valid_indices = np.arange(len(self.metadata))

        # 2. Apply optional subsetting
        if 0.0 < subset_fraction < 1.0:
            total_samples = len(self.valid_indices)
            subset_size = max(int(total_samples * subset_fraction), 1)
            print(f"Subsetting {split} dataset to {subset_fraction*100:.1f}%.")
            rng = np.random.default_rng(seed=42)
            self.valid_indices = rng.choice(
                self.valid_indices, size=subset_size, replace=False
            )
        elif subset_fraction <= 0.0 or subset_fraction > 1.0:
            raise ValueError(
                f"subset_fraction must be in (0, 1]. Received {subset_fraction}"
            )

        # 3. Handle legacy gamma targets
        gamma_path = os.path.join(
            self.preprocessed_data_dir, split, "gamma_targets_persistence.npz"
        )
        if not os.path.exists(gamma_path):
            self.gamma_targets = np.zeros((len(self.metadata), 3), dtype=np.float32)
        else:
            self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        # 4. Geometrical transforms
        self.geom_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )

    def _init_zarr(self):
        if self.store is None:
            self.store = zarr.open(self.zarr_path, mode="r")
            self.group = self.store[self.split]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        self._init_zarr()
        real_idx = self.valid_indices[idx]

        target_phys = self.group["original_precip"][real_idx]
        interp_phys = self.group["interpolated_precip"][real_idx]
        gamma_phys = self.group["gamma_targets"][real_idx]

        target_log = np.log1p(target_phys)
        interp_log = np.log1p(interp_phys)

        target_norm = np.clip(target_log / self.scaler_max_val, 0.0, 1.0)
        interp_norm = np.clip(interp_log / self.scaler_max_val, 0.0, 1.0)

        # Domain mapping to [-1, 1] for generative diffusion training
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0) * 2.0 - 1.0
        interp_tensor = torch.from_numpy(interp_norm).float().unsqueeze(0) * 2.0 - 1.0

        dem_patch = self.group["dem"][real_idx]
        dem_patch = (dem_patch - self.dem_mean) / (self.dem_std + 1e-8)
        dem_tensor = torch.from_numpy(dem_patch).float().unsqueeze(0)

        input_stack = torch.cat([interp_tensor, dem_tensor], dim=0)

        if self.is_train:
            input_stack, target_tensor = self.geom_transform(input_stack, target_tensor)

        target_gamma_tensor = torch.from_numpy(np.log1p(gamma_phys)).float()

        return input_stack, target_tensor, target_gamma_tensor
