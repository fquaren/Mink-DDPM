import os
import yaml
import json
import argparse
import numpy as np
import xarray as xr
import zarr
from scipy.ndimage import zoom
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def declutter_precip(arr, threshold):
    arr_copy = arr.copy()
    arr_copy[arr_copy > threshold] = 0
    return arr_copy


def coarsen_array(arr, factor):
    m, n = arr.shape
    m_new, n_new = m // factor, n // factor
    return (
        arr[: m_new * factor, : n_new * factor]
        .reshape(m_new, factor, n_new, factor)
        .mean(axis=(1, 3))
    )


def interpolate_array(arr, factor, target_shape):
    interpolated = zoom(arr, zoom=factor, order=3)

    # NOTE: Bicubic interpolation causes negative ringing artifacts.
    # Precipitation is strictly >= 0.
    interpolated = np.clip(interpolated, a_min=0.0, a_max=None)

    h, w = interpolated.shape
    th, tw = target_shape
    if h > th or w > tw:
        interpolated = interpolated[:th, :tw]
    elif h < th or w < tw:
        interpolated = np.pad(interpolated, ((0, th - h), (0, tw - w)), mode="constant")
    return interpolated


def process_and_write_patch(args):
    """Worker function to process a single patch and write it to the target Zarr store."""
    (idx, patch_meta, timestamp_map, output_zarr_path, group_name, config) = args
    try:
        timestamp_str, y_start, x_start = patch_meta
        patch_size = config["PATCH_SIZE"]
        source_folder, time_idx_in_folder = timestamp_map[timestamp_str]
        with xr.open_zarr(source_folder, consolidated=True) as ds:
            original_precip = (
                ds[config["PRECIP_VAR_NAME"]]
                .isel(
                    time=time_idx_in_folder,
                    y=slice(y_start, y_start + patch_size),
                    x=slice(x_start, x_start + patch_size),
                )
                .load()
                .values
            )

        original_precip[np.isnan(original_precip)] = 0.0
        decluttered = declutter_precip(original_precip, config["DECLUTTER_THRESHOLD"])
        low_res = coarsen_array(decluttered, config["DOWNSCALING_FACTOR"])
        interpolated = interpolate_array(
            low_res, config["DOWNSCALING_FACTOR"], target_shape=(patch_size, patch_size)
        )

        target_zarr = zarr.open(output_zarr_path, mode="r+")
        target_zarr[f"{group_name}/original_precip"][idx] = original_precip.astype(
            np.float32
        )
        target_zarr[f"{group_name}/interpolated_precip"][idx] = interpolated.astype(
            np.float32
        )
        target_zarr[f"{group_name}/coarse_precip"][idx] = low_res.astype(np.float32)
        return None
    except Exception as e:
        return f"Error processing patch {patch_meta} at index {idx}: {e}"


def compute_global_scaler(zarr_path, config):
    """
    Computes the maximum log1p value from the training split
    to be used for domain scaling [-1, 1] in the DDPM.
    """
    print("\n--- Computing Global Training Scaler ---")
    dataset = zarr.open(zarr_path, mode="r")

    if "train" not in dataset:
        print("Warning: 'train' group not found in Zarr. Cannot compute scaler.")
        return

    train_data = dataset["train/original_precip"]
    chunk_size = 5000
    global_max = 0.0

    # Process in chunks to avoid OOM errors
    for i in tqdm(
        range(0, train_data.shape[0], chunk_size), desc="Scanning Train Data"
    ):
        chunk = train_data[i : i + chunk_size]
        # Transform to log space exactly as the dataset will do
        chunk_log = np.log1p(chunk)
        chunk_max = np.max(chunk_log)
        if chunk_max > global_max:
            global_max = chunk_max

    # Save the scaler for the PyTorch dataloader
    scaler_path = os.path.join(config["PREPROCESSED_DATA_DIR"], "precip_max_val.npy")
    np.save(scaler_path, np.array([global_max]))
    print(f"Global Log1p Max saved to {scaler_path}: {global_max:.4f}")


def main():
    """Main script to preprocess data in parallel and save to a consolidated Zarr store."""
    parser = argparse.ArgumentParser(
        description="Preprocess patch data in parallel from metadata files."
    )
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    print("Loading timestamp map from file...")
    map_path = os.path.join(config["METADATA_DIR"], "timestamp_map.json")
    with open(map_path, "r") as f:
        timestamp_map = json.load(f)

    metadata_paths = {
        "train": os.path.join(config["METADATA_DIR"], "train_patches_metadata.txt"),
        "validation": os.path.join(config["METADATA_DIR"], "val_patches_metadata.txt"),
        "test": os.path.join(config["METADATA_DIR"], "test_patches_metadata.txt"),
    }

    output_zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    print(f"Initializing output Zarr store at: {output_zarr_path}")
    root = zarr.open(output_zarr_path, mode="w")

    all_tasks = {}
    patch_size = config["PATCH_SIZE"]
    coarse_patch_size = patch_size // config["DOWNSCALING_FACTOR"]

    for group_name, path in metadata_paths.items():
        with open(path, "r") as f:
            lines = f.readlines()

        num_patches = len(lines)
        if num_patches == 0:
            continue

        group = root.create_group(group_name)
        group.create_dataset(
            "original_precip",
            shape=(num_patches, patch_size, patch_size),
            chunks=(1, patch_size, patch_size),
            dtype="float32",
        )
        group.create_dataset(
            "interpolated_precip",
            shape=(num_patches, patch_size, patch_size),
            chunks=(1, patch_size, patch_size),
            dtype="float32",
        )
        group.create_dataset(
            "coarse_precip",
            shape=(num_patches, coarse_patch_size, coarse_patch_size),
            chunks=(1, coarse_patch_size, coarse_patch_size),
            dtype="float32",
        )

        tasks = []
        for i, line in enumerate(lines):
            timestamp_str, y_str, x_str, _ = line.strip().split(",")
            patch_meta = (timestamp_str, int(y_str), int(x_str))
            tasks.append(
                (i, patch_meta, timestamp_map, output_zarr_path, group_name, config)
            )
        all_tasks[group_name] = tasks

    for group_name, tasks in all_tasks.items():
        print(
            f"\n--- Starting parallel processing for '{group_name}' set ({len(tasks)} patches) ---"
        )
        with ProcessPoolExecutor(max_workers=config["MAX_WORKERS"]) as executor:
            futures = [executor.submit(process_and_write_patch, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc=f"Processing {group_name}"
            ):
                result = future.result()
                if result:
                    print(result)

    print("\nPreprocessing pipeline completed successfully.")

    # Compute the scaler
    compute_global_scaler(output_zarr_path, config)


if __name__ == "__main__":
    main()
