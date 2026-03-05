import os
import yaml
import json
import argparse
import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def filter_precip_bounds(arr, min_thresh, max_thresh):
    """Applies both drizzle (lower) and declutter (upper) physical thresholds."""
    arr_copy = arr.copy()
    arr_copy[(arr_copy < min_thresh) | (arr_copy > max_thresh)] = 0.0
    return arr_copy


def coarsen_and_interpolate_conservative(arr, factor):
    """
    Applies area-weighted block mean for coarsening and
    nearest-neighbor interpolation for strict mass conservation.
    """
    m, n = arr.shape
    m_new, n_new = m // factor, n // factor

    # Block mean (Conservative decimation)
    low_res = (
        arr[: m_new * factor, : n_new * factor]
        .reshape(m_new, factor, n_new, factor)
        .mean(axis=(1, 3))
    )

    # Nearest-neighbor interpolation (Strictly non-negative)
    interpolated = np.kron(low_res, np.ones((factor, factor)))

    return low_res, interpolated


def process_batch(batch_payload):
    """Worker function to process a batch of patches to minimize I/O overhead."""
    static_args = batch_payload["static"]
    tasks = batch_payload["tasks"]
    output_zarr_path, dem_path, group_name, config = static_args
    patch_size = config["PATCH_SIZE"]

    results = []

    # --- 1. Establish Static I/O Endpoints ONCE per batch ---
    target_zarr = zarr.open(output_zarr_path, mode="r+")

    try:
        # Changed to open_zarr to bypass hdf5 posix locking
        with xr.open_zarr(dem_path) as ds_dem:
            # Load fully into memory for fast slicing
            dem_memory = ds_dem["elevation"].load()
    except Exception as e:
        return [f"Critical error: Batch failed to load static DEM zarr store: {e}"]

    # Cache for source precipitation datasets to avoid reopening the same day within a batch
    source_ds_cache = {}

    for task in tasks:
        idx, patch_meta, timestamp_map = task
        timestamp_str, y_start, x_start = patch_meta

        try:
            source_folder, time_idx_in_folder = timestamp_map[timestamp_str]

            # --- 2. Extract Precipitation with In-Batch Caching ---
            if source_folder not in source_ds_cache:
                try:
                    ds = xr.open_zarr(source_folder, consolidated=True)
                    if not ds.dims:
                        ds = xr.open_zarr(source_folder, consolidated=False)
                except Exception:
                    ds = xr.open_zarr(source_folder, consolidated=False)
                source_ds_cache[source_folder] = ds

            ds = source_ds_cache[source_folder]

            if "y" not in ds.dims or "x" not in ds.dims:
                results.append(
                    f"Skipped index {idx}: Source {source_folder} dimensions missing."
                )
                continue

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

            h, w = original_precip.shape
            if h != patch_size or w != patch_size:
                results.append(
                    f"Skipped patch {patch_meta} at index {idx}: boundary truncation to shape ({h}, {w})"
                )
                continue

            # --- 3. Extract Elevation (DEM) from memory ---
            dem_patch = dem_memory.isel(
                y=slice(y_start, y_start + patch_size),
                x=slice(x_start, x_start + patch_size),
            ).values

            dh, dw = dem_patch.shape
            if dh != patch_size or dw != patch_size:
                results.append(
                    f"Skipped patch {patch_meta} at index {idx}: DEM boundary truncation."
                )
                continue

            # --- 4. Process Precipitation Physics ---
            original_precip[np.isnan(original_precip)] = 0.0

            drizzle_thresh = config.get("DRIZZLE_THRESHOLD", 0.1)
            declutter_thresh = config.get("DECLUTTER_THRESHOLD", 150.0)
            filtered_precip = filter_precip_bounds(
                original_precip, drizzle_thresh, declutter_thresh
            )

            low_res, interpolated = coarsen_and_interpolate_conservative(
                filtered_precip, config["DOWNSCALING_FACTOR"]
            )

            # --- 5. Write to Zarr Store ---
            target_zarr[f"{group_name}/original_precip"][idx] = filtered_precip.astype(
                np.float32
            )
            target_zarr[f"{group_name}/interpolated_precip"][idx] = interpolated.astype(
                np.float32
            )
            target_zarr[f"{group_name}/coarse_precip"][idx] = low_res.astype(np.float32)
            target_zarr[f"{group_name}/dem"][idx] = dem_patch.astype(np.float32)

        except Exception as e:
            results.append(f"Error processing patch {patch_meta} at index {idx}: {e}")

    return results if results else None


def compute_dem_stats(zarr_path, config):
    """
    Computes the global mean and standard deviation of the DEM patches
    from the training split and saves them to a JSON file.
    """
    print("\n--- Computing DEM Statistics ---")
    dataset = zarr.open(zarr_path, mode="r")

    if "train" not in dataset or "dem" not in dataset["train"]:
        print("Warning: 'train/dem' group not found in Zarr. Cannot compute DEM stats.")
        return

    dem_data = dataset["train/dem"]
    chunk_size = 5000

    # Use float64 accumulators to prevent numerical overflow
    count = 0
    sum_val = np.float64(0.0)
    sum_sq_val = np.float64(0.0)

    for i in tqdm(range(0, dem_data.shape[0], chunk_size), desc="Scanning DEM Data"):
        chunk = dem_data[i : i + chunk_size]
        # Ignore NaNs if any slipped through
        valid_mask = ~np.isnan(chunk)
        valid_chunk = chunk[valid_mask]

        count += valid_chunk.size
        sum_val += np.sum(valid_chunk, dtype=np.float64)
        sum_sq_val += np.sum(valid_chunk**2, dtype=np.float64)

    if count == 0:
        print("Error: No valid DEM data found to compute statistics.")
        return

    dem_mean = sum_val / count
    dem_var = (sum_sq_val / count) - (dem_mean**2)
    dem_std = np.sqrt(dem_var)

    stats = {"dem_mean": float(dem_mean), "dem_std": float(dem_std)}

    stats_path = config.get(
        "DEM_STATS", os.path.join(config["PREPROCESSED_DATA_DIR"], "dem_stats.json")
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(
        f"DEM Statistics saved to {stats_path}: Mean = {dem_mean:.2f}, Std = {dem_std:.2f}"
    )


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

    for i in tqdm(
        range(0, train_data.shape[0], chunk_size), desc="Scanning Train Data"
    ):
        chunk = train_data[i : i + chunk_size]
        chunk_log = np.log1p(chunk)
        chunk_max = np.max(chunk_log)
        if chunk_max > global_max:
            global_max = chunk_max

    scaler_path = os.path.join(config["PREPROCESSED_DATA_DIR"], "precip_max_val.npy")
    np.save(scaler_path, np.array([global_max]))
    print(f"Global Log1p Max saved to {scaler_path}: {global_max:.4f}")


def main():
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

    all_batched_payloads = {}
    patch_size = config["PATCH_SIZE"]
    coarse_patch_size = patch_size // config["DOWNSCALING_FACTOR"]
    batch_size = config.get("BATCH_SIZE", 1000)

    for group_name, path in metadata_paths.items():
        with open(path, "r") as f:
            lines = f.readlines()

        num_patches = len(lines)
        if num_patches == 0:
            continue

        group = root.create_group(group_name)

        # Initialize datasets using the Zarr v2 specification
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
        group.create_dataset(
            "dem",
            shape=(num_patches, patch_size, patch_size),
            chunks=(1, patch_size, patch_size),
            dtype="float32",
        )

        # Ensure the script explicitly targets a zarr directory even if config says .nc
        dem_path_zarr = config["STATIC_DEM_PATH"].replace(".nc", ".zarr")
        static_args = (output_zarr_path, dem_path_zarr, group_name, config)

        dynamic_tasks = []
        for i, line in enumerate(lines):
            timestamp_str, y_str, x_str, _ = line.strip().split(",")
            patch_meta = (timestamp_str, int(y_str), int(x_str))
            dynamic_tasks.append((i, patch_meta, timestamp_map))

        batched_payloads = []
        for i in range(0, len(dynamic_tasks), batch_size):
            batch = dynamic_tasks[i : i + batch_size]
            batched_payloads.append({"static": static_args, "tasks": batch})

        all_batched_payloads[group_name] = batched_payloads

    for group_name, batched_payloads in all_batched_payloads.items():
        total_patches = sum(len(p["tasks"]) for p in batched_payloads)
        print(
            f"\n--- Starting parallel processing for '{group_name}' set ({total_patches} patches across {len(batched_payloads)} batches) ---"
        )

        with ProcessPoolExecutor(max_workers=config["MAX_WORKERS"]) as executor:
            futures = [
                executor.submit(process_batch, payload) for payload in batched_payloads
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(batched_payloads),
                desc=f"Processing {group_name}",
            ):
                results = future.result()
                if results:
                    for res in results:
                        print(res)

    print("\nPreprocessing pipeline completed successfully.")
    compute_global_scaler(output_zarr_path, config)
    compute_dem_stats(output_zarr_path, config)


if __name__ == "__main__":
    main()
