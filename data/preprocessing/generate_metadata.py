import os
import glob
import yaml
import argparse
import shutil
import json
import numpy as np
import xarray as xr
import numba
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


@numba.jit(nopython=True)
def find_valid_patches_numba(frame, patch_size):
    """Uses Numba to JIT-compile a fast loop for finding valid patches and their max intensity."""
    valid_patches_info = []
    frame_h, frame_w = frame.shape
    for y in range(frame_h - patch_size + 1):
        for x in range(frame_w - patch_size + 1):
            patch = frame[y : y + patch_size, x : x + patch_size]
            if not np.isnan(patch).any():
                # Extract the maximum physical intensity of this patch
                patch_max = np.max(patch)
                valid_patches_info.append((y, x, patch_max))
    return valid_patches_info


def scan_zarr_folder_for_patches(folder_path, precip_var_name, patch_size):
    """Worker function: Scans a single daily Zarr folder."""
    local_coords_lines = []
    local_timestamp_map = {}
    try:
        with xr.open_zarr(folder_path, consolidated=True) as ds:
            for t_idx in range(len(ds.time)):
                frame = ds[precip_var_name].isel(time=t_idx).load().values
                timestamp_dt_numpy = ds.time.isel(time=t_idx).values
                timestamp_str = (
                    np.datetime_as_string(timestamp_dt_numpy, unit="s")
                    .replace("-", "")
                    .replace("T", "")
                    .replace(":", "")
                )
                local_timestamp_map[timestamp_str] = (folder_path, t_idx)
                valid_patches_in_frame = find_valid_patches_numba(frame, patch_size)
                for y, x, patch_max in valid_patches_in_frame:
                    # Append the max value to the metadata string
                    local_coords_lines.append(
                        f"{timestamp_str},{y},{x},{patch_max:.4f}\n"
                    )
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
    return local_coords_lines, local_timestamp_map


def save_metadata_to_file(metadata, filepath):
    """Saves a list of metadata strings to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.writelines(metadata)
    print(f"Saved {len(metadata)} entries to {filepath}")


def main():
    """Main script to orchestrate parallel scanning and result consolidation."""
    parser = argparse.ArgumentParser(
        description="Generate patch metadata and timestamp map in parallel."
    )
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
    METADATA_DIR = config["METADATA_DIR"]
    PATCH_SIZE = config["PATCH_SIZE"]
    PRECIP_VAR_NAME = config["PRECIP_VAR_NAME"]
    MAX_WORKERS = config.get("MAX_WORKERS", 4)

    TEMP_METADATA_DIR = os.path.join(METADATA_DIR, "temp_metadata")
    if os.path.exists(TEMP_METADATA_DIR):
        shutil.rmtree(TEMP_METADATA_DIR)
    os.makedirs(TEMP_METADATA_DIR)

    zarr_folders = sorted(glob.glob(os.path.join(RAW_OPERA_DATA_DIR, "[0-9]" * 8)))
    if not zarr_folders:
        raise FileNotFoundError(f"No Zarr folders found in {RAW_OPERA_DATA_DIR}")
    print(
        f"Found {len(zarr_folders)} days to process with up to {MAX_WORKERS} parallel workers."
    )

    timestamp_map = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(
                scan_zarr_folder_for_patches, folder, PRECIP_VAR_NAME, PATCH_SIZE
            ): os.path.join(TEMP_METADATA_DIR, f"{os.path.basename(folder)}.txt")
            for folder in zarr_folders
        }
        for future in tqdm(
            as_completed(future_to_path),
            total=len(zarr_folders),
            desc="Processing daily files",
        ):
            output_path = future_to_path[future]
            try:
                result_lines, local_map = future.result()
                if result_lines:
                    with open(output_path, "w") as f:
                        f.writelines(result_lines)
                if local_map:
                    timestamp_map.update(local_map)
            except Exception as exc:
                print(f"A job generated an exception: {exc}")

    map_path = os.path.join(METADATA_DIR, "timestamp_map.json")
    with open(map_path, "w") as f:
        json.dump(timestamp_map, f)
    print(f"\nSaved timestamp map with {len(timestamp_map)} entries to {map_path}")


if __name__ == "__main__":
    main()
