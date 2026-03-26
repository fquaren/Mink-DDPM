import os
import yaml
import zarr
import numpy as np
import argparse
import gudhi as gd
from skimage import measure
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Prevent numpy/MKL/OpenBLAS from using internal threads during multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def compute_climatological_thresholds(zarr_path, quantiles, drizzle_threshold=0.1):
    """
    Computes global physical thresholds from the training distribution.
    Strictly excludes background zeros to prevent degenerate low quantiles.
    """
    print("--- Computing Global Climatological Thresholds (Empirical CDF) ---")
    store = zarr.open(zarr_path, mode="r")
    if "train" not in store:
        raise ValueError("Train group not found in Zarr store.")

    train_data = store["train/original_precip"]
    chunk_size = 5000

    rng = np.random.default_rng(seed=42)
    indices = np.arange(train_data.shape[0])
    rng.shuffle(indices)

    sampled_wet_pixels = []
    max_pixels_to_sample = 50_000_000
    current_pixel_count = 0

    for i in tqdm(range(0, len(indices), chunk_size), desc="Sampling wet pixels"):
        if current_pixel_count >= max_pixels_to_sample:
            break

        chunk_indices = indices[i : i + chunk_size]
        chunk_indices = np.sort(chunk_indices)
        chunk = train_data.oindex[chunk_indices]

        wet_mask = chunk > drizzle_threshold
        wet_vals = chunk[wet_mask]

        sampled_wet_pixels.append(wet_vals)
        current_pixel_count += len(wet_vals)

    all_wet_pixels = np.concatenate(sampled_wet_pixels)
    print(f"Aggregated {len(all_wet_pixels)} wet pixels.")

    physical_thresholds = np.quantile(all_wet_pixels, quantiles)

    for q, t in zip(quantiles, physical_thresholds):
        print(f"Quantile {q:.2f} -> Threshold: {t:.4f} mm/h")

    return physical_thresholds.astype(np.float32)


def compute_tda_persistence(prec_2d_np_clean):
    """Computes the persistence diagram for a given image."""
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    return cubical_complex.persistence()


def compute_gamma_matrix(
    prec_2d_data, physical_thresholds, pixel_size_km, thresh_b0, thresh_b1
):
    """
    Computes the Gamma matrix (Area, Perimeter, Betti_0, Betti_1)
    applying distinct persistence thresholds for components and holes.
    """
    gamma_matrix = np.zeros((4, len(physical_thresholds)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    thresholds_broadcast_1d = physical_thresholds[np.newaxis, :]

    # --- 1. TDA Calculation for Betti 0 (Connected Components, Feature Index 2) ---
    pairs_d0 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 0], dtype=np.float64
    )

    if pairs_d0.shape[0] > 0:
        births_0 = -pairs_d0[:, 0]
        deaths_0 = -pairs_d0[:, 1]
        is_finite_0 = deaths_0 != -np.inf
        is_background_0 = ~is_finite_0
        deaths_0[is_background_0] = np.inf
        persistence_0 = births_0 - deaths_0
        persistence_0[is_background_0] = np.inf

        # Apply specific B0 threshold
        is_significant_0 = persistence_0 > thresh_b0
        pers_thresh_mask_0 = is_significant_0[:, np.newaxis]

        birth_thresh_mask_0 = births_0[:, np.newaxis] >= thresholds_broadcast_1d
        death_thresh_mask_0 = deaths_0[:, np.newaxis] < thresholds_broadcast_1d

        finite_pass_mask_0 = (
            pers_thresh_mask_0
            & birth_thresh_mask_0
            & death_thresh_mask_0
            & is_finite_0[:, np.newaxis]
        )

        background_low_thresh_mask = thresholds_broadcast_1d <= 0.01
        background_pass_mask_0 = (
            pers_thresh_mask_0
            & birth_thresh_mask_0
            & is_background_0[:, np.newaxis]
            & background_low_thresh_mask
        )

        gamma_matrix[2, :] = np.sum(finite_pass_mask_0, axis=0) + np.sum(
            background_pass_mask_0, axis=0
        )

    # --- 2. TDA Calculation for Betti 1 (Holes, Feature Index 3) ---
    pairs_d1 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 1], dtype=np.float64
    )

    if pairs_d1.shape[0] > 0:
        births_1 = -pairs_d1[:, 0]
        deaths_1 = -pairs_d1[:, 1]
        persistence_1 = births_1 - deaths_1

        # Apply specific B1 threshold
        is_significant_1 = persistence_1 > thresh_b1
        pers_thresh_mask_1 = is_significant_1[:, np.newaxis]

        birth_thresh_mask_1 = births_1[:, np.newaxis] >= thresholds_broadcast_1d
        death_thresh_mask_1 = deaths_1[:, np.newaxis] < thresholds_broadcast_1d

        pass_mask_1 = pers_thresh_mask_1 & birth_thresh_mask_1 & death_thresh_mask_1
        gamma_matrix[3, :] = np.sum(pass_mask_1, axis=0)

    # --- 3. Area and Perimeter Computation ---
    pixel_area_km2 = pixel_size_km**2
    prec_broadcast = prec_2d_np_clean[..., np.newaxis]
    thresholds_broadcast_3d = physical_thresholds[np.newaxis, np.newaxis, :]
    masks_3d = prec_broadcast >= thresholds_broadcast_3d

    # Area (Feature Index 0)
    gamma_matrix[0, :] = np.sum(masks_3d, axis=(0, 1)) * pixel_area_km2

    # Perimeter (Feature Index 1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for i in range(len(physical_thresholds)):
            mask_t = masks_3d[:, :, i]
            if not np.any(mask_t):
                continue
            contours = measure.find_contours(mask_t.astype(float), 0.5)
            perimeter_pixels = sum(
                np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in contours
            )
            gamma_matrix[1, i] = perimeter_pixels * pixel_size_km

    return gamma_matrix


def worker_process_chunk(args):
    """Worker function to process a chunk of images and save directly to Zarr."""
    (
        start_idx,
        end_idx,
        zarr_path,
        group_name,
        physical_thresholds,
        config,
        thresh_b0,
        thresh_b1,
    ) = args

    store = zarr.open(zarr_path, mode="r+")
    group = store[group_name]

    precip_chunk = group["original_precip"][start_idx:end_idx]
    gamma_chunk = np.zeros(
        (precip_chunk.shape[0], 4, len(physical_thresholds)), dtype=np.float32
    )

    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)

    for i in range(precip_chunk.shape[0]):
        gamma_chunk[i] = compute_gamma_matrix(
            precip_chunk[i], physical_thresholds, pixel_size_km, thresh_b0, thresh_b1
        )

    group["gamma_targets"][start_idx:end_idx] = gamma_chunk
    return f"Processed indices {start_idx} to {end_idx} in {group_name}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute Geometric Gamma Targets and append to Zarr."
    )
    parser.add_argument("config", type=str, help="Path to the config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    quantiles = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)

    physical_thresholds = compute_climatological_thresholds(zarr_path, quantiles)

    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    np.save(thresh_path, physical_thresholds)

    # --- Load Empirical Persistence Thresholds ---
    emp_thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "persistence_thresholds.yaml"
    )
    if os.path.exists(emp_thresh_path):
        print(
            f"\n--- Loading Empirical Persistence Thresholds from {emp_thresh_path} ---"
        )
        with open(emp_thresh_path, "r") as f:
            emp_thresh = yaml.safe_load(f)
        thresh_b0 = emp_thresh.get("PERSISTENCE_THRESHOLD_B0", 0.1)
        thresh_b1 = emp_thresh.get("PERSISTENCE_THRESHOLD_B1", 0.1)
    else:
        print(f"\n[WARNING] {emp_thresh_path} not found.")
        print("Falling back to global PERSISTENCE_THRESHOLD from config.yaml.")
        thresh_b0 = config.get("PERSISTENCE_THRESHOLD", 0.1)
        thresh_b1 = thresh_b0

    print(f"Applying Thresholds -> B0: {thresh_b0:.4f}, B1: {thresh_b1:.4f}")

    store = zarr.open(zarr_path, mode="r+")
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)
    print(f"\nUsing worker chunk size: {chunk_size} samples per worker process.")

    for split in ["train", "validation", "test"]:
        if split not in store:
            continue

        print(f"\n--- Extracting Topological Features: {split} ---")
        group = store[split]
        num_samples = group["original_precip"].shape[0]

        # Save metadata to Zarr attributes for downstream transparency
        group.attrs["persistence_threshold_b0"] = float(thresh_b0)
        group.attrs["persistence_threshold_b1"] = float(thresh_b1)

        if "gamma_targets" not in group:
            group.create_dataset(
                "gamma_targets",
                shape=(num_samples, 4, len(quantiles)),
                chunks=(chunk_size, 4, len(quantiles)),
                dtype="float32",
            )

        tasks = []
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            tasks.append(
                (
                    start_idx,
                    end_idx,
                    zarr_path,
                    split,
                    physical_thresholds,
                    config,
                    thresh_b0,
                    thresh_b1,
                )
            )

        max_workers = config.get("MAX_WORKERS", os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_process_chunk, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc=f"Writing {split}"
            ):
                future.result()

    print("\nGamma targets successfully appended to Zarr store.")


if __name__ == "__main__":
    main()
