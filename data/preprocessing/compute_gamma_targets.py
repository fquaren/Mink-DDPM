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

    # We sample a large subset of physical values to estimate the CDF
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
        # Sort indices to optimize Zarr read operations
        chunk_indices = np.sort(chunk_indices)
        chunk = train_data.oindex[chunk_indices]

        # Isolate wet pixels
        wet_mask = chunk > drizzle_threshold
        wet_vals = chunk[wet_mask]

        sampled_wet_pixels.append(wet_vals)
        current_pixel_count += len(wet_vals)

    all_wet_pixels = np.concatenate(sampled_wet_pixels)
    print(f"Aggregated {len(all_wet_pixels)} wet pixels.")

    # Compute exact physical thresholds corresponding to the probability quantiles
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
    prec_2d_data, physical_thresholds, pixel_size_km, persistence_threshold
):
    """
    Computes the Gamma matrix (Area, Perimeter, Euler Characteristic)
    using the physical thresholds derived from the empirical CDF.
    """
    gamma_matrix = np.zeros((3, len(physical_thresholds)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    # --- 1. TDA Calculation for Euler Characteristic (Component 3) ---
    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    pairs_d0 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 0], dtype=np.float64
    )

    if pairs_d0.shape[0] > 0:
        births = -pairs_d0[:, 0]
        deaths = -pairs_d0[:, 1]
        is_finite = deaths != -np.inf
        is_background = ~is_finite
        deaths[is_background] = np.inf
        persistence = births - deaths
        persistence[is_background] = np.inf

        is_significant = persistence > persistence_threshold
        pers_thresh_mask = is_significant[:, np.newaxis]
        births_broadcast = births[:, np.newaxis]
        deaths_broadcast = deaths[:, np.newaxis]
        thresholds_broadcast_1d = physical_thresholds[np.newaxis, :]

        birth_thresh_mask = births_broadcast >= thresholds_broadcast_1d
        death_thresh_mask = deaths_broadcast < thresholds_broadcast_1d

        finite_pass_mask = (
            pers_thresh_mask
            & birth_thresh_mask
            & death_thresh_mask
            & is_finite[:, np.newaxis]
        )
        finite_counts = np.sum(finite_pass_mask, axis=0)

        # Count background component if threshold approaches zero
        background_low_thresh_mask = thresholds_broadcast_1d <= 0.01
        background_pass_mask = (
            pers_thresh_mask
            & birth_thresh_mask
            & is_background[:, np.newaxis]
            & background_low_thresh_mask
        )
        background_counts = np.sum(background_pass_mask, axis=0)

        gamma_matrix[2, :] = finite_counts + background_counts

    # --- 2. Area and Perimeter Computation ---
    pixel_area_km2 = pixel_size_km**2
    prec_broadcast = prec_2d_np_clean[..., np.newaxis]
    thresholds_broadcast_3d = physical_thresholds[np.newaxis, np.newaxis, :]
    masks_3d = prec_broadcast >= thresholds_broadcast_3d

    # Area
    area_counts = np.sum(masks_3d, axis=(0, 1))
    gamma_matrix[0, :] = area_counts * pixel_area_km2

    # Perimeter
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
    start_idx, end_idx, zarr_path, group_name, physical_thresholds, config = args

    store = zarr.open(zarr_path, mode="r+")
    group = store[group_name]

    # Read chunk into memory
    precip_chunk = group["original_precip"][start_idx:end_idx]
    gamma_chunk = np.zeros(
        (precip_chunk.shape[0], 3, len(physical_thresholds)), dtype=np.float32
    )

    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)
    pers_thresh = config.get("PERSISTENCE_THRESHOLD", 0.1)

    for i in range(precip_chunk.shape[0]):
        gamma_chunk[i] = compute_gamma_matrix(
            precip_chunk[i], physical_thresholds, pixel_size_km, pers_thresh
        )

    # Write computed matrix back to Zarr
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

    # 1. Compute physical thresholds
    physical_thresholds = compute_climatological_thresholds(zarr_path, quantiles)

    # Save physical thresholds for the loss function/emulator reference
    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    np.save(thresh_path, physical_thresholds)

    # 2. Process all splits
    store = zarr.open(zarr_path, mode="r+")
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)

    for split in ["train", "validation", "test"]:
        if split not in store:
            continue

        print(f"\n--- Extracting Topological Features: {split} ---")
        group = store[split]
        num_samples = group["original_precip"].shape[0]

        # Initialize Zarr array for Gamma targets
        if "gamma_targets" not in group:
            group.create_dataset(
                "gamma_targets",
                shape=(num_samples, 3, len(quantiles)),
                chunks=(chunk_size, 3, len(quantiles)),
                dtype="float32",
            )

        tasks = []
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            tasks.append(
                (start_idx, end_idx, zarr_path, split, physical_thresholds, config)
            )

        max_workers = config.get("MAX_WORKERS", os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_process_chunk, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc=f"Writing {split}"
            ):
                future.result()  # Catch any exceptions raised in workers

    print("\nGamma targets successfully appended to Zarr store.")


if __name__ == "__main__":
    main()
