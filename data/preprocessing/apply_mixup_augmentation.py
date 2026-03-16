import os
import yaml
import zarr
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from compute_gamma_targets import compute_gamma_matrix

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def worker_process_mixup_chunk(args):
    start_idx, end_idx, zarr_path, physical_thresholds, config = args

    store = zarr.open(zarr_path, mode="r+")
    train_group = store["train"]

    real_chunk = train_group["original_precip"][start_idx:end_idx]
    interp_chunk = train_group["interpolated_precip"][start_idx:end_idx]

    N, H, W = real_chunk.shape
    mixed_chunk_out = np.zeros((N, H, W), dtype=np.float32)
    gamma_chunk = np.zeros((N, 3, len(physical_thresholds)), dtype=np.float32)

    mixup_alpha = config.get("MIXUP_ALPHA", 0.2)
    noise_std = config.get("NOISE_STD", 0.05)
    pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)
    pers_thresh = config.get("PERSISTENCE_THRESHOLD", 0.05)

    rng = np.random.default_rng()

    for i in range(N):
        real_patch = real_chunk[i]
        interp_patch = interp_chunk[i]

        noise = rng.normal(0, noise_std, interp_patch.shape)
        interp_noisy = np.clip(interp_patch + noise, 0.0, None)

        lam = rng.beta(mixup_alpha, mixup_alpha)
        mixed_patch = lam * real_patch + (1 - lam) * interp_noisy

        gamma_chunk[i] = compute_gamma_matrix(
            mixed_patch, physical_thresholds, pixel_size_km, pers_thresh
        )

        mixed_chunk_out[i] = mixed_patch

    train_group["mixup_precip"][start_idx:end_idx] = mixed_chunk_out
    train_group["mixup_gamma_targets"][start_idx:end_idx] = gamma_chunk

    return f"Processed mixup indices {start_idx} to {end_idx}"


def main():
    parser = argparse.ArgumentParser(
        description="Apply Mixup Augmentation to Zarr store."
    )
    parser.add_argument("config", type=str, help="Path to the config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )

    if not os.path.exists(thresh_path):
        raise FileNotFoundError(
            "Physical thresholds missing. Run previous stages first."
        )

    physical_thresholds = np.load(thresh_path)

    store = zarr.open(zarr_path, mode="r+")
    if "train" not in store:
        raise ValueError("Train group missing in Zarr store.")

    train_group = store["train"]
    num_samples = train_group["original_precip"].shape[0]
    _, H, W = train_group["original_precip"].shape
    num_quantiles = len(physical_thresholds)
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)

    if "mixup_precip" not in train_group:
        train_group.create_dataset(
            "mixup_precip",
            shape=(num_samples, H, W),
            chunks=(chunk_size, H, W),
            dtype="float32",
        )
    if "mixup_gamma_targets" not in train_group:
        train_group.create_dataset(
            "mixup_gamma_targets",
            shape=(num_samples, 3, num_quantiles),
            chunks=(chunk_size, 3, num_quantiles),
            dtype="float32",
        )

    tasks = []
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        # scaler_val dependency removed
        tasks.append((start_idx, end_idx, zarr_path, physical_thresholds, config))

    max_workers = config.get("MAX_WORKERS", os.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_process_mixup_chunk, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Applying Mixup"
        ):
            future.result()


if __name__ == "__main__":
    main()
