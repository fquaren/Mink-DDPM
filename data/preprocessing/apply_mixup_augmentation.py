import os
import yaml
import zarr
import numpy as np
import argparse
import jax
import jax.numpy as jnp
import logging
import traceback
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from compute_gamma_targets import compute_gamma_matrix

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- Config & Path Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)


# Configure local worker logging
def setup_worker_logging():
    logger = logging.getLogger("worker_logger")
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("mixup_debug.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


@jax.jit
def apply_mixup_jax(real_chunk, interp_chunk, mixup_alpha, noise_std, prng_key):
    key_noise, key_beta = jax.random.split(prng_key)
    noise = noise_std * jax.random.normal(key_noise, interp_chunk.shape)
    interp_noisy = jnp.clip(interp_chunk + noise, 0.0, None)
    lam = jax.random.beta(
        key_beta, mixup_alpha, mixup_alpha, shape=(real_chunk.shape[0], 1, 1)
    )
    return lam * real_chunk + (1 - lam) * interp_noisy


def worker_process_mixup_chunk(args):
    start_idx, end_idx, zarr_path, physical_thresholds, config, thresh_b0, thresh_b1 = (
        args
    )
    logger = setup_worker_logging()

    try:
        logger.info(f"Started processing indices {start_idx} to {end_idx}.")

        store = zarr.open(zarr_path, mode="r+")
        train_group = store["train"]

        logger.debug(f"[{start_idx}-{end_idx}] Loading Zarr data into memory.")
        real_chunk_np = train_group["original_precip"][start_idx:end_idx]
        interp_chunk_np = train_group["interpolated_precip"][start_idx:end_idx]

        N = real_chunk_np.shape[0]
        gamma_chunk = np.zeros((N, 4, len(physical_thresholds)), dtype=np.float32)

        mixup_alpha = config.get("MIXUP_ALPHA", 0.2)
        noise_std = config.get("NOISE_STD", 0.05)
        pixel_size_km = config.get("PIXEL_SIZE_KM", 2.0)

        logger.debug(f"[{start_idx}-{end_idx}] Dispatching to JAX backend.")
        key = jax.random.PRNGKey(start_idx)
        real_jax = jnp.asarray(real_chunk_np)
        interp_jax = jnp.asarray(interp_chunk_np)

        mixed_chunk_jax = apply_mixup_jax(
            real_jax, interp_jax, mixup_alpha, noise_std, key
        )
        mixed_chunk_np = np.asarray(mixed_chunk_jax)

        logger.debug(
            f"[{start_idx}-{end_idx}] JAX complete. Starting CPU TDA computation."
        )
        for i in range(N):
            gamma_chunk[i] = compute_gamma_matrix(
                mixed_chunk_np[i],
                physical_thresholds,
                pixel_size_km,
                thresh_b0,
                thresh_b1,
            )

        logger.debug(f"[{start_idx}-{end_idx}] TDA complete. Writing results to Zarr.")
        train_group["mixup_precip"][start_idx:end_idx] = mixed_chunk_np
        train_group["mixup_gamma_targets"][start_idx:end_idx] = gamma_chunk

        logger.info(f"Successfully finished indices {start_idx} to {end_idx}.")
        return {"status": "success", "msg": f"Processed {start_idx} to {end_idx}"}

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(
            f"Failed at indices {start_idx} to {end_idx}. Exception:\n{error_trace}"
        )
        return {"status": "error", "msg": error_trace}


def main():
    # Initialize main logger to clear previous run data
    with open("mixup_debug.log", "w") as f:
        f.write("--- Starting Mixup Pipeline ---\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy"
    )
    emp_thresh_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "persistence_thresholds.yaml"
    )

    physical_thresholds = np.load(thresh_path)

    if os.path.exists(emp_thresh_path):
        with open(emp_thresh_path, "r") as f:
            emp_thresh = yaml.safe_load(f)
        thresh_b0 = emp_thresh.get("PERSISTENCE_THRESHOLD_B0", 0.05)
        thresh_b1 = emp_thresh.get("PERSISTENCE_THRESHOLD_B1", 0.05)
    else:
        thresh_b0 = config.get("PERSISTENCE_THRESHOLD", 0.05)
        thresh_b1 = thresh_b0

    store = zarr.open(zarr_path, mode="r+")
    train_group = store["train"]
    num_samples = train_group["original_precip"].shape[0]
    _, H, W = train_group["original_precip"].shape
    num_quantiles = len(physical_thresholds)
    chunk_size = config.get("WORKER_CHUNK_SIZE", 500)

    # --- FIX: Purge legacy arrays before reallocation ---
    if "mixup_precip" in train_group:
        del train_group["mixup_precip"]

    train_group.create_dataset(
        "mixup_precip",
        shape=(num_samples, H, W),
        chunks=(chunk_size, H, W),
        dtype="float32",
    )

    if "mixup_gamma_targets" in train_group:
        del train_group["mixup_gamma_targets"]

    train_group.create_dataset(
        "mixup_gamma_targets",
        shape=(num_samples, 4, num_quantiles),
        chunks=(chunk_size, 4, num_quantiles),
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
                physical_thresholds,
                config,
                thresh_b0,
                thresh_b1,
            )
        )

    max_workers = config.get("MAX_WORKERS", max(1, os.cpu_count() // 4))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_process_mixup_chunk, task) for task in tasks]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Applying JAX Mixup"
        ):
            res = future.result()
            if res["status"] == "error":
                print(f"\nWorker Error:\n{res['msg']}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
