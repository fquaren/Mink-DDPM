import os
import yaml
import zarr
import numpy as np
import argparse
import gudhi as gd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"


def process_single_image(image_np):
    """
    Computes finite persistence values for a single precipitation field.
    Returns two lists: finite persistences for B0 and B1.
    """
    # Invert the field so local maxima become minima for sub-level set filtration
    neg_prec_field = -np.nan_to_num(image_np, nan=-1.0).astype(np.float64)

    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    persistence_pairs = cubical_complex.persistence()

    p_b0, p_b1 = [], []

    for dim, (b, d) in persistence_pairs:
        if not np.isfinite(b) or not np.isfinite(d):
            continue  # Skip essential/infinite features (true signal)

        persistence = abs(b - d)
        if persistence <= 1e-6:
            continue  # Skip zero-persistence computational artifacts

        if dim == 0:
            p_b0.append(persistence)
        elif dim == 1:
            p_b1.append(persistence)

    return p_b0, p_b1


def compute_empirical_thresholds(config_path, num_samples=2000, target_percentile=99.0):
    print("--- Computing Empirical Persistence Thresholds ---")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    store = zarr.open(zarr_path, mode="r")

    if "train" not in store:
        raise ValueError("Train split not found in Zarr store.")

    train_data = store["train/original_precip"]
    total_images = train_data.shape[0]

    # Randomly sample images to build the noise distribution
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(
        total_images, size=min(num_samples, total_images), replace=False
    )
    sample_indices = np.sort(sample_indices)

    print(f"Extracting {len(sample_indices)} images from train set...")
    sampled_images = train_data.oindex[sample_indices]

    all_p_b0 = []
    all_p_b1 = []

    max_workers = config.get("MAX_WORKERS", os.cpu_count() // 2)

    print("Extracting topological pairs (this may take a moment)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img) for img in sampled_images]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="TDA Processing"
        ):
            b0_vals, b1_vals = future.result()
            all_p_b0.extend(b0_vals)
            all_p_b1.extend(b1_vals)

    all_p_b0 = np.array(all_p_b0)
    all_p_b1 = np.array(all_p_b1)

    print("\n--- Topological Noise Statistics ---")
    print(f"Total finite B0 pairs analyzed: {len(all_p_b0)}")
    print(f"Total finite B1 pairs analyzed: {len(all_p_b1)}")

    thresh_b0 = np.percentile(all_p_b0, target_percentile) if len(all_p_b0) > 0 else 0.0
    thresh_b1 = np.percentile(all_p_b1, target_percentile) if len(all_p_b1) > 0 else 0.0
    unified_thresh = max(thresh_b0, thresh_b1)

    print(f"\n{target_percentile}th Percentile Thresholds:")
    print(f"  B0 (Components) Noise Floor : {thresh_b0:.4f}")
    print(f"  B1 (Holes) Noise Floor      : {thresh_b1:.4f}")
    print(f"  Conservative Unified Max    : {unified_thresh:.4f}")

    # Save parameters for downstream pipeline use
    output_dict = {
        "PERSISTENCE_THRESHOLD_B0": float(thresh_b0),
        "PERSISTENCE_THRESHOLD_B1": float(thresh_b1),
        "PERSISTENCE_THRESHOLD_UNIFIED": float(unified_thresh),
    }

    out_file = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "persistence_thresholds.yaml"
    )
    with open(out_file, "w") as f:
        yaml.dump(output_dict, f)

    print(f"\nThresholds saved to: {out_file}")
    print(
        "ACTION REQUIRED: Update your config.yaml or compute_gamma_matrix to use these values."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument(
        "--samples", type=int, default=2000, help="Number of images to sample"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile to define noise floor",
    )
    args = parser.parse_args()

    compute_empirical_thresholds(args.config, args.samples, args.percentile)
