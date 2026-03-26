import numpy as np
from scipy.ndimage import label, uniform_filter, center_of_mass
import pandas as pd


# --- Analytical FSS (Fractions Skill Score) Function ---
def compute_fss(pred_np, target_np, window_size, threshold):
    """Calculates the Fractions Skill Score (FSS) for a single window and threshold."""
    pred_mask = (pred_np >= threshold).astype(float)
    target_mask = (target_np >= threshold).astype(float)

    # Calculate fractions using a uniform filter (mean over the window)
    pred_frac = uniform_filter(pred_mask, size=window_size)
    target_frac = uniform_filter(target_mask, size=window_size)

    # Calculate MSE of the fractions
    mse = np.mean((pred_frac - target_frac) ** 2)
    # Calculate the reference variance
    ref_var = np.mean(pred_frac**2) + np.mean(target_frac**2)

    if ref_var == 0:
        return 1.0  # Perfect score if both fields are zero

    fss = 1.0 - (mse / ref_var)
    return fss


# --- Analytical SAL (Structure, Amplitude, Location) Function ---
def compute_sal(pred_np, target_np, threshold, pixel_area_km2=4.0):
    """Calculates the Wernli et al. (2008) SAL metric."""

    # --- Amplitude (A) ---
    mean_pred = np.mean(pred_np)
    mean_target = np.mean(target_np)
    A_denom = 0.5 * (mean_pred + mean_target)
    A = (mean_pred - mean_target) / (A_denom + 1e-6)

    # --- Object Identification ---
    pred_mask = pred_np >= threshold
    target_mask = target_np >= threshold
    pred_objects, n_pred = label(pred_mask)
    target_objects, n_target = label(target_mask)

    if n_pred == 0 or n_target == 0:
        # SAL is undefined or partially undefined if one field has no objects
        return np.nan, A, np.nan

    # --- Location (L) ---
    com_pred_total = center_of_mass(pred_np)
    com_target_total = center_of_mass(target_np)

    # L1: Normalized distance between total centers of mass
    d_max = np.sqrt(pred_np.shape[0] ** 2 + pred_np.shape[1] ** 2)
    L1 = (
        np.sqrt(
            (com_pred_total[0] - com_target_total[0]) ** 2
            + (com_pred_total[1] - com_target_total[1]) ** 2
        )
        / d_max
    )

    # L2: Normalized distance between object-weighted CoMs
    def get_weighted_com_dist(field, objects, n_obj, com_total):
        r_num = 0
        r_den = 0
        for i in range(1, n_obj + 1):
            mask = objects == i
            obj_mass = np.sum(field[mask])
            obj_com = center_of_mass(field, labels=objects, index=i)
            dist = np.sqrt(
                (obj_com[0] - com_total[0]) ** 2 + (obj_com[1] - com_total[1]) ** 2
            )
            r_num += obj_mass * dist
            r_den += obj_mass
        return r_num / (r_den + 1e-6)

    r_pred = get_weighted_com_dist(pred_np, pred_objects, n_pred, com_pred_total)
    r_target = get_weighted_com_dist(
        target_np, target_objects, n_target, com_target_total
    )
    L2 = 2 * (np.abs(r_pred - r_target)) / d_max
    L = L1 + L2

    # --- Structure (S) ---
    def get_scaled_volume(field, objects, n_obj):
        total_mass = np.sum(field[objects > 0])
        v_scaled = 0
        for i in range(1, n_obj + 1):
            mask = objects == i
            obj_mass = np.sum(field[mask])
            v_scaled += obj_mass * (obj_mass / (np.max(field[mask]) + 1e-6))
        return v_scaled / (total_mass + 1e-6)

    V_pred = get_scaled_volume(pred_np, pred_objects, n_pred)
    V_target = get_scaled_volume(target_np, target_objects, n_target)
    S_denom = 0.5 * (V_pred + V_target)
    S = (V_pred - V_target) / (S_denom + 1e-6)

    return S, A, L


def compute_batch_fss(pred_batch, target_batch, window_size, threshold):
    """
    Wraps the 2D FSS computation for [B, H, W] arrays.
    Returns an array of shape [B].
    """
    batch_size = pred_batch.shape[0]
    fss_scores = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        fss_scores[i] = compute_fss(
            pred_batch[i], target_batch[i], window_size, threshold
        )

    return fss_scores


def compute_batch_sal(pred_batch, target_batch, threshold, pixel_area_km2=4.0):
    """
    Wraps the 2D SAL computation for [B, H, W] arrays.
    Returns three arrays of shape [B] for S, A, and L.
    """
    batch_size = pred_batch.shape[0]
    S_scores = np.zeros(batch_size, dtype=np.float32)
    A_scores = np.zeros(batch_size, dtype=np.float32)
    L_scores = np.zeros(batch_size, dtype=np.float32)

    for i in range(batch_size):
        S, A, L = compute_sal(pred_batch[i], target_batch[i], threshold, pixel_area_km2)
        S_scores[i] = S
        A_scores[i] = A
        L_scores[i] = L

    return S_scores, A_scores, L_scores


# --- Data Aggregation and Analysis Functions ---
def create_metrics_dataframe(
    preds_gamma,
    targets_gamma,
    inputs_phys,
    targets_phys,
    preds_phys,
    dems,
    total_losses,
    mse_losses,
    surrogate_losses,
    quantile_levels,
    pixel_size_km,
):
    """Compiles 1D metrics and stores multidimensional matrices safely."""
    df = pd.DataFrame(
        {
            "Total_Loss": total_losses,
            "MSE_Loss": mse_losses,
            "Surrogate_Loss": surrogate_losses,
        }
    )

    # Store multi-dimensional gamma arrays as lists within the series
    # to bypass pandas dimensionality constraints.
    df["pred_gamma"] = list(preds_gamma)
    df["target_gamma"] = list(targets_gamma)

    return df


def calculate_grouped_metrics(metrics_df):
    """Extracts mean statistics for scalar variables across the dataset."""
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    return metrics_df[numeric_cols].mean().to_dict()


def calculate_per_feature_gamma_metrics(metrics_df, quantile_levels):
    """Computes MAE and MSE natively across the gamma quantile matrices."""
    preds = np.stack(metrics_df["pred_gamma"].values)
    targets = np.stack(metrics_df["target_gamma"].values)

    return {
        "gamma_mae": np.mean(np.abs(preds - targets), axis=0),
        "gamma_mse": np.mean((preds - targets) ** 2, axis=0),
    }


def compute_isoperimetric_violation(pred_log):
    """
    Computes the physical isoperimetric violation rate: P >= sqrt(4 * pi * A).
    Expects predictions in log space [N, 3, Q], converts to physical space.
    """
    # Transform back to physical units
    pred_phys = np.expm1(pred_log)

    flat_A = pred_phys[:, 0, :].flatten()
    flat_P = pred_phys[:, 1, :].flatten()

    mask = flat_A > 1e-2
    A_valid = flat_A[mask]
    P_valid = flat_P[mask]

    if len(A_valid) == 0:
        return 0.0

    P_min = np.sqrt(4 * np.pi * A_valid)
    violation_mask = P_valid < (P_min - 1e-4)

    return np.mean(violation_mask) * 100.0
