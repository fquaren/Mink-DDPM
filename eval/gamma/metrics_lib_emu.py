import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm


def _calculate_per_sample_metrics(preds, targets):
    """
    Calculates R2, MSE, and Target Variance for each sample's vector (A, P, B0).
    """
    n_samples = len(preds)
    components = ["A", "P", "B0"]

    # Initialize storage with lowercase keys for internal consistency
    metrics = {}
    for m in ["R2", "MSE", "Var"]:
        for c in components:
            metrics[f"{m}_{c}"] = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="Calculating Sample Metrics"):
        for j, comp in enumerate(components):
            pred_sample = preds[i, j, :]
            target_sample = targets[i, j, :]

            mask = np.isfinite(pred_sample) & np.isfinite(target_sample)

            if np.sum(mask) < 2:
                metrics[f"R2_{comp}"][i] = np.nan
                metrics[f"MSE_{comp}"][i] = np.nan
                metrics[f"Var_{comp}"][i] = np.nan
                continue

            t_clean = target_sample[mask]
            p_clean = pred_sample[mask]

            variance = np.var(t_clean)
            mse = mean_squared_error(t_clean, p_clean)

            metrics[f"Var_{comp}"][i] = variance
            metrics[f"MSE_{comp}"][i] = mse

            if variance < 1e-9:
                metrics[f"R2_{comp}"][i] = np.nan
            else:
                metrics[f"R2_{comp}"][i] = r2_score(t_clean, p_clean)

    return metrics


def _get_precipitation_groups(mean_precip_col):
    """Applies the robust precipitation grouping strategy."""
    groups = pd.Series("Zero", index=mean_precip_col.index, dtype=object)
    non_zero_mask = mean_precip_col > 0

    if non_zero_mask.any():
        non_zero_means = mean_precip_col[non_zero_mask]
        p33 = np.quantile(non_zero_means, 0.33)
        p67 = np.quantile(non_zero_means, 0.67)
        bins = [-np.inf, p33, p67, np.inf]
        labels = ["Low", "Mid", "High"]
        binned_data = pd.cut(non_zero_means, bins=bins, labels=labels)
        groups.loc[binned_data.index] = binned_data
    return groups


def create_metrics_dataframe(
    all_preds_phys,
    all_targets_phys,
    all_original_images,
    all_total_losses,
    all_geom_losses,
):
    print("\nCreating comprehensive metrics DataFrame (with MSE/Var)...")
    sample_metrics = _calculate_per_sample_metrics(all_preds_phys, all_targets_phys)
    mean_precip = np.mean(all_original_images, axis=(1, 2))

    # Unpack the N x 3 array into three 1D arrays of shape (N,)
    data = {
        "total_loss": all_total_losses,
        "geom_loss_A": all_geom_losses[:, 0],
        "geom_loss_P": all_geom_losses[:, 1],
        "geom_loss_B0": all_geom_losses[:, 2],
        "mean_precip": mean_precip,
    }
    data.update(sample_metrics)

    df = pd.DataFrame(data)
    df["precip_group"] = _get_precipitation_groups(df["mean_precip"])

    # Restore the omitted object columns
    df["target_image"] = [img for img in all_original_images]
    df["pred_gamma"] = [gamma for gamma in all_preds_phys]
    df["target_gamma"] = [gamma for gamma in all_targets_phys]

    print("DataFrame created.")
    return df


def calculate_grouped_metrics(metrics_df):
    """
    Calculates the MEAN of metrics grouped by precip_group.
    """
    print("\nCalculating sample-wise metrics by precipitation group...")

    # Explicitly list the newly unpacked geometric loss components
    ordered_cols = [
        "total_loss",
        "geom_loss_A",
        "geom_loss_P",
        "geom_loss_B0",
    ]

    for comp in ["A", "P", "B0"]:
        ordered_cols.extend([f"R2_{comp}", f"MSE_{comp}", f"Var_{comp}"])

    # Group and Mean
    grouped = metrics_df.groupby("precip_group")
    group_means = grouped[ordered_cols].mean()
    group_counts = grouped.size().to_frame("n_samples")

    group_metrics = pd.concat([group_means, group_counts], axis=1)

    # Add 'All' group
    all_metrics = metrics_df[ordered_cols].mean().to_frame("All").T
    all_metrics["n_samples"] = len(metrics_df)

    final_metrics = pd.concat([group_metrics, all_metrics])

    group_order = ["Zero", "Low", "Mid", "High", "All"]
    final_metrics = final_metrics.reindex(group_order).dropna(how="all")

    return final_metrics


def calculate_global_group_metrics(metrics_df, all_preds, all_targets):
    """
    Calculates global metrics on concatenated vectors.
    """
    print("\nCalculating global component-wise metrics...")

    groups = ["Zero", "Low", "Mid", "High", "All"]
    components = ["A", "P", "B0"]

    results = {
        metric: {g: {c: np.nan for c in components} for g in groups}
        for metric in ["R2", "MSE", "Var"]
    }

    for grp in groups:
        if grp == "All":
            indices = metrics_df.index.to_numpy()
        else:
            indices = metrics_df[metrics_df["precip_group"] == grp].index.to_numpy()

        if len(indices) < 2:
            continue

        preds_sub = all_preds[indices]
        targets_sub = all_targets[indices]

        for i, comp in enumerate(components):
            p_flat = preds_sub[:, i, :].flatten()
            t_flat = targets_sub[:, i, :].flatten()

            mask = np.isfinite(p_flat) & np.isfinite(t_flat)
            p_clean = p_flat[mask]
            t_clean = t_flat[mask]

            if len(t_clean) < 2:
                continue

            var_val = np.var(t_clean)
            mse_val = mean_squared_error(t_clean, p_clean)

            results["Var"][grp][comp] = var_val
            results["MSE"][grp][comp] = mse_val

            if var_val < 1e-9:
                results["R2"][grp][comp] = np.nan
            else:
                results["R2"][grp][comp] = r2_score(t_clean, p_clean)

    # Construct DataFrame with Multi-level columns or flattened
    final_rows = []
    for grp in groups:
        row = {}
        for comp in components:
            row[f"R2_{comp}"] = results["R2"][grp][comp]
            row[f"MSE_{comp}"] = results["MSE"][grp][comp]
            row[f"Var_{comp}"] = results["Var"][grp][comp]
        final_rows.append(pd.Series(row, name=grp))

    global_metrics = pd.concat(final_rows, axis=1).T

    # Reorder columns to match sample-wise
    ordered_cols = []
    for comp in components:
        ordered_cols.extend([f"R2_{comp}", f"MSE_{comp}", f"Var_{comp}"])

    global_metrics = global_metrics[ordered_cols]

    return global_metrics


def calculate_per_feature_metrics(all_preds_phys, all_targets_phys, quantiles):
    """
    Calculates R^2, MSE, and Target Variance for each individual feature
    across all samples.
    """
    print("\nCalculating per-feature metrics (R^2, MSE, Var)...")

    n_samples, n_components, n_quantiles = all_preds_phys.shape

    # Flatten from (N, 3, Q) to (N, 3*Q) for easier filtering
    preds_flat = all_preds_phys.reshape(n_samples, -1)
    targets_flat = all_targets_phys.reshape(n_samples, -1)

    # Filter samples with any NaNs/Infs in targets or preds
    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)

    n_valid = np.sum(mask)
    if n_valid < n_samples:
        print(f"Warning: Filtering {n_samples - n_valid} samples with NaNs/Infs.")

    # Define DataFrame structure (3 components)
    idx = pd.Index(["Area", "Perimeter", "B0"], name="Component")
    cols = pd.Index(quantiles, name="Quantile (mm/hr)")

    if n_valid < 2:
        print("CRITICAL: Less than 2 valid samples. Returning NaN metrics.")
        r2_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        mse_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        var_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        mean_by_component = pd.DataFrame(
            {
                "Avg_R2": np.nan,
                "Avg_MSE": np.nan,
                "Avg_Target_Var": np.nan,
            },
            index=idx,
        )

        return {
            "r2_matrix": r2_matrix,
            "mse_matrix": mse_matrix,
            "var_matrix": var_matrix,
            "mean_by_component": mean_by_component,
            "quantiles": quantiles,
        }

    # --- Calculate Metrics ---
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_raw = r2_score(
            targets_flat[mask], preds_flat[mask], multioutput="raw_values"
        )

    mse_raw = mean_squared_error(
        targets_flat[mask], preds_flat[mask], multioutput="raw_values"
    )

    var_raw = np.var(targets_flat[mask], axis=0)

    # --- Format as DataFrames ---
    # Reshape back to (3, n_quantiles)
    r2_matrix = pd.DataFrame(r2_raw.reshape(3, n_quantiles), index=idx, columns=cols)
    mse_matrix = pd.DataFrame(mse_raw.reshape(3, n_quantiles), index=idx, columns=cols)
    var_matrix = pd.DataFrame(var_raw.reshape(3, n_quantiles), index=idx, columns=cols)

    mean_by_component = pd.DataFrame(
        {
            "Avg_R2": r2_matrix.mean(axis=1),
            "Avg_MSE": mse_matrix.mean(axis=1),
            "Avg_Target_Var": var_matrix.mean(axis=1),
        }
    )

    print("Mean metrics by component (averaged over quantiles):")
    print(mean_by_component.to_string(float_format="%.4e"))

    return {
        "r2_matrix": r2_matrix,
        "mse_matrix": mse_matrix,
        "var_matrix": var_matrix,
        "mean_by_component": mean_by_component,
        "quantiles": quantiles,
    }
