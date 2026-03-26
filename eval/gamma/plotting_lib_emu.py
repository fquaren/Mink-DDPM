import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy

# --- 1. Single Sample Plotting ---


def _plot_single_gamma_comparison(
    sample_row: pd.Series,
    quantiles: np.ndarray,
    title: str,
    sub_folder: str,
    output_dir: str,
):
    """
    Plots a single sample's comparison, passed as a DataFrame row.

    'sample_row' is a row from metrics_df, containing columns:
    'pred_gamma', 'target_gamma', 'target_image', 'total_loss',
    'geom_loss', 'R2_A', 'R2_P', 'R2_B0', 'R2_B1', 'mean_precip',
    and its name is the 'sample_idx'.
    """

    # --- Extract data from the row ---
    pred_gamma = sample_row["pred_gamma"]
    target_gamma = sample_row["target_gamma"]
    target_image = sample_row["target_image"]
    mean_precip = sample_row["mean_precip"]
    sample_idx = sample_row.name  # Get the original sample index

    # Extract metrics
    loss = sample_row["total_loss"]
    geom_loss = sample_row["geom_loss"]
    R2_A = sample_row["R2_A"]
    R2_P = sample_row["R2_P"]
    R2_B0 = sample_row["R2_B0"]
    R2_B1 = sample_row["R2_B1"]

    # --- Create Plot ---
    gamma_types = ["Area (km²)", "Perimeter (km)", "B0", "B1"]
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)

    # Plot 1: Target Image
    ax_img = fig.add_subplot(gs[0, 0])
    # 1. Mask the zeros (replace 0 with NaN)
    # We use copy to avoid modifying the original dataframe in place (side effects are bad)
    plot_data = target_image.copy()
    plot_data[plot_data <= 0] = np.nan

    # 2. Prepare the Colormap
    # We use 'Blues' for the rain, but copy it to modify the 'bad' value safely
    cmap = copy.copy(plt.get_cmap("Blues"))
    cmap.set_bad(color="lightgrey", alpha=1.0)  # Or 'silver', 'gainsboro'

    # 3. Plot
    # No vmin/vmax hacking needed. NaNs are automatically rendered as 'set_bad' color.
    im = ax_img.imshow(plot_data, cmap=cmap, origin="lower")
    ax_img.set_title(f"Target Image (Mean: {mean_precip:.2f})")
    fig.colorbar(im, ax=ax_img, shrink=0.7, label="Precipitation (mm/hr)")

    # Plot 2-4: Gamma Functions
    for j in range(3):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.plot(quantiles, target_gamma[j], "o-", label="Target", color="royalblue")
        ax.plot(quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon")
        ax.set_title(gamma_types[j])
        ax.set_xlabel("Precip. Threshold (mm/hr)")
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()

    # --- Updated Title ---
    # Create a second line for all the metrics
    metrics_str = (
        f"Total Loss: {loss:.4f} | Geom. Loss: {geom_loss:.4f} | "
        f"R² (A/P/B0/B1): {R2_A:.3f} / {R2_P:.3f} / {R2_B0:.3f} / {R2_B1:.3f}"
    )

    fig.suptitle(
        f"{title} | Sample {sample_idx}\n{metrics_str}",
        fontsize=15,  # Slightly smaller font to fit two lines
        y=1.08,  # Increase y-position to make space for the new line
    )
    # Adjust tight_layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # --- Saving ---
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    # Sanitize title for filename
    safe_title = title.replace(" ", "_").replace("#", "").lower()
    save_path = os.path.join(
        plot_save_dir,
        f"{safe_title}_sample_{sample_idx}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_comparisons(
    metrics_df,
    quantiles,
    output_dir,
    n_samples=10,
):
    """
    Plots best, worst, and average-loss samples, grouped by precipitation.
    This version uses the pandas DataFrame for clean data selection.
    """
    print("\nGenerating plots for best, worst, and average samples (per group)...")

    # Group by precipitation
    grouped = metrics_df.groupby("precip_group")

    # Re-order groups for logical plot output
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]

    for group_name in group_order:
        group_df = grouped.get_group(group_name)
        print(f"\n--- Processing Group: {group_name} (N={len(group_df)}) ---")

        if len(group_df) == 0:
            print("Skipping group, no samples.")
            continue

        current_n = min(n_samples, len(group_df))

        # --- Find Best (nsmallest) ---
        best_samples = group_df.nsmallest(current_n, "total_loss")
        print(f"Plotting {len(best_samples)} best samples...")
        for rank, (idx, row) in enumerate(best_samples.iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Best Sample #{rank+1}", group_name, output_dir
            )

        # --- Find Worst (nlargest) ---
        worst_samples = group_df.nlargest(current_n, "total_loss")
        print(f"Plotting {len(worst_samples)} worst samples...")
        # Rank them from worst to "less worst"
        for rank, (idx, row) in enumerate(worst_samples.iloc[::-1].iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Worst Sample #{rank+1}", group_name, output_dir
            )

        # --- Find Average-Loss ---
        mean_group_loss = group_df["total_loss"].mean()
        if np.isnan(mean_group_loss):
            print("Skipping average loss plots, mean group loss is NaN.")
            continue

        # Find closest to mean
        group_df_copy = group_df.copy()  # Avoid SettingWithCopyWarning
        group_df_copy["dist_to_mean"] = (
            group_df_copy["total_loss"] - mean_group_loss
        ).abs()
        avg_samples = group_df_copy.nsmallest(current_n, "dist_to_mean")

        print(f"Plotting {len(avg_samples)} average-loss samples...")
        for rank, (idx, row) in enumerate(avg_samples.iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Average Loss Sample #{rank+1}", group_name, output_dir
            )


# --- 2. Distribution Plots ---


def plot_metric_distributions(
    metrics_df,
    output_dir,
):
    """
    Generates box plots for the distributions of key evaluation metrics,
    using the metrics_df DataFrame.
    """
    print("\nGenerating metric distribution box plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Evaluation Metric Distributions (Test Set)", fontsize=16, y=1.02)

    # Filter NaNs from the DataFrame columns
    valid_total_losses = metrics_df["total_loss"].dropna()
    valid_geom_losses = metrics_df["geom_loss"].dropna()
    valid_R2_A = metrics_df["R2_A"].dropna()
    valid_R2_P = metrics_df["R2_P"].dropna()
    valid_R2_B0 = metrics_df["R2_B0"].dropna()
    valid_R2_B1 = metrics_df["R2_B1"].dropna()

    medianprops = dict(color="red", linewidth=1.5)

    # Plot 1: Total Loss
    ax1.boxplot(
        valid_total_losses,
        vert=True,
        patch_artist=True,
        labels=["Total Loss"],
        medianprops=medianprops,
    )
    ax1.set_title(f"Total Loss (Config) \nMean: {valid_total_losses.mean():.4f}")
    ax1.set_ylabel("Loss Value")
    ax1.grid(True, linestyle="--", alpha=0.6)
    if not valid_total_losses.empty:
        ax1.set_yscale("log")

    # Plot 2: Geometric Loss
    ax2.boxplot(
        valid_geom_losses,
        vert=True,
        patch_artist=True,
        labels=["Geometric Loss"],
        medianprops=medianprops,
    )
    ax2.set_title(
        f"Geometric (Mahalanobis) Loss \nMean: {valid_geom_losses.mean():.4f}"
    )
    ax2.set_ylabel("Loss Value")
    ax2.grid(True, linestyle="--", alpha=0.6)
    if not valid_geom_losses.empty:
        ax2.set_yscale("log")

    # Plot 3: R^2 Score (per component)
    data_to_plot = [valid_R2_A, valid_R2_P, valid_R2_B0, valid_R2_B1]
    labels = [
        f"Area (Mean: {valid_R2_A.mean():.3f})",
        f"Perim. (Mean: {valid_R2_P.mean():.3f})",
        f"B0 (Mean: {valid_R2_B0.mean():.3f})",
        f"B1 (Mean: {valid_R2_B1.mean():.3f})",
    ]
    ax3.boxplot(
        data_to_plot,
        vert=True,
        patch_artist=True,
        labels=labels,
        medianprops=medianprops,
    )
    ax3.set_title("Per-Sample R² Score by Component")
    ax3.set_ylabel("R² Value")
    ax3.set_ylim(-1.05, 1.05)  # R2 can be negative
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "evaluation_metric_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric distribution plots to: {save_path}")


# --- 3. Grouped Mean/Std Plots ---


def plot_gamma_mean_std_by_quantile(
    metrics_df,
    group_metrics,  # Pass the pre-computed group_metrics summary
    quantiles,
    output_dir,
):
    """
    Plots the mean and std dev of Gamma functions, grouped by precipitation.
    Uses metrics_df to access the raw data for grouping.
    """
    print("\nGenerating plots for mean/std performance by precip group...")

    gamma_types = ["Area (km²)", "Perimeter (km)", "B0", "B1"]
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", "mean_std_groups")
    os.makedirs(plot_save_dir, exist_ok=True)

    grouped = metrics_df.groupby("precip_group")

    # Re-order groups + 'All'
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]
    group_order.append("All")  # Add 'All' group

    for group_name in group_order:
        print(f"--- Plotting Group: {group_name} ---")

        if group_name == "All":
            group_df = metrics_df
            if group_name not in group_metrics.index:
                # Need to handle if 'All' wasn't pre-computed, but it should be
                print("Warning: 'All' group metrics not found in summary.")
                metrics = {k: np.nan for k in group_metrics.columns}
            else:
                metrics = group_metrics.loc["All"]
            n_samples = len(group_df)
        else:
            group_df = grouped.get_group(group_name)
            metrics = group_metrics.loc[group_name]
            n_samples = int(metrics["n_samples"])

        if n_samples == 0:
            print("Skipping group, no samples.")
            continue

        # --- Stack arrays from the DataFrame rows ---
        group_preds = np.stack(group_df["pred_gamma"].values)
        group_targets = np.stack(group_df["target_gamma"].values)

        # --- Calculate statistics for plotting (mean/std) ---
        mean_preds = np.nanmean(group_preds, axis=0)
        std_preds = np.nanstd(group_preds, axis=0)
        mean_targets = np.nanmean(group_targets, axis=0)
        std_targets = np.nanstd(group_targets, axis=0)

        # --- Use pre-computed metrics for title ---
        metric_str = (
            f"Mean Total Loss: {metrics['total_loss']:.4f} | "
            f"Mean R² (A/P/B0/B1): {metrics['R2_A']:.3f} / {metrics['R2_P']:.3f} / {metrics['R2_B0']:.3f} / {metrics['R2_B1']:.3f}"
        )

        fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=False)

        for j in range(4):  # Loop over A, P, B0, B1
            ax = axes[j]

            # Plot Target
            ax.plot(
                quantiles, mean_targets[j], "o-", label="Target Mean", color="royalblue"
            )
            ax.fill_between(
                quantiles,
                mean_targets[j] - std_targets[j],
                mean_targets[j] + std_targets[j],
                color="royalblue",
                alpha=0.2,
                label="Target ±1σ",
            )

            # Plot Prediction
            ax.plot(quantiles, mean_preds[j], "x--", label="Pred. Mean", color="salmon")
            ax.fill_between(
                quantiles,
                mean_preds[j] - std_preds[j],
                mean_preds[j] + std_preds[j],
                color="salmon",
                alpha=0.2,
                label="Pred. ±1σ",
            )

            ax.set_title(gamma_types[j])
            ax.set_xlabel("Precip. Threshold (mm/hr)")
            ax.grid(True, linestyle="--", alpha=0.6)
            if j == 0:
                ax.legend()
                ax.set_ylabel("Value")

            # Use log scale for A and P if their mean is large
            if j < 2 and np.nanmax(mean_targets[j]) > 100:
                ax.set_yscale("log")

        fig.suptitle(
            f"Mean Gamma Function Comparison (±1 Std. Dev.)\n"
            f"Group: {group_name} (N={n_samples})\n"
            f"{metric_str}",
            fontsize=16,
            y=1.08,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = os.path.join(
            plot_save_dir,
            f"mean_std_gamma_group_{group_name.lower()}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved mean/std plots to: {plot_save_dir}")


# --- Training Log Plot ---
def plot_training_log(log_path, output_dir):
    """
    Plots the training history from the training_log.csv file.
    Dynamically adjusts to the columns present in the CSV (Losses, Components, Penalties, Temperature).
    """
    if not os.path.exists(log_path):
        print(
            f"\nWarning: Log file not found at {log_path}. Skipping training history plot."
        )
        return

    print("\nGenerating training history plot...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log file with pandas: {e}. Skipping plot.")
        return

    # --- Determine what to plot based on available columns ---

    # Group 1: Aggregate Losses (Expected to always be present)
    has_total = "train_loss_total" in df.columns and "val_loss_total" in df.columns
    has_main = "train_loss_main" in df.columns and "val_loss_main" in df.columns

    # Group 2: Components (A, P, B0, B1)
    comp_cols = [
        "train_loss_A",
        "train_loss_P",
        "train_loss_B0",
        "train_loss_B1",
        "val_loss_A",
        "val_loss_P",
        "val_loss_B0",
        "val_loss_B1",
    ]
    has_components = all(c in df.columns for c in comp_cols)

    # Group 3: Penalties
    # Find any column starting with "train_penalty" or "val_penalty"
    penalty_cols = [c for c in df.columns if "penalty" in c]
    has_penalties = len(penalty_cols) > 0

    # Group 4: Temperature (Specific to Constrained Model)
    has_temp = "temperature" in df.columns

    # Count active panels
    panels = []
    if has_total or has_main:
        panels.append("losses")
    if has_components:
        panels.append("components")
    if has_penalties:
        panels.append("penalties")
    if has_temp:
        panels.append("temperature")

    if not panels:
        print("Error: No recognized loss columns found in log file.")
        return

    # --- Setup Figure ---
    n_plots = len(panels)
    # Scale height based on number of plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)

    # Ensure axes is iterable even if n_plots is 1
    if n_plots == 1:
        axes = [axes]

    # --- Plotting Loop ---
    for ax, panel_type in zip(axes, panels):

        # 1. Plot Aggregate Losses
        if panel_type == "losses":
            if has_total:
                ax.plot(
                    df["epoch"],
                    df["train_loss_total"],
                    "o-",
                    label="Train Total",
                    color="black",
                    linewidth=1.5,
                    markersize=4,
                )
                ax.plot(
                    df["epoch"],
                    df["val_loss_total"],
                    "o-",
                    label="Val Total",
                    color="red",
                    linewidth=1.5,
                    markersize=4,
                )
            if has_main:
                # If total exists, make main dashed/lighter. If not, make it primary.
                style = "--" if has_total else "-"
                alpha = 0.7 if has_total else 1.0
                ax.plot(
                    df["epoch"],
                    df["train_loss_main"],
                    style,
                    label="Train Main",
                    color="C0",
                    alpha=alpha,
                )
                ax.plot(
                    df["epoch"],
                    df["val_loss_main"],
                    style,
                    label="Val Main",
                    color="C1",
                    alpha=alpha,
                )

            ax.set_ylabel("Loss (Log Scale)")
            ax.set_title("Overall Loss Convergence")
            ax.set_yscale("log")
            ax.legend(loc="upper right", fontsize=9, ncol=2)
            ax.grid(True, linestyle="--", alpha=0.5)

        # 2. Plot Components
        elif panel_type == "components":
            ax.plot(
                df["epoch"],
                df["train_loss_A"],
                "x--",
                label="Train A",
                color="lightblue",
                alpha=0.8,
            )
            ax.plot(
                df["epoch"],
                df["val_loss_A"],
                "x-",
                label="Val A",
                color="blue",
                alpha=0.8,
            )

            ax.plot(
                df["epoch"],
                df["train_loss_P"],
                "x--",
                label="Train P",
                color="lightgreen",
                alpha=0.8,
            )
            ax.plot(
                df["epoch"],
                df["val_loss_P"],
                "x-",
                label="Val P",
                color="green",
                alpha=0.8,
            )

            ax.plot(
                df["epoch"],
                df["train_loss_B0"],
                "x--",
                label="Train B0",
                color="wheat",
                alpha=0.8,
            )
            ax.plot(
                df["epoch"],
                df["val_loss_B0"],
                "x-",
                label="Val B0",
                color="orange",
                alpha=0.8,
            )

            ax.plot(
                df["epoch"],
                df["train_loss_B1"],
                "x--",
                label="Train B1",
                color="lightgray",
                alpha=0.8,
            )
            ax.plot(
                df["epoch"],
                df["val_loss_B1"],
                "x-",
                label="Val B1",
                color="gray",
                alpha=0.8,
            )

            ax.set_ylabel("Component Loss (Log Scale)")
            ax.set_title("Loss Breakdown: Area, Perimeter, B0, B1")
            ax.set_yscale("log")
            ax.legend(loc="upper right", fontsize=9, ncol=4)
            ax.grid(True, linestyle="--", alpha=0.5)

        # 3. Plot Penalties
        elif panel_type == "penalties":
            # Assign colors dynamically
            colors = sns.color_palette("husl", len(penalty_cols))
            for i, col in enumerate(penalty_cols):
                is_train = "train" in col
                style = ":" if is_train else "-"
                width = 1.0 if is_train else 1.5
                label_clean = (
                    col.replace("train_", "T: ")
                    .replace("val_", "V: ")
                    .replace("penalty_", "")
                )

                ax.plot(
                    df["epoch"],
                    df[col],
                    linestyle=style,
                    linewidth=width,
                    label=label_clean,
                    color=colors[i],
                )

            ax.set_ylabel("Penalty Value")
            ax.set_title("Constraint Penalties")
            # Only log scale if values are substantially > 0
            if df[penalty_cols].max().max() > 0:
                ax.set_yscale("log")
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.grid(True, linestyle="--", alpha=0.5)

        # 4. Plot Temperature
        elif panel_type == "temperature":
            ax.plot(
                df["epoch"],
                df["temperature"],
                "p-",
                color="purple",
                label="Softmax Temperature",
            )
            ax.set_ylabel("Temp (Linear)")
            ax.set_title("Annealing Schedule")
            ax.set_ylim(0, 1.05)  # Temp usually 0 to 1
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5)

    # Set X-label on the bottom-most plot
    axes[-1].set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved robust training history plot to: {save_path}")


# --- 4. Per-Feature Matrices Plot ---


def plot_per_feature_matrices(per_feature_metrics, output_dir):
    """
    Generates vertically stacked heatmaps for R2, MSE, and Variance matrices.
    Cells are allowed to be rectangular (wider) to accommodate large value annotations.

    Args:
        per_feature_metrics (dict): Dictionary containing 'r2_matrix',
                                    'mse_matrix', 'var_matrix' DataFrames.
        output_dir (str): Path to save the plots.
    """
    print("\nGenerating per-feature matrix heatmaps (Vertical Stack)...")

    # Extract DataFrames
    r2_df = per_feature_metrics["r2_matrix"]
    mse_df = per_feature_metrics["mse_matrix"]
    var_df = per_feature_metrics["var_matrix"]
    quantiles = per_feature_metrics["quantiles"]

    # Setup figure: 3 Rows, 1 Column.
    # figsize=(16, 24) gives a very wide aspect ratio for each subplot (approx 16:8)
    # giving plenty of horizontal room for scientific notation.
    fig, axes = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle(
        "Per-Feature Metric Matrices (Component vs Quantile)", fontsize=20, y=0.92
    )

    # --- 1. R2 Heatmap ---
    sns.heatmap(
        r2_df,
        ax=axes[0],
        cmap="RdBu",
        center=0.5,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "R² Score"},
        square=False,  # Allow rectangular cells to fit text
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 12},  # Larger font for readability
    )
    axes[0].set_title("R² Score (Blue > 0.5, Red < 0.5)", fontsize=16)
    axes[0].set_xlabel("")  # Remove x-label for top plots to reduce clutter

    # --- 2. MSE Heatmap ---
    sns.heatmap(
        mse_df,
        ax=axes[1],
        cmap="viridis",
        annot=True,
        fmt=".2e",  # Scientific notation for large values
        cbar_kws={"label": "Mean Squared Error"},
        square=False,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 12},
    )
    axes[1].set_title("MSE (Lower is Better)", fontsize=16)
    axes[1].set_xlabel("")

    # --- 3. Variance Heatmap ---
    sns.heatmap(
        var_df,
        ax=axes[2],
        cmap="magma",
        annot=True,
        fmt=".2e",
        cbar_kws={"label": "Target Variance"},
        square=False,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 12},
    )
    axes[2].set_title("Target Variance (Data Spread)", fontsize=16)
    axes[2].set_xlabel("Quantile (mm/hr)", fontsize=14)

    # Adjust formatting for all axes
    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels([f"{q:.1f}" for q in quantiles], rotation=0, fontsize=12)
        ax.tick_params(axis="both", which="both", length=0)  # Clean look

    # Increase vertical spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    save_path = os.path.join(output_dir, "evaluation_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved matrix heatmaps to: {save_path}")


# --- 5. QQ Plots (New Addition) ---


def plot_qq_summary(metrics_df, output_dir):
    """
    Generates Quantile-Quantile (QQ) plots comparing the distribution of
    Predicted vs Target values for A, P, B0, and B1 across the entire dataset.
    """
    print("\nGenerating QQ plots for A, P, B0, and B1 distributions...")

    # Extract flattened arrays from the dataframe series
    # Each row contains an array of shape (4, N_Quantiles).
    # We stack them to get (N_Samples, 4, N_Quantiles), then flatten per component.

    all_preds = np.stack(metrics_df["pred_gamma"].values)  # [N, 4, Q]
    all_targets = np.stack(metrics_df["target_gamma"].values)  # [N, 4, Q]

    components = ["Area (A)", "Perimeter (P)", "B0", "B1"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Percentiles to compute (0% to 100%)
    eval_percentiles = np.linspace(0, 100, 101)

    for i in range(4):
        ax = axes[i]
        comp_name = components[i]

        # Flatten all quantile thresholds into one massive distribution of values
        # This answers: "Does the model predict the full range of Areas correctly?"
        flat_pred = all_preds[:, i, :].flatten()
        flat_target = all_targets[:, i, :].flatten()

        # Remove NaNs if any (e.g., from masked arrays)
        flat_pred = flat_pred[~np.isnan(flat_pred)]
        flat_target = flat_target[~np.isnan(flat_target)]

        # Compute Percentiles
        q_pred = np.percentile(flat_pred, eval_percentiles)
        q_target = np.percentile(flat_target, eval_percentiles)

        # Plot
        ax.plot(
            q_target,
            q_pred,
            "o-",
            color="royalblue",
            markersize=4,
            label="Model vs Target",
        )

        # 1:1 Reference Line
        min_val = min(q_target.min(), q_pred.min())
        max_val = max(q_target.max(), q_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="1:1 Ideal")

        ax.set_title(f"QQ Plot: {comp_name}")
        ax.set_xlabel("Target Quantiles")
        ax.set_ylabel("Predicted Quantiles")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Log scale is often useful for A and P which span orders of magnitude
        if i < 2:  # A and P
            ax.set_xscale("log")
            ax.set_yscale("log")

    fig.suptitle("Global Quantile-Quantile (QQ) Plots", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, "evaluation_qq_plots.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved QQ plots to: {save_path}")


def plot_jacobian_spectrum(jacobian_data, output_dir):
    """
    Plots the distribution of Gradient Norms (Jacobian Spectrum approximation)
    to analyze the isometry/stability of the emulator.

    Args:
        jacobian_data (dict): Dictionary containing lists of norms:
                              {'Area': [...], 'Perimeter': [...], 'B0': [...], 'B1': [...]}
        output_dir (str): Path to save the plot.
    """
    print("\nGenerating Jacobian Spectrum (Gradient Norm) analysis plots...")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Area": "royalblue",
        "Perimeter": "salmon",
        "B0": "lightgray",
        "B1": "darkgray",
    }

    # We plot the Kernel Density Estimate (KDE) and Histogram
    for component, norms in jacobian_data.items():
        if len(norms) == 0:
            continue

        # Convert to numpy and handle potential NaNs
        norms_np = np.array(norms)
        norms_np = norms_np[np.isfinite(norms_np)]

        if len(norms_np) == 0:
            continue

        sns.histplot(
            norms_np,
            ax=ax,
            label=f"{component} Sensitivity",
            color=colors.get(component, "gray"),
            kde=True,
            log_scale=True,  # Critical: Gradients often span orders of magnitude
            element="step",
            fill=False,
            stat="density",
            linewidth=2,
        )

    ax.set_title(
        "Jacobian Spectrum Analysis (Input Sensitivity)\nIsometry Check: Ideally narrow, non-zero distribution",
        fontsize=14,
    )
    ax.set_xlabel("Gradient Norm || d_Output / d_Input || (Log Scale)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend()

    # Interpretation text
    text_str = (
        "Left (< 1e-4): Vanishing Gradients (Insensitive)\n"
        "Right (> 1e2): Exploding/Shattered (Unstable)\n"
        "Center: Isometric/Stable Region"
    )
    plt.text(
        0.02,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "jacobian_spectrum.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved Jacobian spectrum plot to: {save_path}")


# --- 6. New Validation Plots (Isometry & Physics) ---


def plot_isoperimetric_check(all_preds, output_dir):
    """
    Validates physical consistency: P >= sqrt(4 * pi * A).
    """
    print("\nGenerating Isoperimetric Consistency Check plot...")

    # Flatten to treat every quantile of every sample as a data point
    # Shape: [N, 4, Q] -> Flattened Area and Perimeter
    # We only care about Area > epsilon to avoid log(0) issues or division issues

    flat_A = all_preds[:, 0, :].flatten()
    flat_P = all_preds[:, 1, :].flatten()

    # Filter out tiny values to focus on the shape manifold
    mask = flat_A > 1e-2
    A_valid = flat_A[mask]
    P_valid = flat_P[mask]

    if len(A_valid) == 0:
        print("Warning: No valid Area samples > 0.01 found for Isoperimetric plot.")
        return

    # Theoretical minimum perimeter (Circle)
    P_min = np.sqrt(4 * np.pi * A_valid)

    # Calculate Violations
    # Allow small numerical epsilon
    violation_mask = P_valid < (P_min - 1e-4)
    violation_rate = np.mean(violation_mask) * 100

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter Plot
    # We plot sqrt(A) vs P so the relationship is linear: P = k * sqrt(A)
    x_val = np.sqrt(A_valid)
    y_val = P_valid

    # Downsample for plotting speed if needed
    if len(x_val) > 10000:
        idx = np.random.choice(len(x_val), 10000, replace=False)
        x_val = x_val[idx]
        y_val = y_val[idx]
        violation_mask_plot = violation_mask[idx]
    else:
        violation_mask_plot = violation_mask

    # Plot valid points
    ax.scatter(
        x_val[~violation_mask_plot],
        y_val[~violation_mask_plot],
        alpha=0.3,
        s=5,
        c="dodgerblue",
        label="Physically Valid",
    )

    # Plot violating points
    ax.scatter(
        x_val[violation_mask_plot],
        y_val[violation_mask_plot],
        alpha=0.5,
        s=10,
        c="crimson",
        label="Violation",
    )

    # Plot Theoretical Line
    max_sqrt_A = x_val.max()
    line_x = np.linspace(0, max_sqrt_A, 100)
    line_y = np.sqrt(4 * np.pi) * line_x
    ax.plot(
        line_x,
        line_y,
        "k--",
        linewidth=2,
        label=r"Theoretical Limit ($P = \sqrt{4\pi A}$)",
    )

    ax.set_title(
        f"Isoperimetric Consistency Check\nViolation Rate: {violation_rate:.2f}%"
    )
    ax.set_xlabel(r"$\sqrt{\text{Area}}$ (Linear Scale)")
    ax.set_ylabel("Perimeter")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "validation_isoperimetric_check.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved Isoperimetric plot to: {save_path}")


def plot_dry_input_error(all_preds, all_inputs, output_dir, threshold=1e-5):
    """
    Checks the 'Zero Mode' stability.
    Plots the distribution of predicted Area for inputs that are effectively zero.
    """
    print("\nGenerating Dry Input Error Analysis...")

    # Identify Dry Samples: sum(input) < threshold
    # all_inputs shape: [N, H, W] or [N, 1, H, W]
    input_sums = np.sum(all_inputs, axis=(-2, -1)).flatten()
    dry_indices = np.where(input_sums < threshold)[0]

    if len(dry_indices) == 0:
        print("No dry samples found in test set. Skipping Dry Error plot.")
        return

    # Extract Predicted Area for these dry samples
    # We take the MAX Area predicted across quantiles for each dry sample
    # If the max is 0, then the whole vector is 0 (good).
    dry_preds_A = all_preds[dry_indices, 0, :]  # [N_dry, Q]
    max_dry_A = np.max(dry_preds_A, axis=1)  # [N_dry]

    mean_error = np.mean(max_dry_A)
    max_error = np.max(max_dry_A)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(max_dry_A, bins=50, kde=False, color="darkorange", ax=ax)

    ax.set_title(
        f"Dry Input Response Analysis (N={len(dry_indices)})\n"
        f"Mean Ghost Area: {mean_error:.4f} km² | Max Ghost Area: {max_error:.4f} km²"
    )
    ax.set_xlabel("Predicted Max Area (km²) for Empty Input")
    ax.set_ylabel("Count")
    ax.set_yscale("log")  # Log scale to see the tiny errors vs perfect zeros
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "validation_dry_input_error.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved Dry Error plot to: {save_path}")


def plot_saliency_maps(saliency_data, output_dir):
    """
    Plots input images alongside their gradient maps (Saliency).
    saliency_data: list of tuples (input_img, grad_img, title)
    """
    print("\nGenerating Saliency Map visualizations...")

    save_dir = os.path.join(output_dir, "evaluation_plots", "saliency")
    os.makedirs(save_dir, exist_ok=True)

    for i, (inp, grad, title) in enumerate(saliency_data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Input
        im0 = axes[0].imshow(inp, cmap="Blues", origin="lower")
        axes[0].set_title(f"Input: {title}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot Gradient
        # We use a divergent map centered at 0 to show push/pull
        limit = np.max(np.abs(grad)) + 1e-9
        im1 = axes[1].imshow(
            grad, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit
        )
        axes[1].set_title("Gradient w.r.t Input (Sensitivity)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        safe_title = title.replace(" ", "_").lower()
        plt.savefig(os.path.join(save_dir, f"saliency_{i}_{safe_title}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved {len(saliency_data)} saliency maps to {save_dir}")
