import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy


# --- 1. Single Sample Plotting (Comprehensive) ---
def _plot_comprehensive_sample(
    sample_row: pd.Series,
    quantiles: np.ndarray,
    title: str,
    sub_folder: str,
    output_dir: str,
):
    """
    Plots a 2-row layout:
    Row 1: DEM | Input (LR) | Prediction (SR) | Target (HR)
    Row 2: Gamma Curves (Area, Perimeter, CC)
    """
    target_image = sample_row["target_image"]
    pred_image = sample_row["pred_image"]
    input_image = sample_row["input_image"]
    dem_image = sample_row["dem_image"]

    pred_gamma = sample_row["pred_gamma"]
    target_gamma = sample_row["target_gamma"]
    mean_precip = sample_row["mean_precip"]
    sample_idx = sample_row.name

    loss = sample_row.get("total_loss", np.nan)
    trust = sample_row.get("Trust_Score", np.nan)

    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 0.7], hspace=0.25, wspace=0.2)

    # --- ROW 1: SPATIAL FIELDS ---

    # 1. Determine Global Max for Precipitation Colorbar (Input, Pred, Target)
    # We want them to share the exact same scale for visual comparison.
    # Handle NaNs just in case.
    vmax = np.nanmax(
        [np.nanmax(target_image), np.nanmax(pred_image), np.nanmax(input_image)]
    )
    if vmax <= 0:
        vmax = 1.0  # Safety

    precip_norm = mcolors.Normalize(vmin=0, vmax=vmax)

    # 2. Prepare Colormap with Grey for Zeros (NaNs)
    # We copy 'Blues' and set the 'bad' (masked) color to lightgrey
    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)  # Grey background for dry areas

    # Helper to mask zeros for plotting
    def mask_zeros(img):
        masked = img.copy()
        # Use a small epsilon or the drizzle threshold logic (e.g. 0.05)
        masked[masked <= 0.05] = np.nan
        return masked

    # A. DEM (Terrain)
    ax_dem = fig.add_subplot(gs[0, 0])
    im_dem = ax_dem.imshow(dem_image, cmap="terrain", origin="lower")
    ax_dem.set_title("Digital Elevation Model (DEM)")
    fig.colorbar(im_dem, ax=ax_dem, shrink=0.6, label="Elevation")

    # B. Input (Low Res)
    ax_in = fig.add_subplot(gs[0, 1])
    # Apply masking and shared cmap/norm
    im_in = ax_in.imshow(
        mask_zeros(input_image), cmap=precip_cmap, norm=precip_norm, origin="lower"
    )
    ax_in.set_title("Input (Low Res)")
    fig.colorbar(im_in, ax=ax_in, shrink=0.6, label="mm/hr")

    # C. Prediction (SR)
    ax_pred = fig.add_subplot(gs[0, 2])
    im_pred = ax_pred.imshow(
        mask_zeros(pred_image), cmap=precip_cmap, norm=precip_norm, origin="lower"
    )
    ax_pred.set_title("Prediction (SR)")
    fig.colorbar(im_pred, ax=ax_pred, shrink=0.6, label="mm/hr")

    # D. Target (HR)
    ax_targ = fig.add_subplot(gs[0, 3])
    im_targ = ax_targ.imshow(
        mask_zeros(target_image), cmap=precip_cmap, norm=precip_norm, origin="lower"
    )
    ax_targ.set_title(f"Target (High Res) | Mean: {mean_precip:.2f}")
    # Key: This colorbar now applies to B, C, and D equally
    fig.colorbar(im_targ, ax=ax_targ, shrink=0.6, label="mm/hr")

    # --- ROW 2: TOPOLOGY (GAMMA) ---
    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, :], wspace=0.3
    )

    for j in range(3):
        ax = fig.add_subplot(gs_bottom[0, j])
        ax.plot(
            quantiles,
            target_gamma[j],
            "o-",
            label="Target",
            color="royalblue",
            linewidth=2,
        )
        ax.plot(
            quantiles, pred_gamma[j], "x--", label="Pred", color="salmon", linewidth=2
        )
        ax.set_title(gamma_types[j], fontsize=14)
        ax.set_xlabel("Threshold (mm/hr)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Log scale for Area/Perim if values are large
        if j < 2 and np.max(target_gamma[j]) > 100:
            ax.set_yscale("log")

        if j == 0:
            ax.legend(fontsize=12)

    # --- Title ---
    metrics_str = f"Loss: {loss:.4f}"
    if not np.isnan(trust):
        metrics_str += f" | Trust: {trust:.3f}"

    fig.suptitle(f"{title} | Sample {sample_idx}\n{metrics_str}", fontsize=18, y=0.96)

    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    safe_title = title.replace(" ", "_").replace("#", "").lower()
    plt.savefig(
        os.path.join(plot_save_dir, f"{safe_title}_sample_{sample_idx}.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_sample_comparisons_fixed(metrics_df, quantiles, output_dir, n_samples=5):
    print(f"\nGenerating comprehensive plots (Best/Mid/Worst {n_samples}) per group...")
    if "precip_group" not in metrics_df:
        return
    grouped = metrics_df.groupby("precip_group")
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]

    for group_name in group_order:
        group_df = grouped.get_group(group_name)
        if len(group_df) == 0:
            continue
        print(f"Processing Group: {group_name} (N={len(group_df)})")

        # Sort by Total Loss
        sorted_df = group_df.sort_values("total_loss", ascending=True)

        # Best N
        for i in range(min(n_samples, len(sorted_df))):
            _plot_comprehensive_sample(
                sorted_df.iloc[i], quantiles, f"Best #{i+1}", group_name, output_dir
            )
        # Worst N
        worst_samples = sorted_df.tail(min(n_samples, len(sorted_df))).iloc[::-1]
        for i in range(len(worst_samples)):
            _plot_comprehensive_sample(
                worst_samples.iloc[i],
                quantiles,
                f"Worst #{i+1}",
                group_name,
                output_dir,
            )
        # Mid N
        if len(sorted_df) > 2 * n_samples:
            mid_idx = len(sorted_df) // 2
            mid_samples = sorted_df.iloc[
                max(0, mid_idx - n_samples // 2) : mid_idx + n_samples // 2
            ]
            for i in range(len(mid_samples)):
                _plot_comprehensive_sample(
                    mid_samples.iloc[i],
                    quantiles,
                    f"Median #{i+1}",
                    group_name,
                    output_dir,
                )


def plot_trust_analysis(metrics_df, output_dir):
    # Check if Trust_Score exists AND is not entirely NaN
    if (
        "Trust_Score" not in metrics_df.columns
        or metrics_df["Trust_Score"].isna().all()
    ):
        print("No valid Trust Score data found (Vanilla Mode). Skipping trust plots.")
        return

    print("\nGenerating Trust Gate analysis plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Trust vs Precip Intensity
    # Use log scale for precip if highly skewed
    sns.scatterplot(
        data=metrics_df,
        x="mean_precip",
        y="Trust_Score",
        hue="precip_group",
        alpha=0.6,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Trust Score vs Precipitation Intensity")
    axes[0].set_ylabel("Trust Weight (1.0 = High Confidence)")
    axes[0].set_xlabel("Mean Precipitation (mm/hr)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Trust Distribution by Group
    sns.boxplot(
        data=metrics_df,
        x="precip_group",
        y="Trust_Score",
        order=["Zero", "Low", "Mid", "High"],
        ax=axes[1],
        palette="viridis",
    )
    axes[1].set_title("Trust Score Distribution by Group")
    axes[1].set_ylabel("Trust Weight")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trust_analysis.png"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_distributions(metrics_df, output_dir):
    print("\nGenerating metric distribution box plots...")
    cols = ["total_loss", "mae_loss", "surrogate_loss"]
    if "Trust_Score" in metrics_df:
        cols.append("Trust_Score")

    n_plots = len(cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        if col in metrics_df.columns:
            data = metrics_df[col].dropna()
            axes[i].boxplot(
                data,
                vert=True,
                patch_artist=True,
                labels=[col],
                medianprops={"color": "red"},
            )
            axes[i].set_title(f"{col}\nMean: {data.mean():.4f}")
            axes[i].grid(True, linestyle="--", alpha=0.6)
            if "loss" in col and data.mean() > 0:
                axes[i].set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "evaluation_distributions.png"), bbox_inches="tight"
    )
    plt.close(fig)


def plot_gamma_mean_std_by_quantile(metrics_df, group_metrics, quantiles, output_dir):
    print("\nGenerating mean/std gamma plots...")
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", "mean_std_groups")
    os.makedirs(plot_save_dir, exist_ok=True)
    grouped = metrics_df.groupby("precip_group")
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups] + [
        "All"
    ]

    for group_name in group_order:
        if group_name == "All":
            group_df = metrics_df
        else:
            group_df = grouped.get_group(group_name)
        if len(group_df) == 0:
            continue

        preds = np.stack(group_df["pred_gamma"].values)
        targets = np.stack(group_df["target_gamma"].values)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["Area", "Perimeter", "CCs"]

        for j in range(3):
            ax = axes[j]
            m_t, s_t = np.nanmean(targets[:, j], axis=0), np.nanstd(
                targets[:, j], axis=0
            )
            m_p, s_p = np.nanmean(preds[:, j], axis=0), np.nanstd(preds[:, j], axis=0)

            ax.plot(quantiles, m_t, "o-", label="Target", color="royalblue")
            ax.fill_between(
                quantiles, m_t - s_t, m_t + s_t, color="royalblue", alpha=0.2
            )
            ax.plot(quantiles, m_p, "x--", label="Pred", color="salmon")
            ax.fill_between(quantiles, m_p - s_p, m_p + s_p, color="salmon", alpha=0.2)

            ax.set_title(titles[j])
            if j == 0:
                ax.legend()
            if j < 2:
                ax.set_yscale("log")
            ax.grid(True, alpha=0.5)

        fig.suptitle(f"Group: {group_name} (N={len(group_df)})", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f"mean_std_{group_name.lower()}.png"))
        plt.close(fig)


def plot_per_feature_matrices(per_feature_metrics, output_dir):
    print("\nGenerating per-feature matrices...")
    matrices = [
        per_feature_metrics["r2_matrix"],
        per_feature_metrics["mae_matrix"],
        per_feature_metrics["var_matrix"],
    ]
    titles = ["R² Score", "MSE", "Target Variance"]
    cmaps = ["RdBu", "viridis", "magma"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 24))
    for i, ax in enumerate(axes):
        sns.heatmap(
            matrices[i],
            ax=ax,
            cmap=cmaps[i],
            annot=True,
            fmt=".2e" if i > 0 else ".2f",
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": titles[i]},
        )
        ax.set_title(titles[i], fontsize=14)
        ax.set_xticklabels([f"{q:.2f}" for q in per_feature_metrics["quantiles"]])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "evaluation_matrices.png"), bbox_inches="tight"
    )
    plt.close(fig)


def plot_training_log(log_path, output_dir, config):
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}")
        return

    print("\nGenerating training log plot...")
    try:
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # 1. Loss
        if "train_loss_total" in df:
            axes[0].plot(df["epoch"], df["train_loss_total"], label="Train Total")
        if "val_loss_total" in df:
            axes[0].plot(df["epoch"], df["val_loss_total"], label="Val Total")

        # Check if values are valid for log scale
        if df["train_loss_total"].min() > 0:
            axes[0].set_yscale("log")

        axes[0].legend()
        axes[0].set_title("Loss History")
        axes[0].grid(True, alpha=0.3)

        # 2. Metric / Audit
        # Ensure we only set log scale if valid values exist
        if (
            "val_consistency_gap" in df
            and df["val_consistency_gap"].notna().any()
            and df["val_consistency_gap"].max() > 0
        ):
            axes[1].set_yscale("log")
            axes[1].plot(
                df["epoch"],
                df["val_consistency_gap"],
                color="purple",
                label="Consistency Gap",
            )
        if "val_intrinsic_error" in df:
            axes[1].plot(
                df["epoch"],
                df["val_intrinsic_error"],
                color="orange",
                label="Intrinsic Emu Error",
            )

        # Check for valid log scale values
        if "val_consistency_gap" in df and df["val_consistency_gap"].max() > 0:
            axes[1].set_yscale("log")

        axes[1].legend()
        axes[1].set_title("Emulator Audit")
        axes[1].grid(True, alpha=0.3)

        # 3. Trust & Weights
        # --- FIX: Check for the correct column name 'avg_trust_weight' ---
        trust_col = "avg_trust_weight"
        if trust_col not in df.columns and "avg_train_trust" in df.columns:
            trust_col = "avg_train_trust"  # Backwards compatibility

        if trust_col in df:
            axes[2].plot(
                df["epoch"],
                df[trust_col],
                color="green",
                label="Avg Trust (Train)",
                linewidth=2,
            )
        else:
            print(
                f"Warning: Trust column not found. Available columns: {df.columns.tolist()}"
            )

        if "current_metric_weight" in df:
            ax2_twin = axes[2].twinx()
            ax2_twin.plot(
                df["epoch"],
                df["current_metric_weight"],
                color="gray",
                linestyle="--",
                label="Metric Weight Lambda",
            )
            ax2_twin.set_ylabel("Lambda")
            ax2_twin.legend(loc="upper right")

        axes[2].set_ylabel("Trust Score (0-1)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_title("Trust & Weights")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_history.png"))
        print(f"Plot saved to {os.path.join(output_dir, 'training_history.png')}")
        plt.close(fig)

    except Exception as e:
        print(f"Log plot failed: {e}")
        import traceback

        traceback.print_exc()


def plot_perception_distortion(metrics_df, output_dir):
    """
    Plots the Perception-Distortion plane (MAE vs Log-Spectral Distance).
    """
    print("\nGenerating Perception-Distortion (Pareto) plot...")

    # Critical check: Ensure variable names match physical quantities in your future work.
    if "mae_loss" not in metrics_df or "spectral_dist" not in metrics_df:
        print("Missing 'mae_loss' or 'spectral_dist'. Skipping.")
        return

    # Filter for valid data
    plot_df = metrics_df[metrics_df["mean_precip"] > 0.05].copy()
    if len(plot_df) < 10:
        print("Not enough wet samples for robust Pareto plot. Using all data.")
        plot_df = metrics_df

    # --- CALCULATION OF MEANS ---
    # We calculate statistics on the data actually being plotted (plot_df)
    mean_distortion = plot_df["mae_loss"].mean()
    mean_perception = plot_df["spectral_dist"].mean()

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(5, 5)

    # Main Scatter
    ax_main = fig.add_subplot(gs[1:, :-1])

    scatter = ax_main.scatter(
        plot_df["mae_loss"],
        plot_df["spectral_dist"],
        c=np.log1p(plot_df["mean_precip"]),
        cmap="viridis",
        alpha=0.6,
        edgecolor="w",
        linewidth=0.3,
        s=30,
    )

    try:
        sns.kdeplot(
            data=plot_df,
            x="mae_loss",
            y="spectral_dist",
            levels=5,
            color="k",
            linewidths=0.8,
            ax=ax_main,
            alpha=0.5,
        )
    except Exception:
        pass

    ax_main.set_xlabel("Distortion (MAE) [mm/hr]", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Perception (Log-Spectral Dist)", fontsize=12, fontweight="bold")
    ax_main.grid(True, linestyle="--", alpha=0.4)

    # Marginal Histogram (X - Distortion)
    ax_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_x.hist(plot_df["mae_loss"], bins=50, color="gray", alpha=0.7, density=True)
    ax_x.axis("off")

    title_str = (
        f"Perception-Distortion Trade-off\n"
        f"Mean MAE: {mean_distortion:.4f} | Mean Perception: {mean_perception:.4f}"
    )
    ax_x.set_title(title_str, fontsize=12)

    # Marginal Histogram (Y - Perception)
    ax_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_y.hist(
        plot_df["spectral_dist"],
        bins=50,
        orientation="horizontal",
        color="gray",
        alpha=0.7,
        density=True,
    )
    ax_y.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Log(Mean Precip Intensity)")

    ax_main.text(
        0.05,
        0.05,
        "Ideal Region\n(Sharp & Accurate)",
        transform=ax_main.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"),
    )

    out_path = os.path.join(output_dir, "perception_distortion_pareto.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved Perception-Distortion plot to {out_path}")
    plt.close(fig)
