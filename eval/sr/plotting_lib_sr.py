import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm

# Import shared libraries
import eval.sr.io_lib_sr as io_lib
import eval.sr.metrics_lib_sr as metrics_lib
import eval.sr.plotting_lib_sr as plotting_lib

# Import DDPM specific modules
from models.SR.ddpm.ddpm import ContextUnet
from models.SR.ddpm.diffusion import Diffusion

# Import Physics Logic
from src.loss import MinkowskiLoss
from src.utils import load_emulator
from data.preprocessing.compute_gamma_targets import compute_gamma_matrix

warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)


class DataDenormalizer:
    def __init__(self, stats_path):
        try:
            data = np.load(stats_path)
            self.max_val = float(data.item())
            print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")
        except FileNotFoundError:
            print(
                f"Warning: Scaling stats not found at {stats_path}. Defaulting to 1.0."
            )
            self.max_val = 1.0

    def unnormalize(self, x_norm):
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()
        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        return np.maximum(x_phys, 0.0)

    def unnormalize_torch(self, x_norm):
        max_val_tensor = torch.tensor(
            self.max_val, device=x_norm.device, dtype=x_norm.dtype
        )
        x_scaled = x_norm * max_val_tensor
        # CRITICAL: Clamp the exponent to avoid Inf in float32
        x_scaled = torch.clamp(x_scaled, max=50.0)
        x_phys = torch.expm1(x_scaled)
        return F.relu(x_phys)


def load_ddpm_model(config, device, run_dir):
    checkpoint_path = os.path.join(run_dir, "ddpm_latest.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(run_dir, "ddpm_best.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"DDPM checkpoint not found at {checkpoint_path}")

    print(f"Loading DDPM model from: {checkpoint_path}")
    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    # Align instantiation parameters exactly with training script
    diffusion = Diffusion(
        img_size=config.get("PATCH_SIZE", 128),
        device=device,
    )
    return model, diffusion


def compute_crps_ensemble(pred_ensemble, target):
    term_1 = torch.mean(torch.abs(pred_ensemble - target), dim=1)
    diffs = torch.abs(pred_ensemble.unsqueeze(2) - pred_ensemble.unsqueeze(1))
    term_2 = torch.mean(diffs, dim=(1, 2))
    crps_map = term_1 - 0.5 * term_2
    return crps_map.mean(dim=(1, 2))


def run_ddpm_crps_audit(
    model,
    diffusion,
    loader,
    device,
    denormalizer,
    drizzle_threshold,
    n_ensemble=16,
    n_batches=10,
):
    print(f"\n--- Running CRPS Audit (N={n_ensemble} samples per image) ---")
    crps_scores = []

    with torch.no_grad():
        for i, (X, Y_true, _) in enumerate(tqdm(loader, desc="CRPS Ensemble")):
            if i >= n_batches:
                break

            X, Y_true = X.to(device), Y_true.to(device)

            max_batch_allowed = 128 // n_ensemble
            B_sub = min(X.shape[0], max(1, max_batch_allowed))
            X_sub = X[:B_sub]
            Y_true_sub = Y_true[:B_sub]

            X_repeated = X_sub.repeat_interleave(n_ensemble, dim=0)

            with torch.amp.autocast("cuda"):
                samples_norm = diffusion.sample_ddim(
                    model, n=B_sub * n_ensemble, conditions=X_repeated, ddim_steps=50
                )

            # DOMAIN INVERSION [-1, 1] -> [0, 1]
            samples_shifted = (samples_norm.clamp(-1.0, 1.0) + 1.0) / 2.0
            Y_true_shifted = (Y_true_sub + 1.0) / 2.0

            # Denormalize to Physical mm/h
            samples_phys = denormalizer.unnormalize_torch(samples_shifted)
            samples_phys = samples_phys * (samples_phys > drizzle_threshold).float()
            samples_ensemble = samples_phys.view(
                B_sub, n_ensemble, Y_true.shape[2], Y_true.shape[3]
            )

            Y_phys = denormalizer.unnormalize_torch(Y_true_shifted)
            Y_phys = Y_phys * (Y_phys > drizzle_threshold).float()

            batch_crps = compute_crps_ensemble(samples_ensemble, Y_phys)
            crps_scores.extend(batch_crps.cpu().numpy())

    avg_crps = np.mean(crps_scores)
    print(f"Average CRPS (Ensemble={n_ensemble}): {avg_crps:.4f}")
    return avg_crps


def run_ddpm_prediction_loop(
    model,
    diffusion,
    loader,
    audit_criterion,
    emulator,
    device,
    quantile_levels,
    pixel_size_km,
    denormalizer,
    pers_thresh,
    drizzle_threshold=0.1,
):
    model.eval()
    if emulator:
        emulator.eval()

    all_preds_gamma_analytic, all_targets_gamma_analytic = [], []
    all_preds_phys, all_targets_phys, all_inputs_phys = [], [], []
    all_total_losses, all_mse_losses, all_surrogate_losses = [], [], []
    all_fss, all_sal_S, all_sal_A, all_sal_L = [], [], [], []

    audit_results = {
        "L1_phys_err": [],
        "L2_perc_err": [],
        "L3_intr_err": [],
        "consistency_gap": [],
    }

    mae_criterion = nn.L1Loss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    print("\n--- Starting DDPM Inference (Fast DDIM) ---")
    with torch.no_grad():
        for batch_idx, (X, Y_true_norm, Y_gamma_log_target) in enumerate(tqdm(loader)):
            X, Y_true_norm, Y_gamma_log_target = (
                X.to(device),
                Y_true_norm.to(device),
                Y_gamma_log_target.to(device),
            )

            with torch.amp.autocast("cuda"):
                pred_X_norm = diffusion.sample_ddim(
                    model, n=X.shape[0], conditions=X, ddim_steps=50
                )

            # DENORMALIZE [-1, 1] -> [0, 1] -> Physical mm/h
            pred_X_shifted = (pred_X_norm.clamp(-1.0, 1.0) + 1.0) / 2.0
            Y_true_shifted = (Y_true_norm + 1.0) / 2.0
            X_shifted = (X[:, 0:1, :, :] + 1.0) / 2.0

            pred_X_phys = denormalizer.unnormalize_torch(pred_X_shifted)
            Y_true_phys = denormalizer.unnormalize_torch(Y_true_shifted)
            X_phys = denormalizer.unnormalize_torch(X_shifted)

            pred_X_phys = pred_X_phys * (pred_X_phys > drizzle_threshold).float()
            Y_true_phys = Y_true_phys * (Y_true_phys > drizzle_threshold).float()

            pred_X_np = pred_X_phys.squeeze(1).cpu().numpy()
            Y_true_np = Y_true_phys.squeeze(1).cpu().numpy()

            eval_threshold = 1.0
            eval_window = 5

            fss_batch = metrics_lib.compute_batch_fss(
                pred_X_np, Y_true_np, eval_window, eval_threshold
            )
            S_batch, A_batch, L_batch = metrics_lib.compute_batch_sal(
                pred_X_np, Y_true_np, eval_threshold, pixel_size_km**2
            )

            all_fss.append(fss_batch)
            all_sal_S.append(S_batch)
            all_sal_A.append(A_batch)
            all_sal_L.append(L_batch)

            batch_gammas = [
                compute_gamma_matrix(
                    pred_X_np[i], quantile_levels, pixel_size_km, pers_thresh
                )
                for i in range(pred_X_np.shape[0])
            ]

            all_preds_gamma_analytic.append(np.array(batch_gammas))
            all_targets_gamma_analytic.append(Y_gamma_log_target.cpu().numpy())

            loss_mae_sample = mae_criterion(pred_X_phys, Y_true_phys).mean(
                dim=(1, 2, 3)
            )
            loss_mse_sample = mse_criterion(pred_X_phys, Y_true_phys).mean(
                dim=(1, 2, 3)
            )
            loss_surr_sample = torch.zeros_like(loss_mae_sample)

            if emulator and audit_criterion:
                gamma_pred_phys = emulator(pred_X_phys)
                gamma_true_phys = emulator(Y_true_phys)

                gamma_pred_log = torch.log1p(gamma_pred_phys)
                gamma_true_log = torch.log1p(gamma_true_phys)

                l1 = audit_criterion(gamma_pred_log, Y_gamma_log_target)
                l2 = audit_criterion(gamma_pred_log, gamma_true_log)
                l3 = audit_criterion(gamma_true_log, Y_gamma_log_target)
                gap = torch.abs(l1 - l2)

                loss_surr_sample = l1

                audit_results["L1_phys_err"].append(l1.cpu().numpy())
                audit_results["L2_perc_err"].append(l2.cpu().numpy())
                audit_results["L3_intr_err"].append(l3.cpu().numpy())
                audit_results["consistency_gap"].append(gap.cpu().numpy())
            else:
                nan_vec = np.full(X.shape[0], np.nan)
                for k in audit_results:
                    audit_results[k].append(nan_vec)

            all_preds_phys.append(pred_X_phys.squeeze(1).cpu().numpy())
            all_targets_phys.append(Y_true_phys.squeeze(1).cpu().numpy())
            all_inputs_phys.append(X_phys.squeeze(1).cpu().numpy())

            all_total_losses.append(loss_mae_sample.cpu().numpy())
            all_mse_losses.append(loss_mse_sample.cpu().numpy())
            all_surrogate_losses.append(loss_surr_sample.cpu().numpy())

    return (
        np.concatenate(all_preds_gamma_analytic, axis=0),
        np.concatenate(all_targets_gamma_analytic, axis=0),
        np.concatenate(all_preds_phys, axis=0),
        np.concatenate(all_targets_phys, axis=0),
        np.concatenate(all_inputs_phys, axis=0),
        np.concatenate(all_total_losses, axis=0),
        np.concatenate(all_mse_losses, axis=0),
        np.concatenate(all_surrogate_losses, axis=0),
        np.concatenate(all_fss, axis=0),
        np.concatenate(all_sal_S, axis=0),
        np.concatenate(all_sal_A, axis=0),
        np.concatenate(all_sal_L, axis=0),
        {k: np.concatenate(v, axis=0) for k, v in audit_results.items()},
    )


def main(run_dir):
    config, device = io_lib.setup_evaluation(run_dir)

    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_precip_max_val.npy"
    )
    denormalizer = DataDenormalizer(scaler_path)

    model, diffusion = load_ddpm_model(config, device, run_dir)
    dem_stats = io_lib.load_dem_stats(config)
    test_loader = io_lib.load_data(config, dem_stats, denormalizer.max_val)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
    EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)

    emulator = None
    audit_criterion = None
    if EMULATOR_PATH and os.path.exists(EMULATOR_PATH):
        print(f"Loading emulator for Minkowski Audit: {EMULATOR_PATH}")
        emulator = load_emulator(EMULATOR_PATH, config, device)
        audit_criterion = MinkowskiLoss(QUANTILE_LEVELS).to(device)
    else:
        print("Running in VANILLA MODE (No Emulator Audit).")

    (
        all_preds_gamma,
        all_targets_gamma,
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        all_fss,
        all_sal_S,
        all_sal_A,
        all_sal_L,
        audit_results,
    ) = run_ddpm_prediction_loop(
        model,
        diffusion,
        test_loader,
        audit_criterion,
        emulator,
        device,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
        denormalizer,
        config.get("PERSISTENCE_THRESHOLD", 0.05),
        config.get("DRIZZLE_THRESHOLD", 0.1),
    )

    all_dems = [None] * len(all_preds_phys)

    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_gamma,
        all_targets_gamma,
        all_inputs_phys,
        all_targets_phys,
        all_preds_phys,
        all_dems,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    metrics_df["FSS"] = all_fss
    metrics_df["SAL_Structure"] = all_sal_S
    metrics_df["SAL_Amplitude"] = all_sal_A
    metrics_df["SAL_Location"] = all_sal_L

    metrics_df["L1_Physical_Error"] = audit_results["L1_phys_err"]
    metrics_df["Consistency_Flag"] = audit_results["consistency_gap"]

    group_metrics = metrics_lib.calculate_grouped_metrics(metrics_df)
    per_feature_gamma_metrics = metrics_lib.calculate_per_feature_gamma_metrics(
        metrics_df, QUANTILE_LEVELS
    )

    io_lib.save_metrics_text(run_dir, group_metrics, per_feature_gamma_metrics)
    io_lib.save_metrics_npz(run_dir, metrics_df, per_feature_gamma_metrics)

    if emulator:
        pass_rate = (metrics_df["Consistency_Flag"] < 0.3).mean() * 100
        print(f"\nEmulator Pass Rate (Flag < 0.3): {pass_rate:.2f}%")

    avg_crps = run_ddpm_crps_audit(
        model,
        diffusion,
        test_loader,
        device,
        denormalizer,
        config.get("DRIZZLE_THRESHOLD", 0.1),
    )

    with open(os.path.join(run_dir, "crps_score.txt"), "w") as f:
        f.write(f"Average CRPS (N=16, 10 Batches): {avg_crps:.6f}\n")

    plotting_lib.plot_sample_comparisons_fixed(metrics_df, QUANTILE_LEVELS, run_dir)
    plotting_lib.plot_per_feature_matrices(per_feature_gamma_metrics, run_dir)
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df, group_metrics, QUANTILE_LEVELS, run_dir
    )
    plotting_lib.plot_metric_distributions(metrics_df, run_dir)

    print("\nDDPM Evaluation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    main(args.run_dir)
