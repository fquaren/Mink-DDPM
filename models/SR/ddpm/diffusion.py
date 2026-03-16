import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Diffusion(nn.Module):
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=128,
        device="cuda",
    ):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define linear beta schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, s=0.008):
        """
        Implements the cosine variance schedule as proposed by Nichol & Dhariwal (2021).
        """
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_images(self, x, t):
        """
        Diffuses images x to timestep t.
        Returns x_t and the noise used.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """Samples n random timesteps for training."""
        return torch.randint(low=1, high=self.noise_steps, size=(n,)).to(self.device)

    def sample(self, model, n, conditions):
        """
        Generates new images from noise using the model and conditional input.
        conditions: The low-res/upsampled images [B, C, H, W]
        """
        model.eval()
        print(f"Sampling {n} images....")

        # Start from pure noise
        x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)

        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # Predict noise
                predicted_noise = model(x, t, conditions)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # DDPM Sampling equation
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                ) + torch.sqrt(beta) * noise

        model.train()

        x = x.clamp(-1.0, 1.0)
        return x

    def sample_ddim(self, model, n, conditions, ddim_steps=50, eta=0.0):
        """
        Generates new images using the deterministic Denoising Diffusion Implicit Models (DDIM).
        Drastically accelerates inference during validation (e.g., 50 steps vs 1000).

        Args:
            model: The ContextUnet.
            n (int): Batch size.
            conditions (Tensor): The LR/DEM conditioning tensor [B, C, H, W].
            ddim_steps (int): The number of sub-sampled timesteps.
            eta (float): The stochasticity parameter. eta=0.0 is purely deterministic DDIM.
        """
        model.eval()
        print(f"Sampling {n} images using DDIM ({ddim_steps} steps)...")

        # Start from pure noise
        x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)

        # 1. Create the sub-sampled timestep sequence
        # e.g., if noise_steps=1000 and ddim_steps=50, step_ratio = 20
        step_ratio = self.noise_steps // ddim_steps
        timesteps = (np.arange(1, ddim_steps + 1) * step_ratio) - 1
        # Reverse it to go from T -> 0
        timesteps = list(reversed(timesteps))

        # Add an explicit 0 at the end to map back to the data domain
        timesteps = timesteps + [0]

        with torch.no_grad():
            for i in tqdm(range(len(timesteps) - 1), position=0, leave=False):
                # Current and next (previous in reverse time) timestep
                t_current = timesteps[i]
                t_prev = timesteps[i + 1]

                # Convert to Tensors
                t = (torch.ones(n) * t_current).long().to(self.device)
                t_prev_tensor = (torch.ones(n) * t_prev).long().to(self.device)

                # 2. Predict noise
                with torch.amp.autocast("cuda"):
                    predicted_noise = model(x, t, conditions)

                # Fetch alphas corresponding to the sub-sampled timesteps
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev_tensor][:, None, None, None]

                # 3. Predict the original x0 (the un-noised data)
                # x0 = (x_t - sqrt(1 - alpha_hat_t) * noise) / sqrt(alpha_hat_t)
                pred_x0 = (
                    x - torch.sqrt(1 - alpha_hat_t) * predicted_noise
                ) / torch.sqrt(alpha_hat_t)

                # We map x0 to [-1, 1] structurally (optional, but stabilizing)
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

                # 4. Calculate variance corresponding to the DDIM process
                # When eta = 0.0, sigma_t = 0.0 -> deterministic sampling
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_hat_prev)
                    / (1 - alpha_hat_t)
                    * (1 - alpha_hat_t / alpha_hat_prev)
                )

                # 5. Compute the DDIM step to get x_{t-1}
                # "direction pointing to x_t"
                dir_xt = torch.sqrt(1 - alpha_hat_prev - sigma_t**2) * predicted_noise

                # Add noise if eta > 0.0, else purely deterministic
                noise = torch.randn_like(x) if t_prev > 0 else 0.0

                x = torch.sqrt(alpha_hat_prev) * pred_x0 + dir_xt + sigma_t * noise

        model.train()

        # Clamp output strictly to the defined data domain
        x = x.clamp(-1.0, 1.0)
        return x
