"""DDPM noise schedule, forward process (add_noise) and reverse process (sample)."""
import torch


class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=self.device), alpha_bars[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        # Pre-computed constants for sampling (DDPM posterior q(x_{t-1} | x_t, x_0))
        self.posterior_coef_x0 = betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars)
        self.posterior_coef_xt = (1.0 - alpha_bars_prev) * alphas.sqrt() / (1.0 - alpha_bars)
        self.posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    # ---------------- forward process ----------------

    def add_noise(self, x0, t, noise=None):
        """x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise

    def predict_x0(self, x_t, t, eps):
        """Invert add_noise given the predicted noise (clamped to the valid range)."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ((x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)

    # ---------------- reverse process ----------------

    @torch.no_grad()
    def sample(self, model, shape, n_snapshots=8, generator=None):
        """Ancestral DDPM sampling.

        Returns:
            x:         final images [B, C, H, W] in [-1, 1]
            snapshots: list of n_snapshots tensors (on CPU) = the intermediate x_t's,
                       ordered from noisy -> clean
        """
        was_training = model.training
        model.eval()

        B = shape[0]
        x = torch.randn(shape, device=self.device, generator=generator)
        snap_steps = set(torch.linspace(self.T - 1, 0, n_snapshots).long().tolist())
        snapshots = []

        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            x0 = self.predict_x0(x, t, eps)

            mean = self.posterior_coef_x0[i] * x0 + self.posterior_coef_xt[i] * x
            if i > 0:
                noise = torch.randn(shape, device=self.device, generator=generator)
                x = mean + self.posterior_var[i].sqrt() * noise
            else:
                x = mean

            if i in snap_steps:
                snapshots.append(x.clamp(-1, 1).cpu())

        model.train(was_training)
        return x.clamp(-1, 1), snapshots
