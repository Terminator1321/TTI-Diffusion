import torch
import math


class CosineNoiseScheduler:
    def __init__(self, timesteps=1000, s=0.008, max_beta=0.999):
        self.timesteps = timesteps
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)

        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])

        self.betas = torch.clamp(betas, max=max_beta)
        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas,dim=0)
        
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_cumprod.to(x0.device)[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)[:, None, None, None]
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise
    
if __name__ == "__main__":

    scheduler = CosineNoiseScheduler(1000)

    print("Betas:", scheduler.betas.shape)
    print("Alphas:", scheduler.alphas.shape)
    print("Alpha cumprod:", scheduler.alpha_cumprod.shape)

    # Simulate a batch of 64x64 RGB images
    x0 = torch.randn(4, 3, 64, 64)

    # Different noise levels for each image
    t = torch.tensor([0, 250, 500, 999])

    xt, noise = scheduler.add_noise(x0, t)

    print("Original:", x0.shape)
    print("Noisy:", xt.shape)
    print("Noise:", noise.shape)

