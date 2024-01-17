from typing import List, Optional
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from unet import UNet


class DDPM(nn.Module):
    def __init__(
        self, 
        resolution: int,
        in_channels: int,
        channels: List[int],
        dropout: float = 0.0,
        num_res_blocks: int = 2,
        num_timesteps: int = 1000
    ) -> None:
        super().__init__()
        assert resolution in [28, 32, 256]
        self.unet = UNet(resolution, in_channels, channels, num_res_blocks, dropout) 
        self.num_timesteps = num_timesteps
        self.in_channels = in_channels
        self.resolution = resolution

    def forward(self, x: Tensor):
        device = x.device

        B = x.size(0)
        T = self.num_timesteps

        x = 2.0 * x - 1.0
        t = torch.randint(
            1,
            self.num_timesteps+1, 
            (B,),
            device=device
        )

        betas = torch.linspace(1e-4, 0.02, self.num_timesteps, device=device)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(0)
        alpha_cumprod = alphas_cumprod[t-1].reshape(B, 1, 1, 1)

        noise = torch.randn_like(x)
        x_t = alpha_cumprod.sqrt() * x + torch.sqrt( 1 - alpha_cumprod ) * noise

        noise_pred = self.unet(x_t, t)
        loss = F.mse_loss(noise_pred, noise, reduction='none').sum([1, 2, 3])

        return loss
        
    @torch.no_grad()
    def sample(self, n: int = 1, generator: Optional[torch.Generator] = None):
        device = next(self.parameters()).device
        T = self.num_timesteps
        x_t = torch.randn(
            n,
            self.in_channels,
            self.resolution,
            self.resolution,
            generator=generator,
            device=device
        )

        betas = torch.linspace(1e-4, 0.02, self.num_timesteps, device=device)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(0)

        for t in range(T, 0, -1):
            z = torch.randn(x_t.size(), generator=generator, device=device) \
                if t > 1 else torch.zeros_like(x_t)

            beta = betas[t-1]
            alpha = alphas[t-1]
            alpha_cumprod = alphas_cumprod[t-1]

            t_tensor = torch.tensor([t for _ in range(n)], device=device)
            x_t = alpha ** (-0.5) * ( 
                x_t - ( ( 1 - alpha ) / torch.sqrt( 1 - alpha_cumprod ) )
                * self.unet(x_t, t_tensor)
            ) + math.sqrt(beta) * z

        x = ( x_t + 1.0 ) / 2.0

        return x
