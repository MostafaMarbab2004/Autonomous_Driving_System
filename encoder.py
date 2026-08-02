# ──────────────────────────────────────────────────────────
# encoder.py  ←  Convolutional VAE  (128×128 RGB → latent z)
# ──────────────────────────────────────────────────────────
#
#  Architecture
#  ─────────────
#  Encoder:  3 × Conv2d (stride-2)   →  Flatten → μ, log_σ²
#  Decoder:  Linear → Reshape → 3 × ConvTranspose2d → sigmoid
#
#  Input  : (B, 3, 128, 128)  float in [0, 1]
#  Output : recon (B,3,128,128),  μ (B, latent_dim),  log_var (B, latent_dim)
# ──────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN that maps an image to (mu, log_var) in latent space."""

    def __init__(self, latent_dim: int = 128, img_channels: int = 3):
        super().__init__()
        # 128 → 64 → 32 → 16 → 8 → 4  (5 stride-2 convs)
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32,  kernel_size=4, stride=2, padding=1),  # 64×64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1),           # 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),           # 16×16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # 8×8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),           # 4×4
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 512 * 4 * 4 = 8192
        self.flat_dim = 512 * 4 * 4
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim) # featuers in 8192 out 128
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim) # featuers in 8192 out 128

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Maps a latent vector z back to an image."""

    def __init__(self, latent_dim: int = 128, img_channels: int = 3):
        super().__init__()
        # 512 * 4 * 4 = 8192
        self.flat_dim = 512 * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        # 4 → 8 → 16 → 32 → 64 → 128  (5 stride-2 deconvs)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8×8
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16×16
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),  # 64×64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32,  img_channels, kernel_size=4, stride=2, padding=1),  # 128×128
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """
    Full β-VAE.

    Usage
    ─────
    model = VAE(latent_dim=128)
    recon, mu, log_var = model(images)          # training
    z = model.encode(images)                    # inference → deterministic μ
    """

    def __init__(self, latent_dim: int = 128, img_channels: int = 3, beta: float = 1.0):
        super().__init__()
        self.beta    = beta
        self.encoder = Encoder(latent_dim, img_channels)
        self.decoder = Decoder(latent_dim, img_channels)

    # ── reparameterisation trick ───────────────────────────
    @staticmethod
    def reparameterise(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── forward pass ──────────────────────────────────────
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z    = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    # ── deterministic encoding (inference) ────────────────
    def encode(self, x):
        """Return μ (no sampling) — use for the PPO agent."""
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    # ── β-VAE loss ────────────────────────────────────────
    def loss(self, recon, x, mu, logvar):
        """
        Reconstruction  : MSE Loss
        KL divergence   : analytical mean
        Returns         : total_loss, recon_loss, kl_loss  (all scalars)
        """
        # Reconstruction loss (Sum over pixels/channels, Mean over batch)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        
        # KL divergence (Sum over latent dims, Mean over batch)
        kl_loss    = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / x.size(0)
        
        total      = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss
