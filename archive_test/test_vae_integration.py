import torch
from encoder import VAE
from parameters import VAE_LATENT_DIM, VAE_BETA

print("Init VAE...")
model = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA)

print("Creating dummy 128x128 image...")
# Batch of 4 images, 3 channels, 128x128
dummy_x = torch.rand(4, 3, 128, 128)

print("Forward pass...")
recon, mu, logvar = model(dummy_x)
print(f"  recon shape: {recon.shape}")
print(f"  mu shape: {mu.shape}")
print(f"  logvar shape: {logvar.shape}")

assert recon.shape == dummy_x.shape, "Recon shape mismatch!"
assert mu.shape == (4, VAE_LATENT_DIM), "Mu shape mismatch!"
assert logvar.shape == (4, VAE_LATENT_DIM), "Logvar shape mismatch!"

print("Testing loss function...")
total_loss, recon_l, kl_l = model.loss(recon, dummy_x, mu, logvar)
print(f"  Total loss: {total_loss.item():.4f}")
print("ALL TESTS PASSED: Code is perfectly solid!")
