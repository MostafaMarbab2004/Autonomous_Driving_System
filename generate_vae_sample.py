import os
import random
import torch
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Try to use existing imports from the project
import sys
sys.path.append(r"E:\Carla_PPO_project_the.py")

from encoder import VAE
from parameters import VAE_LATENT_DIM, VAE_BETA
from settings import DATA_DIR, VAE_CHECKPOINT

def generate_sample(num_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE
    vae = VAE(latent_dim=128, beta=VAE_BETA).to(device)
    ckpt = torch.load(VAE_CHECKPOINT, map_location=device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    
    # Get random images
    images = glob.glob(os.path.join(DATA_DIR, "*.png"))
    if not images:
        print(f"No images found in {DATA_DIR}")
        return
    
    sample_paths = random.sample(images, min(num_samples, len(images)))
    
    # Transform (matches train_vae.py)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    fig, axes = plt.subplots(len(sample_paths), 2, figsize=(8, 3 * len(sample_paths)))
    
    # Ensure axes is a 2D array even if num_samples == 1
    if len(sample_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, sample_path in enumerate(sample_paths):
        print(f"Processing image {idx+1}: {sample_path}")
        img = Image.open(sample_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            recon, mu, logvar = vae(x)
            
        # Convert back to images
        orig_img = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
        recon_img = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Clip values
        orig_img = np.clip(orig_img, 0, 1)
        recon_img = np.clip(recon_img, 0, 1)
        
        # Plot
        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title("Original (128x128)")
            
        axes[idx, 1].imshow(recon_img)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title("VAE Reconstruction (128-dim)")
            
    output_path = r"C:\Users\mosta\.gemini\antigravity\brain\0d8c7dc9-da00-4605-9cd5-9243f26a9110\scratch\vae_reconstruction_grid.png"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstruction grid to {output_path}")

if __name__ == "__main__":
    generate_sample(num_samples=4)
