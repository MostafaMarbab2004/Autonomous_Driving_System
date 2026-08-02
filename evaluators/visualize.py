import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from encoder import VAE
from parameters import VAE_LATENT_DIM, VAE_BETA
from settings import DATA_DIR, VAE_CHECKPOINT, IMG_WIDTH, IMG_HEIGHT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    print(f"Loading VAE from {VAE_CHECKPOINT}")
    model = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    if os.path.exists(VAE_CHECKPOINT):
        ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Model checkpoint not found!")
        return

    # Find images
    import glob
    paths = glob.glob(os.path.join(DATA_DIR, "*.png"))
    if not paths:
        print("No images found to visualize.")
        return
        
    # Select 6 random images
    sampled_paths = random.sample(paths, 6)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    originals = []
    reconstructions = []
    
    with torch.no_grad():
        for path in sampled_paths:
            img_pil = Image.open(path).convert("RGB")
            img_t = transform(img_pil).unsqueeze(0).to(DEVICE) # (1, 3, H, W)
            
            recon, _, _ = model(img_t)
            
            # Convert back to images for plotting
            orig_np = img_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
            recon_np = recon.squeeze(0).cpu().permute(1, 2, 0).numpy()
            
            originals.append(orig_np)
            reconstructions.append(recon_np)
            
    # Plotting
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle('VAE Reconstruction Results\nTop: Original | Bottom: Reconstructed', fontsize=16)
    
    for i in range(6):
        axes[0, i].imshow(originals[i])
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis('off')
        
    out_path = r"C:\Users\mosta\.gemini\antigravity\brain\cefa6035-c623-4311-9e9b-2510545f6876\artifacts\vae_visualize_new.png"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize()
