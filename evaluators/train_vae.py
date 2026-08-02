# ──────────────────────────────────────────────────────────
# train_vae.py  ←  VAE training loop
# ──────────────────────────────────────────────────────────
#
#  Run from the project root:
#      python evaluators/train_vae.py
#
#  For Google Colab, copy this file + encoder.py + parameters.py
#  into your Colab working directory, then run:
#      !python train_vae.py
# ──────────────────────────────────────────────────────────

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from encoder    import VAE
from parameters import (VAE_LATENT_DIM, VAE_BETA, VAE_BATCH_SIZE,
                         VAE_EPOCHS, VAE_LR, VAE_WEIGHT_DECAY, VAE_VAL_SPLIT)
from settings   import DATA_DIR, VAE_CHECKPOINT, IMG_WIDTH, IMG_HEIGHT


# ── Device ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────
class CarlaImageDataset(Dataset):
    """Loads 128x128 PNG frames collected by data_collector.py."""

    def __init__(self, image_dir: str):
        self.paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"No PNG images found in {image_dir}")
        print(f"Found {len(self.paths)} images.")
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            # --- Data Augmentation to improve VAE robustness ---
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),          # → [0, 1]  shape (C, H, W)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Helpers ───────────────────────────────────────────────
def save_checkpoint(model, epoch, val_loss, best_val: float):
    state = {
        "epoch"      : epoch,
        "val_loss"   : val_loss,
        "best_val"   : best_val,
        "latent_dim" : VAE_LATENT_DIM,
        "beta"       : VAE_BETA,
        "state_dict" : model.state_dict(),
    }
    torch.save(state, VAE_CHECKPOINT)
    print(f"  ✅ Checkpoint saved  →  {VAE_CHECKPOINT}")


def load_checkpoint(model):
    if os.path.exists(VAE_CHECKPOINT):
        state = torch.load(VAE_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["state_dict"])
        epoch = int(state["epoch"])
        best_val = float(state.get("best_val", state["val_loss"]))
        print(f"  📂 Resumed from epoch {epoch}  (checkpoint val_loss={state['val_loss']:.4f})", flush=True)
        return epoch, best_val
    return 0, float("inf")


# ── Training ──────────────────────────────────────────────
def train():
    # Dataset + split
    dataset  = CarlaImageDataset(DATA_DIR)
    val_size  = max(1, int(len(dataset) * VAE_VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=VAE_BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=VAE_BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=VAE_WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    start_epoch, best_val = load_checkpoint(model)

    if start_epoch >= VAE_EPOCHS:
        print(
            f"\n⚠️  Checkpoint is already at epoch {start_epoch} (VAE_EPOCHS={VAE_EPOCHS}). "
            "Raise VAE_EPOCHS in parameters.py to train further, or remove/move the checkpoint "
            "to train from scratch.\n",
            flush=True,
        )
        return

    print(f"\n🚀 Starting VAE training  |  epochs={VAE_EPOCHS}  |  latent_dim={VAE_LATENT_DIM}\n", flush=True)

    for epoch in range(start_epoch + 1, VAE_EPOCHS + 1):
        # ── train ──
        model.train()
        t_loss = t_recon = t_kl = 0.0
        
        # Add tqdm for live progress tracking
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{VAE_EPOCHS}")
        for batch in pbar:
            x = batch.to(DEVICE)
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = model.loss(recon, x, mu, logvar)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            t_loss  += loss.item()
            t_recon += recon_l.item()
            t_kl    += kl_l.item()
            
            # Update the progress bar with the current batch's loss
            pbar.set_postfix({'loss': f"{loss.item():.2f}"})

        n = len(train_loader)

        # ── validate ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(DEVICE)
                recon, mu, logvar = model(x)
                loss, _, _ = model.loss(recon, x, mu, logvar)
                v_loss += loss.item()

        v_loss /= len(val_loader)
        sched.step(v_loss)

        print(f"Epoch [{epoch:3d}/{VAE_EPOCHS}]  "
              f"train={t_loss/n:.2f}  "
              f"recon={t_recon/n:.2f}  "
              f"kl={t_kl/n:.4f}  "
              f"val={v_loss:.2f}", flush=True)

        if v_loss < best_val:
            best_val = v_loss
            save_checkpoint(model, epoch, v_loss, best_val)

    print(f"\n✅ Training complete.  Best val loss: {best_val:.4f}")
    print(f"   Checkpoint saved to: {VAE_CHECKPOINT}")


if __name__ == "__main__":
    train()
