# Run from project root:
#   python evaluators/show_vae_reconstruction.py
# Optional: --num 8 --out path.png --show  (opens image on Windows)

import argparse
import glob
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from encoder import VAE
from parameters import VAE_LATENT_DIM, VAE_BETA
from settings import CHECKPOINT_DIR, DATA_DIR, IMG_HEIGHT, IMG_WIDTH, VAE_CHECKPOINT


def load_images(paths, device):
    tfm = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(tfm(img))
    batch = torch.stack(tensors, dim=0).to(device)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Save VAE original vs reconstruction grid.")
    parser.add_argument("--num", type=int, default=8, help="Number of images from DATA_DIR")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(CHECKPOINT_DIR, "vae_reconstruction_preview.png"),
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="Open the saved image (Windows)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(VAE_CHECKPOINT):
        print(f"No checkpoint at:\n  {VAE_CHECKPOINT}\nTrain with evaluators/train_vae.py first.")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))[: max(1, args.num)]
    if not paths:
        print(f"No PNGs in:\n  {DATA_DIR}")
        sys.exit(1)

    ckpt = torch.load(VAE_CHECKPOINT, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("latent_dim", VAE_LATENT_DIM))
    beta = float(ckpt.get("beta", VAE_BETA))
    model = VAE(latent_dim=latent_dim, beta=beta).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = load_images(paths, device)
    with torch.no_grad():
        z = model.encode(x)
        recon = model.decoder(z)

    mse = torch.nn.functional.mse_loss(recon, x, reduction="mean").item()
    # Widen each sample: [original | reconstruction]
    pairs = torch.cat([x, recon], dim=3)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_image(pairs, args.out, nrow=min(4, pairs.size(0)))

    print(
        f"Saved {pairs.size(0)} pairs (original | recon) →\n  {args.out}\n"
        f"  mean MSE (mu-decoded recon vs input): {mse:.6f}\n"
        f"  checkpoint epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}"
    )

    if args.show:
        try:
            os.startfile(args.out)  # noqa: SLF001 — Windows only
        except OSError as e:
            print(f"Could not open viewer: {e}")


if __name__ == "__main__":
    main()
