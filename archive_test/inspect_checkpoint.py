import torch

ckpt = torch.load(
    r"E:\Carla_PPO_project_the.py\checkpoints\vae_checkpoint.pth",
    map_location="cpu"
)
sd = ckpt["state_dict"]
print("=== ALL KEYS + SHAPES ===")
for k, v in sd.items():
    print(f"  {k:50s}  {tuple(v.shape)}")
