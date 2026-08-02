# ──────────────────────────────────────────────────────────
# settings.py  ←  CARLA connection & file-system constants
# ──────────────────────────────────────────────────────────

import os

# ── CARLA server ───────────────────────────────────────────
HOST = "localhost"
PORT = 2000
TRAFFIC_MANAGER_PORT = 8000
TIMEOUT = 60.0          # seconds

# ── Map ────────────────────────────────────────────────────
TOWN = "Town10HD_Opt"
SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 0.05   # 20 fps physics tick

# ── Sensor resolution ──────────────────────────────────────
IMG_WIDTH  = 128
IMG_HEIGHT = 128
IMG_FOV    = 90

MAX_LIDAR_POINTS = 256    # ↓ was 1024 — matches PPO_MAX_LIDAR_POINTS in parameters.py

# ── File paths ─────────────────────────────────────────────
BASE_DIR           = r"E:\Carla_PPO_project_the.py"
DATA_DIR           = os.path.join(BASE_DIR, "data", "images_128x128")
CHECKPOINT_DIR     = os.path.join(BASE_DIR, "checkpoints")
DEMONSTRATION_DIR  = os.path.join(BASE_DIR, "demonstrations")
VAE_CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "vae_checkpoint.pth")
PPO_CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pth")

os.makedirs(DATA_DIR,       exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEMONSTRATION_DIR, exist_ok=True)
