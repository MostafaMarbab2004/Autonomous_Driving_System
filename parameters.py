# ──────────────────────────────────────────────────────────
# parameters.py  - all training hyperparameters
# ──────────────────────────────────────────────────────────

# ── VAE 
VAE_LATENT_DIM   = 128       # z-vector size
VAE_BETA         = 0.5       # Lower beta to force the network to reconstruct cars and fine details!

VAE_BATCH_SIZE   = 64
VAE_EPOCHS       = 250       # raise further if resuming from an older checkpoint (must be > saved epoch)
VAE_LR           = 1e-4
VAE_WEIGHT_DECAY = 1e-5
VAE_VAL_SPLIT    = 0.1      # fraction of data used for validation

# ── PPO 
PPO_LATENT_DIM   = VAE_LATENT_DIM   # must match VAE
PPO_STATE_DIM    = 6                # [speed/50, steer, dist_wp, cos_hdg, sin_hdg, lane_offset] from env._get_obs()
PPO_LIDAR_FEAT_DIM   = 64              # Size of extracted LiDAR feature vector
PPO_MAX_LIDAR_POINTS = 256             # was 1024 — 4× smaller obs, ~16× faster update
PPO_ACTOR_HIDDEN  = 256
PPO_CRITIC_HIDDEN = 256

# continuous action space:  [steer, accel]  (accel > 0 = throttle, < 0 = brake)
PPO_ACTION_DIM   = 2

PPO_LR_ACTOR     = 1e-4
PPO_LR_CRITIC    = 3e-4
PPO_GAMMA        = 0.99
PPO_GAE_LAMBDA   = 0.95
PPO_CLIP_EPS     = 0.2
PPO_ENTROPY_COEF = 0.04   # higher exploration pressure to prevent policy collapse
PPO_VALUE_COEF   = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Training loop
PPO_STEPS_PER_UPDATE  = 2048    # raised from 256 — more reward signal per update (richer batches)
PPO_EPOCHS_PER_UPDATE = 10     # gradient epochs on each batch
PPO_MINIBATCH_SIZE    = 64
PPO_MAX_EPISODES      = 5000
PPO_SAVE_EVERY        = 50     # episodes
PPO_MAX_STEPS_PER_EP  = 500   # hard stop per episode (prevents infinite episodes)
