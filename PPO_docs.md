# CARLA Autonomous Driving: PPO & VAE

This repository contains the reinforcement learning (RL) pipeline for training an autonomous driving agent in the CARLA simulator. It uses a Variational Autoencoder (VAE) to compress camera images and Proximal Policy Optimization (PPO) to learn driving behavior using both camera data and LiDAR.

---

## How to Run

1. **Start CARLA Simulator:** Ensure `CarlaUE4.exe` is running on your machine (default port `2000`).
2. **Run Training:** Open your terminal and execute:
   ```bash
   python main_runner.py
   ```
   The script will automatically connect to CARLA, spawn the vehicle, and begin the training/driving loop.

---

## Key Files & Structure

| File | Description |
|------|-------------|
| `main_runner.py` | The central orchestrator. Connects to CARLA, handles the driving loop, runs visualization, and triggers PPO updates. |
| `ppo_agent.py` | The "Brain" of the vehicle. Contains PyTorch neural networks for the PPO Actor and Critic, plus the Rollout Buffer. |
| `Simulation/enviroment.py` | Handles all CARLA interactions spawning the car, calculating rewards, tracking collisions, managing waypoints. |
| `encoder.py` | Contains the VAE architecture that compresses RGB camera images into a compact feature vector. |
| `settings.py` | Configuration for CARLA (host, port, map, camera dimensions, file paths). |
| `parameters.py` | Core training hyperparameters (learning rates, buffer size, LiDAR points, VAE dimensions). |

### Auxiliary & Imitation Learning Scripts

| File | Description |
|------|-------------|
| `behavioral_cloning.py` | Pre-trains the PPO Actor network using collected autopilot or human demonstrations. |
| `collect_autopilot_data.py` | Collects driving demonstrations using CARLA's built-in autopilot. |
| `collect_human_data.py` | Lets you manually drive and save observations and actions for imitation learning. |
| `audit_ppo_checkpoint.py` | Diagnostic tool to verify neural network shapes match the current architecture. |
| `evaluators/train_vae.py` | Dedicated script to train the VAE on the collected image dataset. |
| `free_drive.py` | Jump into CARLA and drive manually without collecting data or training. |

---

## Checkpoints & Saving

The agent reads and writes progress to single files to prevent clutter:

| File | Description |
|------|-------------|
| `checkpoints/ppo_checkpoint.pth` | Main agent checkpoint. Every run of `main_runner.py` resumes from this file. |
| `checkpoints/vae_checkpoint.pth` | Pre-trained Vision (VAE) model. |
| `ppo_rewards.csv` | Spreadsheet tracking reward history across all episodes. |

> **Important:** If you change `MAX_LIDAR_POINTS` or VAE dimensions in `settings.py` or `parameters.py`, PyTorch will no longer be able to load the old `ppo_checkpoint.pth`. It will silently restart training from scratch. To resume a previous model, these values must exactly match what they were when that model was trained.

---

## VAE Loss Function

The VAE is trained using a combined loss of two terms:

**Total Loss:**

```
L_VAE = L_recon + beta * L_KL
```

**Reconstruction Loss** — Mean Squared Error between the input image `x` and the reconstructed image `x_hat`:

```
L_recon = mean( (x - x_hat)^2 )
```

**KL Divergence** — Ensures the learned latent distribution is close to a standard normal distribution N(0, I):

```
L_KL = -0.5 * sum( 1 + log(sigma^2) - mu^2 - sigma^2 )
```

**Beta parameter** — Controls the trade-off between reconstruction quality and latent space regularity:

| Beta value | Effect |
|------------|--------|
| Low (0.1 - 0.5) | Sharper reconstructions, less structured latent space |
| Medium (0.5 - 1.0) | Balanced reconstruction and regularity |
| High (> 1.0) | More disentangled latent space, blurrier reconstructions |

In this project, `beta = 0.5` was used to prioritise reconstruction sharpness while maintaining a well-structured latent space.

---

## Reparameterization Trick

Instead of sampling `z` directly from the learned distribution (which is not differentiable), the latent vector is computed as:

```
z = mu + sigma * epsilon
where epsilon ~ N(0, I)
```

This separates the randomness (`epsilon`) from the learnable parameters (`mu` and `sigma`), allowing gradients to flow through the encoder during backpropagation.

---

## PPO Clipped Objective

The core PPO update uses a clipped surrogate objective to prevent destructively large policy updates:

```
L_CLIP = E[ min( r(theta) * A, clip(r(theta), 1-epsilon, 1+epsilon) * A ) ]
```

Where:
- `r(theta) = pi_new(a|s) / pi_old(a|s)` ratio of new to old policy probabilities
- `A` Advantage estimate (how much better this action was than expected)
- `epsilon` Clipping range (typically 0.2), preventing updates that change the policy too drastically

---

## Documentation

For a complete breakdown of the reward functions and network architectures, please refer to `CARLA_PPO_Documentation.md`.
