# CARLA Autonomous Driving: PPO & VAE

This repository contains the reinforcement learning (RL) pipeline for training an autonomous driving agent in the CARLA simulator. It uses a Variational Autoencoder (VAE) to compress camera images and Proximal Policy Optimization (PPO) to learn driving behavior using both camera data and LiDAR.

## How to Run

1. **Start CARLA Simulator:** Ensure `CarlaUE4.exe` is running on your machine (default port `2000`).
2. **Run Training:** Open your terminal and execute:
   ```bash
   python main_runner.py
   ```
   The script will automatically connect to CARLA, spawn the vehicle, and begin the training/driving loop.




https://github.com/user-attachments/assets/7a07804e-4026-4dd6-80f3-3d6d1c58b055


   

## Key Files & Structure

*   **`main_runner.py`**: The central orchestrator. It connects to CARLA, handles the driving loop, runs the visualization, and triggers PPO updates.
*   **`ppo_agent.py`**: The "Brain" of the vehicle. Contains the PyTorch neural networks for the PPO Actor and Critic, as well as the Rollout Buffer memory.
*   **`Simulation/enviroment.py`**: Handles all interactions with the CARLA world (spawning the car, calculating rewards, tracking collisions, managing waypoints).
*   **`encoder.py`**: Contains the VAE architecture that compresses RGB camera images down to a small feature vector.
*   **`settings.py`**: Configuration for CARLA (host, port, map, camera dimensions, file paths).
*   **`parameters.py`**: Core mathematical hyper-parameters for training (learning rates, buffer size, number of LiDAR points, VAE dimensions).

### Auxiliary & Imitation Learning Scripts

*   **`behavioral_cloning.py`**: A script to pre-train the PPO agent's Actor network using collected human or autopilot demonstrations.
*   **`collect_autopilot_data.py`**: Automates the collection of driving demonstrations by letting CARLA's built-in autopilot drive the vehicle and saving the data.
*   **`collect_human_data.py`**: Allows you to manually drive the vehicle (via keyboard or steering wheel) and save the observations and actions for imitation learning.
*   **`audit_ppo_checkpoint.py`**: A diagnostic tool to verify that the neural network shapes inside your `ppo_checkpoint.pth` match the current architecture defined in your code.
*   **`evaluators/train_vae.py`**: A dedicated script to train the Variational Autoencoder (VAE) on the collected image dataset.
*   **`free_drive.py`**: A simple script to jump into the CARLA environment and drive around manually without collecting data or training.

##  Checkpoints & Saving

The agent strictly reads and writes its progress to single files to prevent clutter:

*   **`checkpoints/ppo_checkpoint.pth`**: This is your main agent. Every time you run `main_runner.py`, it resumes from this file.
*   **`checkpoints/vae_checkpoint.pth`**: The pre-trained Vision model.
*   **`ppo_rewards.csv`**: A single spreadsheet tracking the reward history of all your episodes.

###  Important Note on Modifying Parameters
If you change certain variables in `settings.py` or `parameters.py` that alter the size of the neural network (such as changing `MAX_LIDAR_POINTS` from `1024` to `256`), **PyTorch will no longer be able to load the old `ppo_checkpoint.pth` file.** It will assume the file is incompatible and will silently restart training from scratch (overwriting `ppo_checkpoint.pth` as it learns).

To resume a previously successful model, the `MAX_LIDAR_POINTS` and VAE dimensions must perfectly match what they were when that model was trained!

##  Documentation
For a complete mathematical breakdown of the reward functions and network architectures, please refer to `PPO_docs.md`.
