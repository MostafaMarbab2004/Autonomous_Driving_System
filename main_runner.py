# ──────────────────────────────────────────────────────────
# main_runner.py  ←  PPO training loop inside CARLA
# ──────────────────────────────────────────────────────────
#
#  Prerequisites
#  ─────────────
#  1. CARLA server running  (CarlaUE4.exe)
#  2. vae_checkpoint.pth in  E:\Carla_PPO_project_the.py\checkpoints\
#     (download from Colab after train_vae.py finishes)
#
#  Run:
#      python main_runner.py
# ──────────────────────────────────────────────────────────

import sys
sys.path.append(r'E:\Carla_PPO_project_the.py')

import os
import time
import cv2
import numpy as np
import torch
import math
from torchvision import transforms

from Simulation.connection  import ClientConnection
from Simulation.enviroment  import CarlaEnvironment
from encoder                import VAE
from ppo_agent              import PPOAgent, DEVICE
from parameters import (
    VAE_LATENT_DIM, VAE_BETA,
    PPO_STEPS_PER_UPDATE, PPO_MAX_EPISODES, PPO_SAVE_EVERY,
    PPO_MAX_STEPS_PER_EP,
)
from settings import (
    HOST, PORT, TOWN, SYNCHRONOUS_MODE,
    IMG_WIDTH, IMG_HEIGHT,
    MAX_LIDAR_POINTS,
    VAE_CHECKPOINT, PPO_CHECKPOINT,
)


# ── Image pre-processing (same as train_vae.py) ───────────
_to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),          # → (C, H, W) in [0, 1]
])


def obs_to_tensors(obs):
    """
    Convert CarlaEnvironment._get_obs() dict to tensors on DEVICE,
    and also return the raw lidar numpy array.
    """
    rgb    = obs["rgb"]                                  # (H, W, 3) float32 [0,1]
    img_np = (rgb * 255).astype(np.uint8)               # PIL needs uint8
    img_t  = _to_tensor(img_np).unsqueeze(0).to(DEVICE) # (1, 3, H, W)
    state  = obs["state"]                               # numpy (PPO_STATE_DIM,)
    lidar  = obs["lidar"]                               # numpy (MAX_LIDAR_POINTS, 3)
    return img_t, state, lidar


# ── Load VAE ──────────────────────────────────────────────
def load_vae():
    vae = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    if not os.path.exists(VAE_CHECKPOINT):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {VAE_CHECKPOINT}\n"
            "Train the VAE on Colab first, then download vae_checkpoint.pth."
        )
    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    print(f"VAE loaded  (latent_dim={ckpt['latent_dim']}, "
          f"val_loss={ckpt['val_loss']:.4f}, epoch={ckpt['epoch']})")
    return vae


# ── Training loop ─────────────────────────────────────────
def train():
    # ── Connect to CARLA ──────────────────────────────────
    print("Connecting to CARLA…")
    conn = ClientConnection(HOST, PORT)
    result = conn.setup(
        synchronous_mode="yes" if SYNCHRONOUS_MODE else "no",
        town_name=TOWN,
    )
    if result is None:
        print("Could not connect to CARLA. Is CarlaUE4.exe running?")
        return
    client, world = result

    # ── Build environment ─────────────────────────────────
    env = CarlaEnvironment(client, world,
                           img_width=IMG_WIDTH, img_hight=IMG_HEIGHT,
                           max_lidar_points=MAX_LIDAR_POINTS)

    # ── Load models ───────────────────────────────────────
    vae   = load_vae()
    agent = PPOAgent()
    loaded, _ = agent.load(PPO_CHECKPOINT)

    if loaded:
        print(f"Resuming existing checkpoint: {PPO_CHECKPOINT}")
    else:
        print(f"Starting NEW training from scratch.")
        
    csv_path = "ppo_rewards.csv"
    
    # ── Counters ──────────────────────────────────────────
    total_steps    = 0
    episode_rewards = []

    print(f"\n Starting PPO training |  max_episodes={PPO_MAX_EPISODES}")
    print(f"   Metrics: {csv_path}")
    print(f"   Checkpoint: {PPO_CHECKPOINT}\n")

    try:
        for episode in range(1, PPO_MAX_EPISODES + 1):
            # Curriculum logic: gradually increase waypoints as the agent gets better
            active_wps = min(10 + (episode // 15), 50)
            
            obs       = env.reset(num_waypoints=active_wps)
            ep_reward = 0.0
            done      = False
            ep_steps  = 0

            while not done:
                # 1. encode image to latent z
                img_t, state, lidar = obs_to_tensors(obs)
                with torch.no_grad():
                    z = vae.encode(img_t)                    # (1, latent_dim)
                    vae_recon = vae.decoder(z)               # (1, 3, H, W) deterministic recon from mu

                obs_t = PPOAgent.make_obs(z, state, lidar)   # (1, input_dim)

                # 2. select action
                action, raw, log_prob, value = agent.select_action(obs_t)

                # 3. step environment
                next_obs, reward, done, info = env.step(action)
                ep_reward   += reward
                total_steps += 1
                ep_steps    += 1

                # Hard episode step limit (prevents infinite frozen episodes)
                if ep_steps >= PPO_MAX_STEPS_PER_EP:
                    done = True

                # 4. store in buffer
                agent.buffer.store(
                    obs      = obs_t.squeeze(0),
                    action   = raw,
                    log_prob = log_prob,
                    reward   = reward,
                    value    = value,
                    done     = float(done),
                )

                # -- Live Visualization ------------------------------
                camera_img = next_obs["rgb"]
                cam_show = (camera_img * 255).astype(np.uint8)
                cam_show = cv2.cvtColor(cam_show, cv2.COLOR_RGB2BGR)
                cam_show = cv2.resize(cam_show, (512, 512), interpolation=cv2.INTER_NEAREST)
                
                # -- Render LiDAR Top-Down --
                lidar_img = np.zeros((512, 512, 3), dtype=np.uint8)
                for p in next_obs["lidar"]:
                    if abs(p[0]) > 0.0002 or abs(p[1]) > 0.0002: # LiDAR is now normalized to [-1,1]
                        # Scale up since values are [-1,1] now (was raw meters before)
                        px = int(p[1] * 400 + 256)   # Y (right) maps to image X
                        py = int(-p[0] * 400 + 256)  # X (forward) maps to image Y (up is negative Y)
                        if 0 <= px < 512 and 0 <= py < 512:
                            cv2.circle(lidar_img, (px, py), 1, (0, 0, 255), -1)
                
                # -- Render Waypoints on LiDAR Map --
                if env.waypoints:
                    veh_tf = env.vehicle.get_transform()
                    veh_loc = veh_tf.location
                    veh_yaw = math.radians(veh_tf.rotation.yaw)
                    
                    # Draw all remaining waypoints
                    for i in range(env.current_wp_idx, len(env.waypoints)):
                        wp_loc = env.waypoints[i].transform.location
                        
                        # Manual 2D inverse transform
                        dx = wp_loc.x - veh_loc.x
                        dy = wp_loc.y - veh_loc.y
                        
                        # Rotate based on vehicle yaw
                        lx = dx * math.cos(veh_yaw) + dy * math.sin(veh_yaw)
                        ly = -dx * math.sin(veh_yaw) + dy * math.cos(veh_yaw)
                        
                        lx_norm = lx / 50.0
                        ly_norm = ly / 50.0
                        
                        wpx = int(ly_norm * 400 + 256)
                        wpy = int(-lx_norm * 400 + 256)
                        
                        if 0 <= wpx < 512 and 0 <= wpy < 512:
                            # Current target is green, future are blue
                            c = (0, 255, 0) if i == env.current_wp_idx else (255, 100, 0)
                            cv2.circle(lidar_img, (wpx, wpy), 3, c, -1)

                # Draw ego vehicle dot and LiDAR title
                cv2.circle(lidar_img, (256, 256), 4, (255, 255, 255), -1) # Ego is white dot
                cv2.putText(lidar_img, "LiDAR Top-Down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Collision check from environment directly 
                c_status = "CRASHED" if env.collision.has_collided else "OK"
                c_color = (0, 0, 255) if env.collision.has_collided else (0, 255, 0)
                cv2.putText(lidar_img, f"Collision Sensor: {c_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)

                # -- VAE Reconstruction View --
                recon = vae_recon.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                recon = np.clip(recon, 0.0, 1.0)
                recon_show = (recon * 255).astype(np.uint8)
                recon_show = cv2.cvtColor(recon_show, cv2.COLOR_RGB2BGR)
                recon_show = cv2.resize(recon_show, (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.putText(recon_show, "VAE Reconstruction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
                
                # Overlay Camera Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cam_show, f"Speed: {info.get('speed', 0):.1f} km/h", (10, 30), font, 0.7, (0, 255, 0), 2)
                cv2.putText(cam_show, f"Reward: {ep_reward:.1f}", (10, 60), font, 0.7, (0, 255, 0), 2)
                cv2.putText(cam_show, f"WP: {info.get('waypoint_idx',0)}/{info.get('total_waypoints',0)}", (10, 90), font, 0.7, (0, 255, 0), 2)
                
                cv2.putText(cam_show, f"Steer:    {action[0]:+.2f}", (10, 140), font, 0.6, (200, 200, 200), 1)
                cv2.putText(cam_show, f"Accel:    {action[1]:+.2f}", (10, 170), font, 0.6, (100, 255, 100), 1)
                cv2.putText(cam_show, f"Throttle: {info.get('throttle', 0):.2f}", (10, 200), font, 0.6, (100, 255, 100), 1)
                cv2.putText(cam_show, f"Brake:    {info.get('brake', 0):.2f}", (10, 230), font, 0.6, (100, 100, 255), 1)
                cv2.putText(cam_show, f"Ep step:  {ep_steps}/{PPO_MAX_STEPS_PER_EP}", (10, 260), font, 0.6, (200, 200, 0), 1)
                if info.get("unstuck_boost", False):
                    cv2.putText(cam_show, "UNSTUCK BOOST", (10, 290), font, 0.7, (0, 200, 255), 2)
                
                # Combine camera, LiDAR and VAE recon side by side
                show_img = np.hstack((cam_show, lidar_img, recon_show))
                
                cv2.imshow("Agent View", show_img)
                cv2.waitKey(1)
                # ----------------------------------------------------

                if total_steps % 100 == 0:
                    print(f"   ... Driving progress: step {total_steps:4d} | current ep reward: {ep_reward:5.1f} | speed: {info.get('speed', 0):.1f}km/h")

                obs = next_obs

                # 5. PPO update when buffer is full
                if total_steps % PPO_STEPS_PER_UPDATE == 0:
                    img_t, state, lidar = obs_to_tensors(next_obs)
                    with torch.no_grad():
                        z_last = vae.encode(img_t)
                    last_obs = PPOAgent.make_obs(z_last, state, lidar)
                    agent.update(last_obs)
                    print(f"  [step {total_steps}] PPO update done")

            episode_rewards.append(ep_reward)
            avg   = np.mean(episode_rewards[-20:])
            speed = info.get("speed", 0.0)
            wp    = info.get("waypoint_idx", 0)
            total = info.get("total_waypoints", 0)

            print(f"Episode {episode:4d} ",
                  f"reward={ep_reward:8.1f} ", 
                  f"avg20={avg:8.1f} ",
                  f"speed={speed:.1f} km/h ",
                  f"wp={wp}/{total}")

            # Log to CSV for plotting
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write("episode,reward,avg_reward,speed,waypoint,total_waypoints\n")
            with open(csv_path, "a") as f:
                f.write(f"{episode},{ep_reward},{avg},{speed},{wp},{total}\n")

            if episode % PPO_SAVE_EVERY == 0:
                agent.save(PPO_CHECKPOINT)
                print(f" PPO checkpoint saved → {PPO_CHECKPOINT}")

    except KeyboardInterrupt:
        print("\n Training interrupted by user.")

    finally:
        print("\nCleaning up…")
        agent.save(PPO_CHECKPOINT)
        env.close()
        print("Done.")


if __name__ == "__main__":
    train()
