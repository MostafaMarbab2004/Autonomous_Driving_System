# ──────────────────────────────────────────────────────────
# evaluate_agent.py  ←  Run the trained PPO agent (NO training)
# ──────────────────────────────────────────────────────────
#
#  What this script does:
#  ─────────────────────
#  1. Loads the saved VAE + PPO checkpoints (read-only)
#  2. Runs the agent in DETERMINISTIC mode (uses policy mean, no noise)
#  3. No buffer storing, zero gradient updates — pure inference
#  4. Shows the live camera + LiDAR window
#  5. Prints per-episode stats and saves them to eval_results.csv
#
#  Run:
#      python evaluate_agent.py
#
#  Optional flags (edit the CONFIG block below):
#      NUM_EPISODES     – how many episodes to evaluate
#      STOCHASTIC       – set True to sample from policy instead of using mean
#      SAVE_VIDEO       – set True to save a video of each episode (requires OpenCV)
# ──────────────────────────────────────────────────────────

import sys
sys.path.append(r'E:\Carla_PPO_project_the.py')

import os
import time
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms

from Simulation.connection import ClientConnection
from Simulation.enviroment  import CarlaEnvironment
from encoder                import VAE
from ppo_agent              import PPOAgent, DEVICE
from parameters import (
    VAE_LATENT_DIM, VAE_BETA,
    PPO_MAX_STEPS_PER_EP,
)
from settings import (
    HOST, PORT, TOWN, SYNCHRONOUS_MODE,
    IMG_WIDTH, IMG_HEIGHT,
    MAX_LIDAR_POINTS,
    VAE_CHECKPOINT, PPO_CHECKPOINT,
)

# ══════════════════════════════════════════════════════════
#   CONFIG — edit these to customise the evaluation run
# ══════════════════════════════════════════════════════════
NUM_EPISODES  = 20        # number of episodes to evaluate
STOCHASTIC    = False     # False = deterministic (policy mean) | True = stochastic
SAVE_VIDEO    = False     # True = write one .avi file per episode
RESULTS_CSV   = os.path.join(r'E:\Carla_PPO_project_the.py', 'eval_results.csv')
VIDEO_DIR     = os.path.join(r'E:\Carla_PPO_project_the.py', 'eval_videos')
# ══════════════════════════════════════════════════════════

_to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])


def obs_to_tensors(obs):
    rgb    = obs["rgb"]
    img_np = (rgb * 255).astype(np.uint8)
    img_t  = _to_tensor(img_np).unsqueeze(0).to(DEVICE)
    state  = obs["state"]
    lidar  = obs["lidar"]
    return img_t, state, lidar


def load_vae():
    if not os.path.exists(VAE_CHECKPOINT):
        raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CHECKPOINT}")
    vae = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    print(f"✅ VAE loaded  (latent_dim={ckpt['latent_dim']}, epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return vae


def load_agent():
    if not os.path.exists(PPO_CHECKPOINT):
        raise FileNotFoundError(
            f"PPO checkpoint not found: {PPO_CHECKPOINT}\n"
            "You need to train first before evaluating."
        )
    agent = PPOAgent()
    agent.load(PPO_CHECKPOINT)
    # Freeze weights — no gradients needed during evaluation
    for param in agent.actor.parameters():
        param.requires_grad_(False)
    for param in agent.critic.parameters():
        param.requires_grad_(False)
    agent.actor.eval()
    agent.critic.eval()
    print(f"✅ PPO checkpoint loaded from: {PPO_CHECKPOINT}")
    print(f"   Mode: {'STOCHASTIC (sampling)' if STOCHASTIC else 'DETERMINISTIC (policy mean)'}")
    return agent


def build_display_frame(cam_img_bgr, lidar_pts, action, info, ep_reward, ep_steps, episode):
    """Build the side-by-side camera + LiDAR display frame."""
    # ── Camera panel ────────────────────────────────────────
    cam = cv2.resize(cam_img_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX

    mode_label = "STOCHASTIC" if STOCHASTIC else "DETERMINISTIC"
    cv2.putText(cam, f"[EVAL] {mode_label}", (10, 25),  font, 0.6, (0, 200, 255), 2)
    cv2.putText(cam, f"Episode:  {episode}/{NUM_EPISODES}",   (10, 55),  font, 0.6, (255, 255, 255), 1)
    cv2.putText(cam, f"Step:     {ep_steps}/{PPO_MAX_STEPS_PER_EP}", (10, 80),  font, 0.6, (200, 200, 0), 1)
    cv2.putText(cam, f"Speed:    {info.get('speed', 0):.1f} km/h",   (10, 110), font, 0.7, (0, 255, 0),  2)
    cv2.putText(cam, f"Reward:   {ep_reward:.1f}",               (10, 140), font, 0.7, (0, 255, 0),  2)
    cv2.putText(cam, f"WP:       {info.get('waypoint_idx',0)}/{info.get('total_waypoints',0)}", (10, 170), font, 0.6, (255, 200, 0), 1)
    cv2.putText(cam, f"Steer:    {action[0]:+.3f}",              (10, 210), font, 0.6, (180, 180, 180), 1)
    cv2.putText(cam, f"Throttle: {action[1]:.3f}",               (10, 235), font, 0.6, (100, 255, 100), 1)
    cv2.putText(cam, f"Brake:    {action[2]:.3f}",               (10, 260), font, 0.6, (100, 100, 255), 1)

    # ── LiDAR top-down panel ─────────────────────────────────
    lidar_img = np.zeros((512, 512, 3), dtype=np.uint8)
    for p in lidar_pts:
        if abs(p[0]) > 0.01 or abs(p[1]) > 0.01:
            px = int(p[1] * 8 + 256)
            py = int(-p[0] * 8 + 256)
            if 0 <= px < 512 and 0 <= py < 512:
                cv2.circle(lidar_img, (px, py), 1, (0, 0, 255), -1)
    cv2.circle(lidar_img, (256, 256), 5, (0, 255, 0), -1)   # ego vehicle
    cv2.putText(lidar_img, "LiDAR Top-Down", (10, 30), font, 0.7, (255, 255, 255), 2)

    return np.hstack((cam, lidar_img))


def evaluate():
    # ── Connect to CARLA ─────────────────────────────────────
    print("Connecting to CARLA…")
    conn = ClientConnection(HOST, PORT)
    result = conn.setup(
        synchronous_mode="yes" if SYNCHRONOUS_MODE else "no",
        town_name=TOWN,
    )
    if result is None:
        print("❌ Could not connect to CARLA. Is CarlaUE4.exe running?")
        return
    client, world = result

    env = CarlaEnvironment(client, world,
                           img_width=IMG_WIDTH, img_hight=IMG_HEIGHT,
                           max_lidar_points=MAX_LIDAR_POINTS)

    # ── Load models (frozen, eval mode) ──────────────────────
    vae   = load_vae()
    agent = load_agent()

    # ── Prepare results CSV ───────────────────────────────────
    if SAVE_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)

    csv_exists = os.path.exists(RESULTS_CSV)
    csv_file   = open(RESULTS_CSV, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "episode", "total_reward", "steps",
            "avg_speed_kmh", "max_speed_kmh",
            "waypoints_reached", "total_waypoints",
            "collided", "mode"
        ])

    # ── Stats accumulators ────────────────────────────────────
    all_rewards  = []
    all_wps      = []
    collision_count = 0

    print(f"\n{'='*60}")
    print(f"  EVALUATION  —  {NUM_EPISODES} episodes")
    print(f"  Mode: {'STOCHASTIC' if STOCHASTIC else 'DETERMINISTIC'}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    video_writer = None

    try:
        for episode in range(1, NUM_EPISODES + 1):
            obs      = env.reset()
            ep_reward = 0.0
            done      = False
            ep_steps  = 0
            speeds    = []

            if SAVE_VIDEO:
                vid_path = os.path.join(VIDEO_DIR, f"eval_ep{episode:03d}.avi")
                video_writer = cv2.VideoWriter(
                    vid_path,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    20, (1024, 512)
                )

            while not done:
                # ── Encode observation ────────────────────────
                img_t, state, lidar = obs_to_tensors(obs)
                with torch.no_grad():
                    z = vae.encode(img_t)   # (1, latent_dim)

                obs_t = PPOAgent.make_obs(z, state, lidar)  # (1, input_dim)

                # ── Select action (NO buffer, NO gradients) ───
                # deterministic=True  → uses policy mean (no exploration noise)
                # deterministic=False → samples from Gaussian (stochastic)
                action, _raw, _log_prob, _value = agent.select_action(
                    obs_t, deterministic=(not STOCHASTIC)
                )

                # ── Step environment ──────────────────────────
                next_obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_steps  += 1
                speeds.append(info.get("speed", 0.0))

                if ep_steps >= PPO_MAX_STEPS_PER_EP:
                    done = True

                # ────────────────── NO buffer.store() ──────────
                # ────────────────── NO agent.update()  ──────────

                # ── Visualisation ─────────────────────────────
                cam_bgr = cv2.cvtColor(
                    (next_obs["rgb"] * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )
                frame = build_display_frame(
                    cam_bgr, next_obs["lidar"],
                    action, info, ep_reward, ep_steps, episode
                )
                cv2.imshow("Agent Evaluation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                if SAVE_VIDEO and video_writer:
                    video_writer.write(frame)

                obs = next_obs

            # ── Episode summary ───────────────────────────────
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            max_speed = float(np.max(speeds))  if speeds else 0.0
            wp_reached = info.get("waypoint_idx", 0)
            total_wp   = info.get("total_waypoints", 0)
            collided   = bool(env.collision.has_collided)

            if collided:
                collision_count += 1
            all_rewards.append(ep_reward)
            all_wps.append(wp_reached)

            status = "💥 CRASH" if collided else "✅ OK   "
            print(
                f"Ep {episode:>3d}/{NUM_EPISODES}  |  "
                f"reward={ep_reward:8.1f}  |  "
                f"steps={ep_steps:>4d}  |  "
                f"avg_speed={avg_speed:5.1f} km/h  |  "
                f"WP={wp_reached:>3d}/{total_wp}  |  "
                f"{status}"
            )

            # Write to CSV
            csv_writer.writerow([
                episode, f"{ep_reward:.2f}", ep_steps,
                f"{avg_speed:.2f}", f"{max_speed:.2f}",
                wp_reached, total_wp,
                int(collided), "stochastic" if STOCHASTIC else "deterministic"
            ])
            csv_file.flush()

            if SAVE_VIDEO and video_writer:
                video_writer.release()
                print(f"   📹 Video saved → {vid_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")

    finally:
        csv_file.close()
        cv2.destroyAllWindows()
        env.close()

        # ── Final summary ─────────────────────────────────────
        if all_rewards:
            print(f"\n{'='*60}")
            print(f"  EVALUATION SUMMARY  ({len(all_rewards)} episodes)")
            print(f"{'='*60}")
            print(f"  Avg reward     : {np.mean(all_rewards):8.1f}  ± {np.std(all_rewards):.1f}")
            print(f"  Best reward    : {np.max(all_rewards):8.1f}")
            print(f"  Worst reward   : {np.min(all_rewards):8.1f}")
            print(f"  Avg waypoints  : {np.mean(all_wps):.1f}")
            print(f"  Collision rate : {collision_count}/{len(all_rewards)} episodes "
                  f"({100*collision_count/len(all_rewards):.0f}%)")
            print(f"  Results CSV    : {RESULTS_CSV}")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluate()
