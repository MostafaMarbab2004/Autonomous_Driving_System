import sys
import os
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

from Simulation.connection import ClientConnection
from Simulation.enviroment import CarlaEnvironment
from encoder import VAE
from ppo_agent import PPOAgent, DEVICE
from parameters import VAE_LATENT_DIM, VAE_BETA
from settings import (
    HOST, PORT, TOWN, SYNCHRONOUS_MODE,
    IMG_WIDTH, IMG_HEIGHT,
    MAX_LIDAR_POINTS,
    VAE_CHECKPOINT, PPO_CHECKPOINT,
)

# Image pre-processing (must exactly match training)
_to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

def obs_to_tensors(obs):
    rdf    = obs["rgb"]
    img_np = (rdf * 255).astype(np.uint8)
    img_t  = _to_tensor(img_np).unsqueeze(0).to(DEVICE)
    state  = obs["state"]
    lidar  = obs["lidar"]
    return img_t, state, lidar

def load_models():
    # ── Load VAE ──
    vae = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    if not os.path.exists(VAE_CHECKPOINT):
        print(f"VAE checkpoint not found: {VAE_CHECKPOINT}")
        sys.exit(1)
    
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["state_dict"])
    vae.eval()
    print("VAE Model Loaded")

    # ── Load PPO ──
    agent = PPOAgent()
    if not os.path.exists(PPO_CHECKPOINT):
        print(f"PPO checkpoint not found at: {PPO_CHECKPOINT}")
        print("You must train the agent first before trying to free-drive.")
        sys.exit(1)
        
    agent.load(PPO_CHECKPOINT)
    # Set to evaluation mode (turns off dropout, noise, etc)
    agent.actor.eval()
    agent.critic.eval()
    print("   PPO Agent Models Loaded")
    
    return vae, agent

def free_drive():
    print("\nConnecting to CARLA Server...")
    conn = ClientConnection(HOST, PORT)
    result = conn.setup(
        synchronous_mode="yes" if SYNCHRONOUS_MODE else "no",
        town_name=TOWN,
    )
    if result is None:
        print(" Could not connect to CARLA.")
        return
    client, world = result

    env = CarlaEnvironment(
        client, world,
        img_width=IMG_WIDTH, img_hight=IMG_HEIGHT,
        max_lidar_points=MAX_LIDAR_POINTS
    )

    vae, agent = load_models()

    print("\n Starting Free Drive Mode!")
    print("The agent is driving using its saved brain. No new training is happening.")
    print("Press Ctrl+C in the terminal to stop at any time.\n")

    try:
        while True:
            # We give it a generous 50 waypoints so it can cruise smoothly
            obs = env.reset(num_waypoints=50) 
            done = False
            ep_steps = 0
            
            # Start Driving Loop
            while not done:
                # 1. Look at the environment
                img_t, state, lidar = obs_to_tensors(obs)
                with torch.no_grad():
                    z = vae.encode(img_t)
                
                obs_t = PPOAgent.make_obs(z, state, lidar)
                
                # 2. Decide what to do
                # By passing deterministic=True, the agent avoids random "jitters"
                # and uses its maximum-confidence action, properly applying tanh squashing
                action, _, _, _ = agent.select_action(obs_t, deterministic=True)
                # 3. Take Action
                next_obs, reward, done, info = env.step(action)
                ep_steps += 1
                
                # Force a respawn if it gets hopelessly stuck for hundreds of frames
                if ep_steps >= 2000: 
                    print("Agent was driving for too long in one route. Respawning.")
                    done = True

                # 4. Visualization HUD
                camera_img = next_obs["rgb"]
                cam_show = (camera_img * 255).astype(np.uint8)
                cam_show = cv2.cvtColor(cam_show, cv2.COLOR_RGB2BGR)
                cam_show = cv2.resize(cam_show, (512, 512), interpolation=cv2.INTER_NEAREST)
                
                # Draw Lidar
                lidar_img = np.zeros((512, 512, 3), dtype=np.uint8)
                for p in next_obs["lidar"]:
                    if abs(p[0]) > 0.0002 or abs(p[1]) > 0.0002:  # LiDAR is now normalized to [-1,1]
                        px = int(p[1] * 400 + 256)   # scale up since values are [-1,1] now
                        py = int(-p[0] * 400 + 256)
                        if 0 <= px < 512 and 0 <= py < 512:
                            cv2.circle(lidar_img, (px, py), 1, (0, 0, 255), -1)
                            
                cv2.circle(lidar_img, (256, 256), 4, (0, 255, 0), -1)
                cv2.putText(lidar_img, "LiDAR Top-Down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                c_status = "CRASHED" if env.collision.has_collided else "OK"
                c_color = (0, 0, 255) if env.collision.has_collided else (0, 255, 0)
                cv2.putText(lidar_img, f"Collision: {c_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
                
                # Draw Camera Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cam_show, "FREE DRIVE - AI IN CONTROL", (10, 30), font, 0.8, (255, 255, 0), 2)
                cv2.putText(cam_show, f"Speed: {info.get('speed', 0):.1f} km/h", (10, 60), font, 0.7, (0, 255, 0), 2)
                
                cv2.putText(cam_show, f"Steer:    {action[0]:+.2f}", (10, 110), font, 0.6, (200, 200, 200), 1)
                cv2.putText(cam_show, f"Accel:    {action[1]:+.2f}", (10, 140), font, 0.6, (100, 255, 100), 1)
                cv2.putText(cam_show, f"Throttle: {info.get('throttle', 0):.2f}", (10, 170), font, 0.6, (100, 255, 100), 1)
                cv2.putText(cam_show, f"Brake:    {info.get('brake', 0):.2f}", (10, 200), font, 0.6, (100, 100, 255), 1)
                
                show_img = np.hstack((cam_show, lidar_img))
                cv2.imshow("CARLA AI Agent", show_img)
                cv2.waitKey(1) # Renders the frame visually
                
                obs = next_obs
                
            print("   Route ended, agent got stuck, or collision detected. Spawning in a new area...")

    except KeyboardInterrupt:
        print("\nFree drive session manually stopped.")
    finally:
        print("Cleaning up resources...")
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    free_drive()
