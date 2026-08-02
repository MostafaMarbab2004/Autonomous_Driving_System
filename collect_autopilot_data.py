import os
import sys
import glob
import time
import pickle
import math
import numpy as np
import pygame

# --- CARLA Path Discovery ---
try:
    # Look for the CARLA egg in common locations
    carla_path = r'E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
    if os.path.exists(carla_path):
        sys.path.append(carla_path)
    else:
        # Fallback to searching if the exact path is different
        search_pattern = r'E:\CARLA_*\WindowsNoEditor\PythonAPI\carla\dist\carla-*.egg'
        eggs = glob.glob(search_pattern)
        if eggs:
            sys.path.append(eggs[0])
except Exception as e:
    print(f"Warning: CARLA path discovery failed: {e}")

import carla

from Simulation.enviroment import CarlaEnvironment
from settings import IMG_WIDTH, IMG_HEIGHT

# --- CONFIG ---
FPS = 20
SAVE_DIR = "demonstrations"
os.makedirs(SAVE_DIR, exist_ok=True)
WINDOW_SIZE = 400  # Size of the Pygame window

def autopilot_policy(env, obs):
    """
    Computes perfect steering and throttle to follow the env.waypoints.
    This acts as a flawless human driver for data collection.
    """
    state = obs['state']
    speed = state[0] * 50.0  # Denormalize speed to km/h
    sin_h = state[4]         # Heading cross product (positive = right)
    
    # 1. Steering Control (P-Controller)
    # The sharper the angle to the waypoint, the harder it steers
    steer = sin_h * 2.0
    steer = np.clip(steer, -1.0, 1.0)
    
    # 2. Speed Control
    target_speed = 25.0
    
    # Slow down heavily for sharp turns
    if abs(sin_h) > 0.2:
        target_speed = 15.0
    if abs(sin_h) > 0.4:
        target_speed = 10.0
        
    if speed < target_speed:
        accel = 0.65  # Smooth throttle
    else:
        accel = -0.5  # Smooth brake
        
    return np.array([steer, accel], dtype=np.float32)

def render_obs(rgb, display):
    """ Takes the agent's camera view and scales it up for the human to see """
    # rgb is shape (IMG_HEIGHT, IMG_WIDTH, 3) float32 in [0, 1]. Convert to uint8 (0-255)
    img_uint8 = (rgb * 255.0).astype(np.uint8)
    
    # Pygame expects (width, height, channels), numpy is (height, width, channels)
    img_transposed = np.transpose(img_uint8, (1, 0, 2))
    surface = pygame.surfarray.make_surface(img_transposed)
    
    # Scale up for visibility
    scaled_surface = pygame.transform.scale(surface, (WINDOW_SIZE, WINDOW_SIZE))
    display.blit(scaled_surface, (0, 0))
    # Removed flip() here so we can draw minimap on top

def draw_minimap(display, env):
    """ Draws a radar overlay in the top-right corner to show upcoming waypoints """
    center_x = WINDOW_SIZE - 90
    center_y = 90
    radius = 70
    
    # Draw background circle
    surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(surface, (0, 0, 0, 150), (center_x, center_y), radius)
    pygame.draw.circle(surface, (255, 255, 255, 200), (center_x, center_y), radius, 2)
    display.blit(surface, (0, 0))
    
    # Draw car in the center (facing UP)
    pygame.draw.polygon(display, (0, 255, 0), [
        (center_x, center_y - 12),
        (center_x - 6, center_y + 6),
        (center_x + 6, center_y + 6)
    ])
    
    try:
        veh_tf = env.vehicle.get_transform()
        v_loc = veh_tf.location
        yaw = math.radians(veh_tf.rotation.yaw)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        
        start_idx = env.current_wp_idx
        end_idx = min(start_idx + 30, len(env.waypoints))
        
        for i in range(start_idx, end_idx):
            w_loc = env.waypoints[i].transform.location
            
            # Relative vector
            dx = w_loc.x - v_loc.x
            dy = w_loc.y - v_loc.y
            
            # Rotate by car's yaw to get local coordinates (CARLA is left-handed, Z up)
            # In CARLA: +X is forward, +Y is right.
            local_x = dx * cos_y + dy * sin_y
            local_y = -dx * sin_y + dy * cos_y
            
            # Scale to minimap (e.g. 1 meter = 2 pixels)
            scale = 2.0
            
            # Forward (+local_x) maps to UP (-Y on screen)
            # Right (+local_y) maps to RIGHT (+X on screen)
            map_x = int(center_x + local_y * scale)
            map_y = int(center_y - local_x * scale)
            
            # Only draw if inside the circle
            dist_sq = (map_x - center_x)**2 + (map_y - center_y)**2
            if dist_sq < (radius - 5)**2:
                color = (255, 50, 50) if i == start_idx else (200, 200, 200)
                pygame.draw.circle(display, color, (map_x, map_y), 3)
                
    except Exception as e:
        pass

    pygame.display.flip()

def save_episode(data, ep_idx):
    if len(data) < 10:
        return # Ignore super short episodes (e.g. immediate crashes)
        
    filename = os.path.join(SAVE_DIR, f"demo_ep_{ep_idx}_{int(time.time())}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} frames to {filename}")

def main():
    pygame.init()
    display = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("CARLA Imitation Learning - You Drive!")
    clock = pygame.time.Clock()

    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Make sure we use the same parameters the PPO agent expects
    # Note: Using your spelling of img_hight
    env = CarlaEnvironment(client, world, img_width=IMG_WIDTH, img_hight=IMG_HEIGHT, max_lidar_points=256)
    
    print("Resetting environment...")
    obs = env.reset()
    
    episode_data = []
    current_steer = 0.0
    
    # --- Find existing max episode count to prevent naming collisions ---
    episode_count = 0
    if os.path.exists(SAVE_DIR):
        existing_files = glob.glob(os.path.join(SAVE_DIR, "demo_ep_*.pkl"))
        for f in existing_files:
            try:
                # filename format: demo_ep_12_16500000.pkl
                basename = os.path.basename(f)
                idx = int(basename.split('_')[2])
                if idx >= episode_count:
                    episode_count = idx + 1
            except Exception:
                pass
    print(f"Starting data collection at episode {episode_count}...")
    
    print("\n" + "="*60)
    print(" AUTOPILOT DATA COLLECTION ACTIVATED!")
    print(" - The car will now drive itself perfectly along the route.")
    print(" - Sit back and let it collect thousands of demonstrations.")
    print(" - Press 'R' to skip to the next episode.")
    print(" - Press 'Q' or 'ESC' to quit and save.")
    print("="*60 + "\n")
    
    running = True
    while running:
        clock.tick(FPS)
        
        # 1. Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Save current episode and reset
                    save_episode(episode_data, episode_count)
                    episode_count += 1
                    episode_data = []
                    obs = env.reset()
                    current_steer = 0.0
        
        if not running:
            break
            
        # Autopilot takes full control
        action = autopilot_policy(env, obs)
        
        # 2. Render the agent's view so the human can see
        render_obs(obs['rgb'], display)
        
        # 2.5 Draw minimap overlay and flip display
        draw_minimap(display, env)
        
        # 3. Step the environment
        next_obs, reward, done, info = env.step(action)
        
        # 4. Save the transition
        # We record what state the agent SAW, and what action YOU TOOK
        transition = {
            "rgb": obs["rgb"],
            "lidar": obs["lidar"],
            "state": obs["state"],
            "action": action
        }
        episode_data.append(transition)
        
        obs = next_obs
        
        if done:
            print("Episode finished (Crash or Destination). Saving data...")
            save_episode(episode_data, episode_count)
            episode_count += 1
            episode_data = []
            obs = env.reset()
            current_steer = 0.0

    # Save any remaining data on exit
    if len(episode_data) > 0:
        save_episode(episode_data, episode_count)
        
    env.close()
    pygame.quit()
    print("Data collection complete. Good job!")

if __name__ == "__main__":
    main()
