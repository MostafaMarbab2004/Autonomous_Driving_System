import sys
sys.path.append(r'E:\Carla_PPO_project_the.py')

import carla
import numpy as np
import random
import time
import os
from PIL import Image

# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────
HOST             = "localhost"
PORT             = 2000
SAVE_PATH        = r'E:\Carla_PPO_project_the.py\data\images_128x128'
TARGET_IMAGES    = 30000
NUM_VEHICLES     = 20   # number of traffic vehicles
NUM_PEDESTRIANS  = 15   # number of pedestrians

# ── Town & Weather ────────────────────────────────────────
# Towns:   Town01  Town02  Town03  Town04  Town05  Town10HD_Opt
TOWN    = None   # set to e.g. "Town05" to force-load, or None to use current

# Weather presets (pick one and uncomment):
# carla.WeatherParameters.ClearNoon
# carla.WeatherParameters.CloudyNoon
# carla.WeatherParameters.WetNoon
# carla.WeatherParameters.WetCloudyNoon
# carla.WeatherParameters.SoftRainNoon
# carla.WeatherParameters.HardRainNoon
# carla.WeatherParameters.ClearSunset
# carla.WeatherParameters.WetCloudySunset
# carla.WeatherParameters.HardRainSunset
WEATHER = carla.WeatherParameters.ClearNoon   # ← change this line

os.makedirs(SAVE_PATH, exist_ok=True)

# Count existing images
existing    = len([f for f in os.listdir(SAVE_PATH) if f.endswith('.png')])
image_count = [existing]
print(f"Found {existing} existing images, continuing from {existing}...")

# ─────────────────────────────────────────────
# Connect
# ─────────────────────────────────────────────
print("Connecting to CARLA...")
client = carla.Client(HOST, PORT)
client.set_timeout(60.0)

if TOWN is not None:
    print(f"Loading map: {TOWN}...")
    world = client.load_world(TOWN)
else:
    world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode    = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

world.set_weather(WEATHER)
print(f"Connected — async mode confirmed | weather set")

# ─────────────────────────────────────────────
# Traffic Manager
# ─────────────────────────────────────────────
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_global_distance_to_leading_vehicle(2.0)
traffic_manager.set_synchronous_mode(False)
traffic_manager.set_random_device_seed(42)  # reproducible traffic

bp_lib = world.get_blueprint_library()

# ─────────────────────────────────────────────
# Spawn traffic vehicles
# ─────────────────────────────────────────────
vehicle_actors = []

def spawn_traffic():
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle_bps = bp_lib.filter('vehicle.*')
    # exclude bikes and motorcycles for stability
    vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

    count = 0
    for spawn_point in spawn_points[:NUM_VEHICLES + 10]:
        if count >= NUM_VEHICLES:
            break
        try:
            bp      = random.choice(vehicle_bps)
            vehicle = world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, 8000)
            vehicle_actors.append(vehicle)
            count += 1
        except:
            continue

    print(f" Spawned {len(vehicle_actors)} traffic vehicles")

# ─────────────────────────────────────────────
# Spawn pedestrians
# ─────────────────────────────────────────────
pedestrian_actors      = []
pedestrian_controllers = []

def spawn_pedestrians():
    pedestrian_bps = bp_lib.filter('walker.pedestrian.*')
    controller_bp  = bp_lib.find('controller.ai.walker')

    for _ in range(NUM_PEDESTRIANS):
        try:
            # Random location on sidewalk
            spawn_point       = carla.Transform()
            spawn_point.location = world.get_random_location_from_navigation()

            if spawn_point.location is None:
                continue

            # Spawn pedestrian
            bp          = random.choice(pedestrian_bps)
            pedestrian  = world.spawn_actor(bp, spawn_point)
            pedestrian_actors.append(pedestrian)

            # Spawn AI controller for pedestrian
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
            pedestrian_controllers.append(controller)

        except:
            continue

    # Start all pedestrian controllers
    world.wait_for_tick()
    for controller in pedestrian_controllers:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(random.uniform(1.0, 2.5))

    print(f"Spawned {len(pedestrian_actors)} pedestrians")

# ─────────────────────────────────────────────
# Spawn ego vehicle
# ─────────────────────────────────────────────
spawn_points = world.get_map().get_spawn_points()
spawn_point  = random.choice(spawn_points)
vehicle_bp   = bp_lib.filter('vehicle.tesla.model3')[0]
ego_vehicle  = world.spawn_actor(vehicle_bp, spawn_point)
time.sleep(1.0)

ego_vehicle.set_autopilot(True, 8000)
print(" Ego vehicle spawned with autopilot ON")

# ─────────────────────────────────────────────
# Spawn traffic and pedestrians
# ─────────────────────────────────────────────
spawn_traffic()
spawn_pedestrians()
time.sleep(2.0)  # let everyone settle

# ─────────────────────────────────────────────
# Spectator
# ─────────────────────────────────────────────
spectator = world.get_spectator()

def update_spectator():
    transform = ego_vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=30, x=-10),
        carla.Rotation(pitch=-45)
    ))

# ─────────────────────────────────────────────
# Camera callback
# ─────────────────────────────────────────────
def on_frame(image):
    if image_count[0] >= TARGET_IMAGES:
        return
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((128, 128, 4))[:, :, :3].copy()
        img   = Image.fromarray(array)
        img.save(os.path.join(SAVE_PATH, f"{image_count[0]:05d}.png"))
        image_count[0] += 1

        if image_count[0] % 500 == 0:
            print(f" {image_count[0]}/{TARGET_IMAGES} images collected")
    except Exception as e:
        print(f"Frame error: {e}")

# ─────────────────────────────────────────────
# Attach camera
# ─────────────────────────────────────────────
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '128')
camera_bp.set_attribute('image_size_y', '128')
camera_bp.set_attribute('fov',          '90')
camera_bp.set_attribute('sensor_tick',  '0.1')

transform = carla.Transform(carla.Location(x=2.0, z=1.4))
camera    = world.spawn_actor(camera_bp, transform, attach_to=ego_vehicle)
camera.listen(on_frame)
time.sleep(1.0)
print("Camera attached")

# ─────────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────────
print(f"\nCollecting images {existing} → {TARGET_IMAGES} with traffic & pedestrians...\n")

try:
    while image_count[0] < TARGET_IMAGES:
        update_spectator()
        time.sleep(0.1)

finally:
    print("\nCleaning up...")

    # Stop and destroy pedestrian controllers first
    for controller in pedestrian_controllers:
        if controller.is_alive:
            controller.stop()
            controller.destroy()

    # Destroy pedestrians
    for pedestrian in pedestrian_actors:
        if pedestrian.is_alive:
            pedestrian.destroy()

    # Destroy traffic vehicles
    for vehicle in vehicle_actors:
        if vehicle.is_alive:
            vehicle.destroy()

    # Destroy camera and ego vehicle
    camera.stop()
    camera.destroy()
    ego_vehicle.destroy()

    print(f"Done! {image_count[0]} images saved to {SAVE_PATH}")

