import sys
import os
import glob

# Try to find CARLA egg
try:
    carla_path = r'E:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg'
    if os.path.exists(carla_path):
        sys.path.append(carla_path)
    else:
        eggs = glob.glob(r'E:\CARLA_*\WindowsNoEditor\PythonAPI\carla\dist\carla-*.egg')
        if eggs:
            sys.path.append(eggs[0])
except Exception:
    pass

import carla
import numpy as np

# ─────────────────────────────────────────────
# RGB Camera Sensor
# ─────────────────────────────────────────────
class RGBCamera:
    def __init__(self, world, vehicle, img_width=64, img_height=64, fov=90):
        self.world      = world
        self.vehicle    = vehicle
        self.img_width  = img_width
        self.img_height = img_height
        self.data       = None
        self.sensor     = None

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(img_width))
        blueprint.set_attribute('image_size_y', str(img_height))
        blueprint.set_attribute('fov',          str(fov))

        transform   = carla.Transform(carla.Location(x=2.0, z=1.4))
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_data)

    def _on_data(self, image):
        array     = np.frombuffer(image.raw_data, dtype=np.uint8)
        array     = array.reshape((self.img_height, self.img_width, 4))
        self.data = array[:, :, :3].astype(np.float32) / 255.0

    def get_data(self):
        if self.data is not None:
            return self.data
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
            self.data   = None


# ─────────────────────────────────────────────
# LiDAR Sensor
# ─────────────────────────────────────────────
class LiDAR:
    def __init__(self, world, vehicle, max_points=1024, range=50,
                 channels=32, points_per_second=56000, rotation_frequency=20):
        self.world      = world
        self.vehicle    = vehicle
        self.max_points = max_points
        self.data       = None
        self.sensor     = None

        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('range',              str(range))
        blueprint.set_attribute('channels',           str(channels))
        blueprint.set_attribute('points_per_second',  str(points_per_second))
        blueprint.set_attribute('rotation_frequency', str(rotation_frequency))

        transform   = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_data)

    def _on_data(self, data):
        points    = np.frombuffer(data.raw_data, dtype=np.float32)
        points    = points.reshape((-1, 4))[:, :3]
        self.data = self._normalize(points)

    def _normalize(self, points):
        # Normalize to [-1, 1] range (CARLA LiDAR range = 50m)
        # This brings LiDAR data to the same scale as state vector features
        points = points / 50.0
        
        # Instead of taking the first max_points (which only gives a small slice of the 360 view),
        # we sample evenly across the entire array to get a full 360-degree sparse representation.
        if len(points) >= self.max_points:
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=int)
            return points[indices].astype(np.float32)
            
        pad = np.zeros((self.max_points - len(points), 3), dtype=np.float32)
        return np.vstack([points, pad]).astype(np.float32)

    def get_data(self):
        if self.data is not None:
            return self.data
        return np.zeros((self.max_points, 3), dtype=np.float32)

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
            self.data   = None


# ─────────────────────────────────────────────
# Collision Sensor
# ─────────────────────────────────────────────
class CollisionSensor:
    def __init__(self, world, vehicle):
        self.world        = world
        self.vehicle      = vehicle
        self.has_collided = False
        self.sensor       = None

        blueprint   = world.get_blueprint_library().find('sensor.other.collision')
        transform   = carla.Transform()
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_data)

    def _on_data(self, event):
        self.has_collided = True

    def reset(self):
        self.has_collided = False

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


# ─────────────────────────────────────────────
# Lane Invasion Sensor
# ─────────────────────────────────────────────
class LaneInvasionSensor:
    """
    Detects when the vehicle crosses lane markings.
    Call reset_count() at the start of each step, then read
    invasion_count after world.tick() to get per-step crossings.
    """
    def __init__(self, world, vehicle):
        self.world          = world
        self.vehicle        = vehicle
        self.invasion_count = 0
        self.sensor         = None

        blueprint   = world.get_blueprint_library().find('sensor.other.lane_invasion')
        transform   = carla.Transform()
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        self.sensor.listen(self._on_data)

    def _on_data(self, event):
        self.invasion_count += 1

    def reset_count(self):
        self.invasion_count = 0

    def stop(self):
        if self.sensor is not None and self.sensor.is_alive:
            try:
                self.sensor.stop()
            except Exception:
                pass

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


