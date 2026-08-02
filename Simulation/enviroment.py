#!/usr/bin/env python
# coding: utf-8

# In[6]:


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

sys.path.append(r'E:\Carla_PPO_project_the.py')

import carla 
import numpy as np 
import random 
import time 
from Simulation.sensors import RGBCamera, LiDAR, CollisionSensor, LaneInvasionSensor


class CarlaEnvironment():
    def __init__(self, client, world, img_width = 64, img_hight = 64, max_lidar_points= 1024):
        self.client = client
        self.world = world
        self.map = world.get_map()
        self.img_width = img_width
        self.img_hight = img_hight 
        self.max_lidar_points = max_lidar_points 
        self.blueprint_library = world.get_blueprint_library()

        # Actors 
        self.vehicle       = None
        self.camera        = None
        self.lidar         = None 
        self.collision     = None
        self.lane_invasion = None

        #Route 
        self.waypoints = []
        self.current_wp_idx = 0 
        
        # Action Smoothing Tracking (2D: steer + accel)
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.frozen_steps   = 0
        self.prev_action    = np.zeros(2, dtype=np.float32)
    
    
    #reseting agent when collied or when reaching for the end point
    
    def reset(self, num_waypoints=20):
        self._destroy_actors()
        
        self.current_wp_idx = 0  # reset route index
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.frozen_steps   = 0
        self.prev_action    = np.zeros(2, dtype=np.float32)
        
        # spawn at a random position
        spawn_points = self.map.get_spawn_points()
        spawn_points = random.choice(spawn_points)
        vehicl_bp    = self.blueprint_library.filter('vehicle.nissan.micra')[0]
        self.vehicle = self.world.spawn_actor(vehicl_bp, spawn_points)
        
        # attach sensors
        self.camera        = RGBCamera(self.world, self.vehicle, self.img_width, self.img_hight)
        self.lidar         = LiDAR(self.world, self.vehicle, self.max_lidar_points)
        self.collision     = CollisionSensor(self.world, self.vehicle)
        self.lane_invasion = LaneInvasionSensor(self.world, self.vehicle)
        
        # generate random route
        self.waypoints = self._generate_route(spawn_points, num_waypoints=num_waypoints)
        
        # In synchronous mode tick to let the car settle on the road
        # and allow sensors to capture the first frame
        for _ in range(10):
            self.world.tick()
        
        # Reset collision flag AFTER warmup ticks — the vehicle drop can wrongly
        # trigger the collision sensor during those initial ticks.
        self.collision.reset()
        self.lane_invasion.reset_count()
        
        return self._get_obs()
    
    
    def step(self, action):
        # ── 2D action space: [steer, accel] ─────────────────
        # accel > 0 → throttle,  accel < 0 → brake
        raw_steer = float(np.clip(action[0], -1.0, 1.0))
        raw_accel = float(np.clip(action[1], -1.0, 1.0))

        # ── Single-layer EMA smoothing (α = 0.5) ────────────
        self.previous_steer = 0.5 * self.previous_steer + 0.5 * raw_steer
        self.previous_accel = 0.5 * self.previous_accel + 0.5 * raw_accel

        # Fast decay when agent clearly wants no braking
        # (prevents EMA memory from trapping the agent in brake)
        if raw_accel > 0.0:
            self.previous_accel = max(self.previous_accel, raw_accel * 0.7)

        steer = self.previous_steer

        # ── Map accel → throttle / brake ─────────────────────
        if self.previous_accel > 0:
            throttle = self.previous_accel
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = abs(self.previous_accel)

        # ── Mechanical Speed Cap + throttle floor ────────────
        speed = self._get_speed()

        # Throttle floor when nearly stopped — must be strong enough
        # to overcome the Micra's rolling resistance
        if brake < 0.05 and throttle < 0.25 and speed < 2.0:
            throttle = max(throttle, 0.25)

        # ── Update frozen counter BEFORE reward so progressive penalty works ──
        if speed < 1.0:
            self.frozen_steps += 1
        else:
            self.frozen_steps = 0

        # If stuck for several ticks force forward motion and tame steering
        unstuck_boost = False
        if speed < 1.0 and self.frozen_steps > 8:
            throttle      = max(throttle, 0.60)
            brake         = 0.0
            steer         = float(np.clip(steer * 0.15, -1.0, 1.0))
            unstuck_boost = True

        if speed >= 20.0:
            throttle = 0.0       # force coasting if over speed limit

        # ── Reset lane-invasion counter before this tick ─────
        self.lane_invasion.reset_count()

        control          = carla.VehicleControl()
        control.steer    = steer
        control.throttle = throttle
        control.brake    = brake

        self.vehicle.apply_control(control)
        self.world.tick()

        # Capture invasions that occurred during this tick
        lane_invasions = self.lane_invasion.invasion_count

        obs    = self._get_obs()
        reward = self._compute_reward(brake, lane_invasions)
        done   = self._is_done()

        info = {
            "speed"           : self._get_speed(),
            "waypoint_idx"    : self.current_wp_idx,
            "total_waypoints" : len(self.waypoints),
            "throttle"        : throttle,
            "brake"           : brake,
            "lane_invasions"  : lane_invasions,
            "unstuck_boost"   : unstuck_boost,
            "frozen_steps"    : self.frozen_steps,
        }

        return obs, reward, done, info
    
   
    
    
    def _navigation_features(self):
        """Distance to next route waypoint (normalized) and sin/cos of heading vs target (XY)."""
        dist_n = 1.0
        cos_h = 0.0
        sin_h = 0.0
        if self.vehicle is None or not self.waypoints:
            return dist_n, cos_h, sin_h
            
        # Visual Waypoint Line removed (Option B: skipping VAE retraining)

        idx = min(self.current_wp_idx, len(self.waypoints) - 1)
        wp = self.waypoints[idx]
        veh_tf = self.vehicle.get_transform()
        v_loc = veh_tf.location
        w_loc = wp.transform.location
        dx = w_loc.x - v_loc.x
        dy = w_loc.y - v_loc.y
        dz = w_loc.z - v_loc.z
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        dist_n = float(np.clip(dist / 40.0, 0.0, 1.0))

        fwd = veh_tf.get_forward_vector()
        fx, fy = float(fwd.x), float(fwd.y)
        norm = float(np.sqrt(dx * dx + dy * dy) + 1e-6)
        tx, ty = dx / norm, dy / norm
        cos_h = fx * tx + fy * ty
        sin_h = fx * ty - fy * tx
        return dist_n, cos_h, sin_h

    def _get_obs(self):
        dist_n, cos_h, sin_h = self._navigation_features()

        # Lateral distance from lane centre (normalised by half-lane width ≈ 1.75 m)
        # 0.0 = perfectly centred, 1.0 = at lane edge
        lane_offset = 0.0
        try:
            car_loc    = self.vehicle.get_transform().location
            nearest_wp = self.map.get_waypoint(car_loc, project_to_road=True)
            lane_offset = float(np.clip(
                car_loc.distance(nearest_wp.transform.location) / 1.75, 0.0, 1.0
            ))
        except Exception:
            pass

        return {
            "rgb"  : self.camera.get_data(),
            "lidar": self.lidar.get_data(),
            "state": np.array([
                self._get_speed() / 50.0,        # [0] speed (normalised)
                self.vehicle.get_control().steer, # [1] current steer
                dist_n,                           # [2] normalised dist to next wp
                cos_h,                            # [3] heading alignment
                sin_h,                            # [4] heading cross product
                lane_offset,                      # [5] lateral lane offset (NEW)
            ], dtype=np.float32)
        }
        
    def _compute_reward(self, brake=0.0, lane_invasions=0):
        reward = 0.0
        speed  = self._get_speed()   # km/h

        # ══════════════════════════════════════════════════════
        # SPEED  — the primary signal that drives the agent to MOVE
        # ══════════════════════════════════════════════════════
        if speed < 1.0:
            # Progressive freeze penalty: starts at -0.5, grows by -0.02 per frozen step
            # At 160 frozen steps this is -3.7 — far bigger than any passive reward.
            reward -= 0.5 + 0.02 * self.frozen_steps
        else:
            # Strong linear reward — at 20 km/h: +2.0/step; at 5 km/h: +0.5/step
            effective_speed = min(speed, 20.0)
            reward += 0.10 * effective_speed

        # ── Heading alignment (only counts when actually moving) ────────
        _, cos_h, _ = self._navigation_features()
        if speed > 1.0:
            # up to +0.15 when perfectly aligned at max speed
            reward += 0.15 * max(0.0, cos_h) * min(speed / 20.0, 1.0)

        # ══════════════════════════════════════════════════════
        # LANE KEEPING
        # ══════════════════════════════════════════════════════

        # ── Continuous Cross-Track Error (soft, per step) ──────────────
        try:
            car_loc          = self.vehicle.get_transform().location
            nearest_wp       = self.map.get_waypoint(car_loc, project_to_road=True)
            lane_center_dist = car_loc.distance(nearest_wp.transform.location)

            # Standard lane ≈ 3.5 m wide. ≤0.5 m off-centre is acceptable.
            if lane_center_dist > 0.5:
                reward -= (lane_center_dist - 0.5) * 0.3
        except Exception:
            pass

        # ── Lane Invasion (hard event — every marking crossed) ──────────
        # Each crossing of a solid or broken marking costs 3.0.
        if lane_invasions > 0:
            reward -= 3.0 * lane_invasions

        # ══════════════════════════════════════════════════════
        # WAYPOINT PROGRESS
        # ══════════════════════════════════════════════════════
        if self.waypoints:
            next_wp = self.waypoints[self.current_wp_idx]
            dist    = self._distance_to(next_wp.transform.location)

            if dist < 2.0:
                # ── Heading guard: agent must be facing the waypoint ──
                # This prevents collecting reward by spinning through the
                # waypoint sphere from behind or sideways.
                _, cos_h, _ = self._navigation_features()
                if cos_h > 0.0:   # cos > 0 means heading is within ±90° of target
                    self.current_wp_idx = min(
                        self.current_wp_idx + 1,
                        len(self.waypoints) - 1
                    )
                    if self.current_wp_idx == len(self.waypoints) - 1:
                        reward += 50.0   # destination reached ✅
                    else:
                        reward += 10.0   # intermediate waypoint ✅

        # ── Anti-Spinning Penalty ────────────────────────────────────────
        # If the agent is turning sharply but barely moving it is likely spinning
        # in place to exploit waypoint rewards. Penalise this pattern.
        current_steer = abs(self.vehicle.get_control().steer)
        if speed < 3.0 and current_steer > 0.6:
            reward -= 1.5  # spinning in place tax

        # ══════════════════════════════════════════════════════
        # SAFETY
        # ══════════════════════════════════════════════════════

        # ── Harsh braking penalty ────────────────────────────────────────
        if brake > 0.3:
            reward -= 0.1

        # ── Collision ───────────────────────────────────────────────────
        if self.collision.has_collided:
            reward -= 50.0

        return reward
    
    
    
    def _is_done(self):
        # NOTE: frozen_steps is updated in step() before reward computation.
        # Here we only read it to check the termination condition.
        if self.frozen_steps > 160:  # 160 steps ≈ 8 seconds at 20 FPS
            return True

        if self.collision.has_collided:
            return True

        if self.waypoints and self.current_wp_idx == len(self.waypoints) - 1:
            last_wp = self.waypoints[-1]
            if self._distance_to(last_wp.transform.location) < 2.0:
                return True

        return False
    
    
    def _generate_route(self, start_transform, num_waypoints = 20):
        waypoints = []
        current_wp = self.map.get_waypoint(start_transform.location)
        
        for _ in range(num_waypoints):
            next_wps = current_wp.next(6.0)  # 6m spacing — harder to exploit by spinning
            if not next_wps: 
                break
            current_wp = random.choice(next_wps)
            waypoints.append(current_wp)
            
            
        return waypoints
    
    def _get_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    def _distance_to(self, location):
        veh_loc = self.vehicle.get_transform().location
        return veh_loc.distance(location)
    
    
    
    def _destroy_actors(self):
        # ── Sync-mode safe teardown ──────────────────────────────
        # In sync mode, sensor.stop() blocks until the pending callback
        # is delivered. We must tick once AFTER stopping to flush the
        # queue, otherwise CARLA deadlocks on the next reset.
        for sensor in [self.camera, self.lidar, self.collision, self.lane_invasion]:
            if sensor is not None:
                try:
                    sensor.stop()          # stop listening first
                except Exception:
                    pass

        # Flush any in-flight sensor callbacks
        try:
            self.world.tick()
        except Exception:
            pass

        for sensor in [self.camera, self.lidar, self.collision, self.lane_invasion]:
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass

        if self.vehicle is not None and self.vehicle.is_alive:
            try:
                self.vehicle.destroy()
            except Exception:
                pass

        self.camera        = None
        self.lidar         = None
        self.collision     = None
        self.lane_invasion = None
        self.vehicle       = None
                
                
    
    def close(self):
        self._destroy_actors()
        print("Environment closed.")      
    
        



