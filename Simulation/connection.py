#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

class ClientConnection(): 
    def __init__(self, host="localhost", port=2000):
        self.host = host
        self.port = port
        self.client = None
        self.world = None

    def setup(self, synchronous_mode="yes", town_name="Town10HD"):
        if synchronous_mode == "yes":
            self.synchronous_mode = True
        elif synchronous_mode == "no":
            self.synchronous_mode = False
        else:
            print("Invalid input")
            return None

        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10e6)

            self.world = self.client.load_world(town_name, map_layers=carla.MapLayer.All)

            settings = self.world.get_settings()
            settings.synchronous_mode = self.synchronous_mode
            
            
            if self.synchronous_mode == True:
                settings.fixed_delta_seconds = 0.05
                
            else:
                settings.fixed_delta_seconds = None
                
                
                
            self.world.apply_settings(settings)
            mode_str = "sync" if self.synchronous_mode else "async"
            print(f"Connected\nDefault map {town_name} loaded in {mode_str} mode")
            return self.client, self.world

        except Exception as e:
            print(f"Failed to connect: {e}")
            return None

if __name__ == "__main__":
    # ─────────────────────────────────────────────
    # User Inputs
    # ─────────────────────────────────────────────
    town        = input("Enter town name (default: Town10HD_Opt): ").strip() or "Town10HD_Opt"
    sync        = input("Synchronous mode? yes/no (default: yes): ").strip() or "yes"
    rm_buildings = input("Remove buildings? yes/no (default: no): ").strip() or "no"

    # ─────────────────────────────────────────────
    # Connect
    # ─────────────────────────────────────────────
    conn = ClientConnection()
    result = conn.setup(synchronous_mode=sync, town_name=town)

    if result is None:
        print("Connection failed. Exiting.")
    else:
        client, world = result

        # ─────────────────────────────────────────
        # Apply Optimizations
        # ─────────────────────────────────────────
        if rm_buildings == "yes":
            world.unload_map_layer(carla.MapLayer.Buildings)
            print("Buildings removed.")


# In[ ]:




