#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import random
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import numpy as np
import time 
import cv2
import math

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(10)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb.set_attribute('image_size_x',f"{self.im_width}")
        self.rgb.set_attribute('image_size_y',f"{self.im_height}")
        self.rgb.set_attribute('fov',f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))

        time.sleep(4)

        col_sensor = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to = self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)


    def process_img(image):
        i = np.array(image.raw_data)
        # print(dir(image))
        i2 = i.reshape((self.im_height,self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow('',i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = -1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        




# actor_list = []

# try:
#     client = carla.Client('localhost',2000)
#     print('client : ',client.get_client_version())
#     print('server : ',client.get_server_version())
#     client.set_timeout(10.0)

#     world = client.get_world()
    
#     blueprint_library = world.get_blueprint_library()

#     bp = blueprint_library.filter('model3')[0]
#     print(bp)

#     spawn_point = random.choice(world.get_map().get_spawn_points())
#     vehicle = world.spawn_actor(bp, spawn_point)
#     # vehicle.set_autopilot(True)

#     vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0))
#     actor_list.append(vehicle)

#     cam_bp = blueprint_library.find("sensor.camera.rgb")
#     cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
#     cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
#     cam_bp.set_attribute('fov','110')

#     spawn_point = carla.Transform(carla.Location(x = 2.5, z = 0.7))

#     sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle)
#     actor_list.append(sensor)
#     sensor.listen(lambda data: process_img(data))




#     time.sleep(5)

# finally:
#     for actor in actor_list:
#         actor.destroy()
#     print('all cleaned up')

