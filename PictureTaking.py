#!/usr/bin/env python3
# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#implementation of montecarlo locolization
from PIL import Image
import cozmo
import cv2
from cozmo.util import degrees, distance_mm, speed_mmps
import Stitching
import random
import Localize
import MCLocalize
import MotionUpdate
import Histogram
import sys
import os


# Use python versions between 3.5 and 3.9 inclusive. This group used 3.9.7
print("Python version")
print (sys.version)

# spins the cozmo 360 degrees to get a panorama image of its current environment
def cozmo_program(robot: cozmo.robot.Robot):
  robot.say_text("Okay Here we goooooooo").wait_for_completed()
  move_arms = robot.set_lift_height(0)
  move_arms.wait_for_completed()
  set_head = robot.set_head_angle((cozmo.robot.MAX_HEAD_ANGLE) / 3, in_parallel = True)
  set_head.wait_for_completed()
  # Enabling Cozmo Camera
  robot.camera.image_stream_enabled = True
  
  # Saves picture of what Cozmo sees every 10 degrees.
  degree = 0
  while(degree < 360) :
    fileName = "takingpics" + str(degree)
 
    robot.turn_in_place(degrees(10)).wait_for_completed()
    
    latest_image = robot.world.latest_image
    annotated = latest_image.annotate_image()
    if latest_image is not None:
      print("image = %s" % latest_image)
      converted = annotated.convert()
      converted.save(fileName, "JPEG", resolution=10)
    degree += 10
    
# Turns the robot a random amount simulating a kidnapping robot problem
def randomTurn(robot: cozmo.robot.Robot):
  # Enabling Cozmo Camera
  robot.camera.image_stream_enabled = True
  # Rotate a random degree
  deg = random.randint(0, 60)  
  robot.turn_in_place(degrees(deg + 20)).wait_for_completed()
    
  # Take a picture and save as "latestImage"
  latest_image = robot.world.latest_image
  annotated = latest_image.annotate_image()
  if latest_image is not None:
    converted = annotated.convert()
    converted.save("latestImage", "JPEG", resolution=10)
  robot.say_text("Oh Noooooooo they kidnapped me").wait_for_completed()
  return deg

# Signals the program's completion
def madeItHome(robot: cozmo.robot.Robot):
  robot.say_text("Im hoooooooome").wait_for_completed()

# Rotates the robot in 5 degree intervals as it gathers data to try and localize
def rotato(robot: cozmo.robot.Robot):
  # Enabling Cozmo Camera
  robot.camera.image_stream_enabled = True
  # Rotate 5 degrees to the right
  robot.turn_in_place(degrees(5 * - 1)).wait_for_completed()
    
  # Take a picture and save as "latestImage"
  latest_image = robot.world.latest_image
  annotated = latest_image.annotate_image()
  if latest_image is not None:
    converted = annotated.convert()
    converted.save("latestImage", "JPEG", resolution=10)

# Initial set up for the panorama
cozmo.run_program(cozmo_program)
# Creates the panorama as Panorama.jpeg
Stitching.run()
# 'Kidnaps' the cozmo by turning a random direction
degree = cozmo.run_program(randomTurn)
# Runs Monte Carlo algorithm 
cozmo.run_program(MCLocalize.monte_carlo_localize)
# Completes localize, alerts with speech
cozmo.run_program(madeItHome)
# Generates histogram as hist.png
Histogram.makeHistogram()
