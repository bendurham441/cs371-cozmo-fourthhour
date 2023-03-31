#!/usr/bin/python
#This class updates our data for when we move the cozmo to make a new picture to compare to the panorama

import cv2
import pandas as pd
import random
import numpy as np
import sys
import math
def motionUpdate():
  #reads in the panorama and the data
  data = pd.read_csv("data/data.csv")
  pano = cv2.imread("Panorama.jpeg")
  dimensions = pano.shape
  width = dimensions[1]
  
  #gives a rough estamite of how many pixels to the right we are moving in the panorama image
  toAdd = math.floor(width / 360)
  data['X'] = data['X'].apply(lambda x: x + (5 * toAdd))
  #data['X'] = data['X'] + (10 * toAdd)
  data.loc[data.X > (width - 320), 'X'] = random.randint(1, width - 330)  
  data.to_csv("data/data.csv", index = False)