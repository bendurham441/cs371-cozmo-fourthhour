#!/usr/bin/python
from string import hexdigits
from turtle import width
import cv2
import pandas as pd
import random
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
from PIL import Image
import math 
import statistics as stat

# Arbitrary values, to model gaussian noise.
sensorVariance = 0.01
proportionalMotionVariance = 0.01

def monte_carlo_localize(robot: cozmo.robot.Robot):
  panoPixelArray = cv2.imread("Panorama.jpeg")
  panoPixelArray.astype("float")
  dimensions = panoPixelArray.shape
  width = dimensions[1]
  hieght = dimensions[0]
  # Initialize cozmo camera
  robot.camera.image_stream_enabled = True
  pixelWeights = [] # predictions

  # Algorithm MCL Line 2
  # fill array with uniform values as starting predictions at initialize
  particles = []  # starts as randomized particles, aka guesses for where the robot is facing
  M = 100   # Number of particles
  i = 0     # Iterations
  while i < M:
      particles.append(random.randint(0, width))
      i = i + 1
  # Saves preliminary predictions to a dataframe
  pointFrame = pd.DataFrame(particles, columns=['particles'])
  
  i = 0
  while i < 10: # time steps is arbitrary
    # NEED TO calculate random movement
    # Rotate 10 degrees to right
    robot.turn_in_place(degrees(-10.0)).wait_for_completed()
    cv_cozmo_image2 = None
    latest_image = robot.world.latest_image
    while latest_image is None:
      latest_image = robot.world.latest_image

    annotated = latest_image.annotate_image()
    if latest_image is not None:
      converted = annotated.convert()
      converted.save("latestImage", "JPEG", resolution=10)
    cozmo_image2 = latest_image.annotate_image(scale=None, fit_size=None, resample_mode=0)
    # Storing pixel RGB values in a 3D array
    cv_cozmo_image2 = np.array(cozmo_image2)

    # empty arrays that hold population number, and weight
    pixelPopulationNumber = []
    pixelWeights = []

    # empty that can hold new population, temp array list for the one above
    newParticles = []

    # Algorithm MCL Line 3
    for pose in particles:
      # Algorithm MCL Line 4
      newPose = sample_motion_model(pose, width)
      # Algorithm MCL line 5:
      # map is [0, 1] interval space for movement, sensing distance from 0
      weight = measurement_model(cv_cozmo_image2, newPose) 
      
      # Algorithm MCL line 6:
      pixelWeights = np.append(pixelWeights,[weight])
      pixelPopulationNumber = np.append(pixelPopulationNumber,[newPose])

    # Compute probabilities (proportional to weights) and cumulative distribution function for sampling of next pose population
    # NOTE: This is the heart of weighted resampling that is _not_ given in the text pseudocode.
    # - first sum weights

    # sum all weight, create new array size M, calculate probability
    sum_weights = 0.0
    for pixel in pixelWeights:
      sum_weights += pixel
    probabilities = []

    for m in pixelWeights:
      probabilities.append(m / sum_weights)

    #Cumulative Distribution Function
    cdf = []
    sum_prob = 0

    for prob in probabilities:
        sum_prob += prob
        cdf.append(sum_prob)
    # cdf[len(probabilities)] = 1.0 last is always 1.0

   # redistribute population to newX

     #Algorithm MCL line 8:
   #resampling
    for m in range(M):
        p = random.uniform(0, 1)
        index = 0
        while p >= cdf[index]:
            index += 1
        newParticles.append(pixelPopulationNumber[index])
    particles = newParticles
    i += 1
  newParticles.sort()

  # updating the CSV file with the original predictions and the newest predictions
  df = pd.DataFrame(newParticles, columns = ['newParticles'])
  df = df.join(pointFrame) # joins new predictions with original predictions
  df = df.sort_values(by=['newParticles'], ascending=False)
  df.to_csv("data/data.csv", index = False)


def sample_motion_model(xPixel, width):
  # making variance proportional to magnitude of motion command
  newX = xPixel + sample_normal_distribution(abs(proportionalMotionVariance))
  return max(0, min(width, newX))

# map in this case is [0, 1] allowable poses
def measurement_model(latestImage, particlePose):
    # Gaussian (i.e. normal) error, see https://en.wikipedia.org/wiki/Normal_distribution
    # same as p_hit in Figure 6.2(a), but without bounds. Table 5.2
  img = Image.open("Panorama.jpeg")
  width, height = img.size
  #get the slice of the panorama that corresponds to the pixel
  particle = slice(img, particlePose, 320, 320, height)
  particle = np.array(particle)
  #resize the images
  cv_particle = cv2.resize(particle, (width, height))
  image2 = cv2.resize(latestImage, (width, height))
  #compare how similar/different they are using MSE
  diff = compare_images(cv_particle, image2)
  #see Text Table 5.2, implementation of probability normal distribution
  return (1.0 / math.sqrt(2 * math.pi * sensorVariance)) * math.exp(- (diff * diff) / (2 * sensorVariance))

def sample_sensor_model(pose):
  # ideal sensor will return pose (distance to right of 0), but we'll model Gaussian noise (could have more components as in class)
  return pose + sample_normal_distribution(sensorVariance)

#see Text Table 5.4, implementation of sample normal distribution
def sample_normal_distribution(variance):
  sum = 0
  for i in range(12):
    sum += (2.0 * random.random()) - 1.0
  return math.sqrt(variance) * sum / 2.0

# Compares images by pixels using Mean Squared Error formula
def compare_images(imageA, imageB):
  # See https://en.wikipedia.org/wiki/Mean_squared_error 
  dimensions = imageA.astype("float").shape
  width = dimensions[1]
  height = dimensions[0]
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  # Dividing the values so they fit 
  err /= (width * height * width * height)
  return err

# Creates a slice of the panorama to compare to latestImage
def slice(imgName, center, pixelLeft, pixelRight, slice_size):
  # slice an image into parts slice_size wide
  # initialize boundaries
  img = Image.open("Panorama.jpeg")
  width, height = img.size
  left = center - pixelLeft
  right = center + pixelRight

  # if we go out of bounds set the limit to be the bounds of the image
  if center < pixelLeft:
      left = 0
  if center > (width - pixelRight):
      right = width

  # newImgSize = dim(center - 20, center + 20)
  upper = 0
  slices = int(math.ceil(width / slice_size))
  count = 1

  for slice in range(slices):
      # if we are at the end, set the lower bound to be the bottom of the image
      if count == slices:
          lower = width
      else:
          lower = int(count * slice_size)

          # box with boundaries
      bbox = (left, upper, right, lower)
      sliced_image = img.crop(bbox)
      cv_sliced = np.array(sliced_image)
      upper += slice_size
      # save the slice

      count += 1
      cv2.imwrite("Sliced.jpeg", cv_sliced)
      return sliced_image

# TO-DO

  #locate in panorama if it has gone a full 360
  #if pixels then knowing what is the point where things cycle around is important
  #subtract that number of pixels to wrap it around

  #MCL motion model
  #update poses when cozmo turns during monte carlo localization
  #need to update pixels by how many pixels 10 would move you

  #Histogram
  #not seeing density around spikes

  #slice
  #needs to wrap around if at one end or the other

  #figure out units for panorama so that motion model
  #(which should be wrapping around)
  #turning in degrees but units should be pixels
  #can't wrap around until we know...
  #what we are calling pixel 0

  #something that would print out where it thinks it is at 
  #and where it is facing as a demonstration of localization