import cv2
import pandas as pd
import random
import numpy as np
import sys
import MotionUpdate
#computes the mse value used for comparing images
def mse(imageA, imageB):
  # the 'Mean Squared Error' between the two images is the
  # sum of the squared difference between the two images;
  # NOTE: the two images must have the same dimension
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1])
  
  # return the MSE, the lower the error, the more "similar"
  # the two images are
  return err
#compares the two images
def compare_images(imageA, imageB):
  # compute the mean squared error and structural similarity
  # index for the images
  m = mse(imageA, imageB)
  #s = ssim(imageA, imageB)
 
  # setup the figure
  
  #print("MSE: %.2f" % (m))
  return m
  
#the initial localization that generates a normal distribution of random hypothesis 
#and then resamples over the hyptohtesis based on the comparison to the first image from the cozmo
def localize():
  pano = cv2.imread("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/Panorama.jpeg")
  dimensions = pano.shape
  width = dimensions[1]
  height = dimensions[0]
  height = min(240, height)
  
  #generate the random points
  randomPoints = []
  numPics = 1000
  for i in range(numPics):
# When generating pictures we subtract 320 so that 
# the program does not go out of bounds when 
# selecting from panorama
    current = random.randint(1, width - 320)
    randomPoints.append(current)
  pointFrame = pd.DataFrame(randomPoints, columns=['X'])
  #get the pictures at those points
  randomPictures = []
  for randomPoint in randomPoints:
    randomPictures.append(pano[0:height, randomPoint:randomPoint+320])
    
  #grab the latest image from the cozmo and calculate the mse values
  homePic = cv2.imread("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/latestImage")
  homePic = homePic[0:height, 0:320]
  MSEList = []
  for i in range(numPics):
    MSEList.append(compare_images(homePic, randomPictures[i]))
  df = pd.DataFrame(MSEList, columns = ['MSE'])
  df = df.join(pointFrame)
  #Get the sum of the values
  MSESum = sum(df['MSE'])
  #invert values
  df['normalizedMSE'] = (df['MSE']/MSESum)
  df['invertedMSE'] = (1/df['normalizedMSE'])
  #Normalize the values
  newSum = sum(df['invertedMSE'])
  df['newProbs'] = (df['invertedMSE']/newSum)
  #if we pick the exact same picture, make its probability 1
  df = df.fillna(1)
  #renormalize in case we filled a NaN with 1
  probSum = sum(df['newProbs'])
  df['newProbs'] = (df['newProbs']/probSum)
  #remove unnnecessary information
  df = df.drop(columns=['normalizedMSE','invertedMSE'])
  #select new values according to the probabilities
  df['X'] = np.random.choice(df['X'], numPics, p=df['newProbs'])
  #randomize new x values
  df['X'] = df['X'].map(lambda x: abs(random.randint(x - 10, x + 10)))
  df.loc[df.X < 0, 'X'] = random.randint(1, width - 330)  
  df.loc[df.X > width - 350, 'X'] = random.randint(1, width - 350)  
  df = df.sort_values(by=['newProbs'], ascending=False)
  df.to_csv("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/data.csv", index = False)
#used for iterations after the initial stage to update our hypothesis based on the previous motions of the cozmo
#and then resample again
def localize2():
  data = pd.read_csv("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/data.csv")
  data['X'] = data['X'].map(lambda x: abs(x))
  pano = cv2.imread("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/Panorama.jpeg")
  dimensions = pano.shape
  width = dimensions[1]
  height = dimensions[0]
  height = min(240, height)
  
  #load in current pics
  currentPics = []
  cur = 0
  xValues = data['X'].tolist()
  for point in xValues:
    currentPics.append(pano[0:height, point:point+320])
    #print(width - (point + 320))
    #print(cur)
    #cur = cur + 1
    #print(pano[0:height, point:point+320].shape[1])
  #grab the new mse values for the new points
  homePic = cv2.imread("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/latestImage")
  homePic = homePic[0:height, 0:320]
  MSEList = []
  for i in range(len(currentPics)):
    MSEList.append(compare_images(homePic,currentPics[i]))
  data.drop(columns=['MSE'])
  data['MSE'] = MSEList
  #Get the sum of the values
  MSESum = sum(data['MSE'])
  #invert values
  data['normalizedMSE'] = (data['MSE']/MSESum)
  data['invertedMSE'] = (1/data['normalizedMSE'])
  #Normalize the values
  newSum = sum(data['invertedMSE'])
  data['newProbs'] = (data['invertedMSE']/newSum)
  #if we pick the exact same picture, make its probability 1
  data = data.fillna(1)
  #renormalize in case we filled a NaN with 1
  probSum = sum(data['newProbs'])
  data['newProbs'] = (data['newProbs']/probSum)
  #remove unnnecessary information
  data = data.drop(columns=['normalizedMSE','invertedMSE'])
  #select new values according to the probabilities
  data['X'] = np.random.choice(data['X'], len(currentPics), p=data['newProbs'])
  #randomize new x values
        #TODO: Implement into motion model, not done here
  data['X'] = data['X'].map(lambda x: abs(random.randint(x - 10, x + 10)))
  data.loc[data.X < 0, 'X'] = random.randint(1, width - 330)  
  data.loc[data.X > width - 350, 'X'] = random.randint(1, width - 350)  
  data = data.sort_values(by=['newProbs'], ascending=False)
  
  data.to_csv("/Accounts/turing/students/s19/ainsma01/AnkiCozmo/FinalProject/data.csv", index = False)

