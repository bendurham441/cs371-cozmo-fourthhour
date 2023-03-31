#!/usr/bin/python
#code needed to stitch images together 
import cv2
def run():
  print("\n\n\n!!!!! STITCH !!! \n\n\n\n")
  images = []
  for i in range(36):
    images.append(cv2.imread('takingpics' + str((i * 10))))       
  stitcher = cv2.Stitcher.create()
  ret,pano = stitcher.stitch(images)
  cv2.imwrite('Panorama.jpeg',pano)