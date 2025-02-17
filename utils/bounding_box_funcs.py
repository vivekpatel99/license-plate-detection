
import numpy as np


def convert_coordinates_for_plot(image, bbox, plot=False):
  x_center, y_center, width, height = bbox

  xmin = x_center - (width / 2)   
  ymin = y_center - (height / 2)
  xmax = x_center + (width / 2) 
  ymax = y_center + (height / 2) 

  if plot:
    img_height, img_width = image.shape[:2]
    # xmin = int(max(0, xmin * img_width))  # Clip to 0
    # ymin = int(max(0, ymin * img_height)) # Clip to 0
    # xmax = int(min(img_width, xmax * img_width)) # Clip to image width
    # ymax = int(min(img_height, ymax * img_height))# Clip to image height
    xmin = int(xmin * img_width) # Clip to 0
    ymin = int(ymin * img_height)# Clip to 0
    xmax =  int (xmax * img_width) # Clip to image width
    ymax =  int(ymax * img_height)# Clip to image height


  return [xmin, ymin, xmax, ymax]#.reshape(1, 4)
