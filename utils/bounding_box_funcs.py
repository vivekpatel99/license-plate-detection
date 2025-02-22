
import numpy as np


def convert_coordinates_for_plot(*,img_height, img_width, bbox, plot=False)->  np.ndarray:
  """
  Convert bounding box coordinates to normalized values
  between 0 and 1.

  Args:
    img_height (int): Height of the image.
    img_width (int): Width of the image.
    bbox (list): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    plot (bool, optional): Whether to plot the bounding box or not. Defaults to False.

  Returns:
    np.ndarray: Normalized bounding box coordinates in the format [[y_min, x_min, y_max, x_max]].
  """
  # Normalize bounding box coordinates
  xmin = bbox[0] / img_width
  ymin = bbox[1] / img_height
  xmax = bbox[2] / img_width
  ymax = bbox[3] / img_height

  if plot:
    """
    Clip the bounding box coordinates to the image boundaries.
    """
    xmin = int(max(0, xmin * img_width))  # Clip to 0
    ymin = int(max(0, ymin * img_height)) # Clip to 0
    xmax = int(min(img_width, xmax * img_width)) # Clip to image width
    ymax = int(min(img_height, ymax * img_height))# Clip to image height

  return [ymin, xmin, ymax, xmax] #.reshape(1, 4)
