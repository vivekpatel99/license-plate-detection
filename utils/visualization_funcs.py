import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.bounding_box_funcs import convert_coordinates_for_plot


def plot_random_images_bbox(*, image_paths:np.ndarray, class_ids:np.ndarray, bboxes:np.ndarray, class_map:dict, NUM_IMAGES_DISPLAY:int=9) -> None:
  fig = plt.figure(figsize=(8, 8))
  random_samples = random.sample(range(len(image_paths)), NUM_IMAGES_DISPLAY)
  print(f"Random samples: {random_samples}")

  for i, idx in enumerate(random_samples):
    ax = fig.add_subplot(3, 3, i+1)
    image = image_paths[idx]
    img_height, img_width, _ = image.shape
    if not isinstance(image, np.ndarray):
      image = cv2.imread(image)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    [x_min, y_min, x_max, y_max] = bboxes[idx][0]
   
    coords  = convert_coordinates_for_plot(img_height=img_height, img_width=img_width, bbox= [x_min, y_min, x_max, y_max], plot=True)
    ymin, xmin, ymax, xmax = coords[0]
    # print(class_ids[idx])
    ax.set_title(class_ids[idx])
    ax.axis('off')
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    plt.imshow(image)

  plt.show()