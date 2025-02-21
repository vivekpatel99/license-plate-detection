
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from PIL import Image
from six import BytesIO
from tqdm.notebook import tqdm

from utils.bounding_box_funcs import convert_coordinates_for_plot


class AnnotationProcessor:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.df = pd.read_csv(str(self.annotation_file))  # Assumes CSV format
        self.images = []
        self.class_ids = []
        self.bboxes = []

    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        path: a file path.

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """

        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size

        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    

    def process_annotations(self, image_dir:Path, label_map:dict):
        """
        Processes annotations and draws bounding boxes on images.

        Args:
            image_dir: The directory containing the images.

        Returns:
            A list of tuples, where each tuple contains:
                - The image with bounding boxes drawn.
                - A list of normalized bounding box coordinates for each object in the image.
        """
        uni_list = self.df['filename'].unique()
        # uni_list =list(self.df['filename'].unique())
        for image_name in uni_list:  # Iterate over unique images
            image_path = image_dir / image_name  # Construct full image path
            try:
                img = self.load_image_into_numpy_array(str(image_path))
                
                if img is None:
                    print(f"Warning: Image not found at {image_path}")
                    continue  # Skip to the next image

                image_annotations = self.df[self.df['filename'] == image_name]  # Get annotations for this image
      
                for _, row in image_annotations.iterrows():
                    x_min = int(row['xmin'])
                    y_min = int(row['ymin'])
                    x_max = int(row['xmax'])
                    y_max = int(row['ymax'])
                    label = label_map[row['class']] 

                    img_width = int(row['width'])
                    img_height = int(row['height'])

                    # Normalize bounding box coordinates
                    cords = convert_coordinates_for_plot(img_height=img_height, img_width=img_width, bbox = [x_min, y_min, x_max, y_max])
    
                    # normalized_boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm, label])  # Include label
                self.images.append(img)
                self.class_ids.append(label)
                self.bboxes.append(cords)

                    # Draw bounding box (optional, for visualization)
                    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
                    # cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

        return self.images, self.class_ids, self.bboxes