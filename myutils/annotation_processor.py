
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from PIL import Image
from six import BytesIO
from tqdm.notebook import tqdm

from myutils.bounding_box_funcs import convert_coordinates_for_plot
from pathlib import Path


class AnnotationProcessor:
    def __init__(self, annotation_file, image_size:int=640):
        self.annotation_file = annotation_file
        self.df = pd.read_csv(str(self.annotation_file))  # Assumes CSV format
        self.images = []
        self.class_ids = []
        self.bboxes = []
        self.target_size = image_size

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

        # img_data = tf.io.gfile.GFile(path, 'rb').read()
        # image = Image.open(BytesIO(img_data))
        # (im_width, im_height) = image.size

        # return np.array(image.getdata()).reshape(
        #     (im_height, im_width, 3)).astype(np.uint8)
        # Read image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize with padding to preserve aspect ratio
        return tf.image.resize_with_pad(image, self.target_size, self.target_size)

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
                labels = []
                cords = []
                for _, row in image_annotations.iterrows():
                    xmin = int(row['xmin'])
                    ymin = int(row['ymin'])
                    xmax = int(row['xmax'])
                    ymax = int(row['ymax'])
                    original_width = int(row['width'])
                    original_height = int(row['height'])

                    # Calculate scale factor # scale = min(640/600, 640/800) = min(1.067, 0.8) = 0.8
                    scale = min(self.target_size / original_height, self.target_size / original_width)

                    # Calculate new dimensions 
                    # new_height = 600 * 0.8 = 480
                    # new_width = 800 * 0.8 = 640
                    new_height = int(original_height * scale)
                    new_width = int(original_width * scale)
                    
                    # Calculate padding
                    # dy = (640 - 480) // 2 = 80
                    # dx = (640 - 640) // 2 = 0
                    dy = (self.target_size - new_height) // 2
                    dx = (self.target_size - new_width) // 2
                    
                    # Adjust bounding box
                    # resized_xmin = 200 * 0.8 + 0 = 160
                    # resized_ymin = 150 * 0.8 + 80 = 200
                    # resized_xmax = 400 * 0.8 + 0 = 320
                    # resized_ymax = 300 * 0.8 + 80 = 320

                    resized_xmin = int(xmin * scale) + dx
                    resized_ymin = int(ymin * scale) + dy
                    resized_xmax = int(xmax * scale) + dx
                    resized_ymax = int(ymax * scale) + dy

                    # Normalize bounding box coordinates
                    converted_cords = convert_coordinates_for_plot(img_height=self.target_size, img_width=self.target_size, 
                                                                   bbox = [resized_xmin, resized_ymin, resized_xmax, resized_ymax])
                    labels.append(label_map[row['class']] )
                    cords.append(converted_cords)

                self.class_ids.append(labels)
                self.bboxes.append(np.array(cords))
                self.images.append(img)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")



        return self.images, self.class_ids, self.bboxes

class _AnnotationProcessor:
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
                labels = []
                cords = []
                for _, row in image_annotations.iterrows():
                    x_min = int(row['xmin'])
                    y_min = int(row['ymin'])
                    x_max = int(row['xmax'])
                    y_max = int(row['ymax'])
                    img_width = int(row['width'])
                    img_height = int(row['height'])

                    # Normalize bounding box coordinates
                    converted_cords = convert_coordinates_for_plot(img_height=img_height, img_width=img_width, bbox = [x_min, y_min, x_max, y_max])
                    labels.append(label_map[row['class']] )
                    cords.append(converted_cords)

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

            self.class_ids.append(labels)
            self.bboxes.append(np.array(cords))
            self.images.append(img)


        return self.images, self.class_ids, self.bboxes