

import logging
import xml.etree.ElementTree as xet
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tqdm import tqdm

from myutils.bounding_box_funcs import normalize_coordinates
from myutils.logs import get_logger


class AnnotationProcessor:
    def __init__(self, annotation_file, image_size:int=640):
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.annotation_file = annotation_file
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
        # Read image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize with padding to preserve aspect ratio
        # return tf.image.resize_with_pad(image, self.target_size, self.target_size)
        return tf.cast(image, tf.float32) 


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
        self.df = pd.read_csv(str(self.annotation_file))  # Assumes CSV format
        uni_list = self.df['filename'].unique()
        # uni_list =list(self.df['filename'].unique())
        for image_name in tqdm(uni_list[:200]):  # Iterate over unique images
            image_path = image_dir / image_name  # Construct full image path
            try:
                img = self.load_image_into_numpy_array(str(image_path))
                
                if img is None:
                    self.log.warning(f"Image not found at {image_path}")
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

                    xmin, ymin, xmax, ymax = self.rescale_coord_to_dst_img_size(xmin, ymin, xmax, ymax, original_width, original_height)

                    # Normalize bounding box coordinates
                    # converted_cords = convert_coordinates_for_plot(img_height=self.target_size, img_width=self.target_size, 
                    #                                                bbox = [xmin, ymin, xmax, ymax])
                    converted_cords = normalize_coordinates(img_height=original_width, 
                                                                   img_width=original_height, 
                                                                   bbox = [xmin, ymin, xmax, ymax])
                    labels.append(label_map[row['class']] )
                    cords.append(converted_cords)
                # if is_small_img:
                #     continue
                self.class_ids.append(labels)
                self.bboxes.append(np.array(cords))
                self.images.append(img)
            except Exception as e:
                self.log.error(f"Processing image {image_name}: {e}")
                
        return self.images, self.class_ids, self.bboxes

    def rescale_coord_to_dst_img_size(self, xmin, ymin, xmax, ymax, original_width, original_height):
        if original_height != self.target_size or original_width != self.target_size:
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

            xmin = int(xmin * scale) + dx
            ymin = int(ymin * scale) + dy
            xmax = int(xmax * scale) + dx
            ymax = int(ymax * scale) + dy
        return xmin,ymin,xmax,ymax

    def process_annotations_xml(self, image_dir:Path, label_map:dict):
        path = self.annotation_file.glob('*.xml') 
        for filename in tqdm(path):
            image_path = image_dir / f'{filename.stem}.png' # Construct full image path
            try:
                img = self.load_image_into_numpy_array(str(image_path))
                
                if img is None:
                    self.log.warning(f"Image not found at {image_path}")
                    continue  # Skip to the next image

                info = xet.parse(filename)
                root = info.getroot()
                img_size_info =  root.find('size')
                original_width = int(img_size_info.find('width').text)
                original_height = int(img_size_info.find('height').text)

                labels = []
                cords = []
                for member_object in root.findall('object'):
                    labels_info = member_object.find('bndbox')
                    xmin = int(labels_info.find('xmin').text)
                    xmax = int(labels_info.find('xmax').text)
                    ymin = int(labels_info.find('ymin').text)
                    ymax = int(labels_info.find('ymax').text)

                    # xmin, ymin, xmax, ymax = self.rescale_coord_to_dst_img_size(xmin, ymin, xmax, ymax, original_width, original_height)

                    # Normalize bounding box coordinates
                    # converted_cords = normalize_coordinates(img_height=self.target_size, img_width=self.target_size, 
                    #                                             bbox = [xmin, ymin, xmax, ymax])
                    converted_cords = normalize_coordinates(img_height=original_height, 
                                                            img_width=original_width, 
                                                            bbox = [xmin, ymin, xmax, ymax])
                    cls = member_object.find('name').text
                    labels.append(label_map[cls] )
                    cords.append(converted_cords)
                    # cords.append([ymin, xmin, ymax, xmax])
 
                self.class_ids.append(labels)
                self.bboxes.append(np.array(cords))
                self.images.append(img)

            except Exception as e:
                self.log.error(f"Processing image {image_path}: {e}")
                
        return self.images, self.class_ids, self.bboxes

