
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tqdm.notebook import tqdm

from utils.bounding_box_funcs import convert_coordinates_for_plot


class PrepareDataset:
    def __init__(self, image_dir: Path, label_dir: Path, dst_img_size:tuple[int, int]=(224,224)) -> None:
        """
        Args:
            image_dir (str): Path to the directory containing the images.
            label_dir (str): Path to the directory containing the labels.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dst_img_size= dst_img_size
        self.images = []
        self.class_ids = []
        self.bboxes = []
        
        self.rebal_images = []
        self.rebal_class_ids = []
        self.rebal_bboxes = []
        # https://www.activeloop.ai/resources/better-object-detection-image-augmentation-with-tensor-flow-and-albumentations/
        self.train_aug = iaa.Sequential([
            iaa.GammaContrast(1.5),
            iaa.Fliplr(0.3),

            # `Sometimes()` applies a function randomly to the inputs with
            # a given probability (0.3, in this case).
            iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
        ])
    def seperate_class_with_datasets(self, class_id:int):
        idx = np.where(self.class_ids == class_id)[0]
        return (np.array(self.images)[idx], self.class_ids[idx], self.bboxes[idx])
    
    def rebalance_by_down_sampling_datasets(self, augment=False, plot=False):

        unique_class_ids, value_counts  = np.unique(self.class_ids, return_counts=True)
        print(f"[INFO] Unique class ids: {unique_class_ids}, value counts: {value_counts}")

        down_sampling_size = value_counts.min()
        print(f"[INFO] Down sampling size: {down_sampling_size}")
        
        self.rebal_images = []
        self.rebal_class_ids = []
        self.rebal_bboxes = []

        for id in unique_class_ids:
            images, class_ids, bboxes = self.seperate_class_with_datasets(id)
            self.rebal_images.extend(images[:down_sampling_size])
            self.rebal_class_ids.extend(class_ids[:down_sampling_size])
            self.rebal_bboxes.extend(bboxes[:down_sampling_size])

        if augment:
            self.prepare_dataset_augmented(plot=plot)

    
        return self.rebal_images, self.rebal_class_ids, self.rebal_bboxes

    def get_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads and parses YOLOv8 labels.

        Args:
            image_dir: Path to the directory containing images.
            label_dir: Path to the directory containing labels.
            dst_img_size: Tuple (width, height) specifying the desired image size.

        Returns:

        images = []
        class_ids = []
        bboxes = []

        for file_name in tqdm(list(self.image_dir.iterdir())[:100]):  # Removed [:100]
            if file_name.suffix.lower() in (".jpg", ".png", ".jpeg"): # Added .jpeg and lower() for robustness
                image_path = file_name
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                if not label_file_path.exists():
                    print(f"Label file not found for image: {image_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"Label file is empty: {label_file_path}")
                    continue

                for line in lines:
                    try:
                        values = np.array([float(value) for value in line.split()]) # Explicit float conversion
                        class_id = int(values[0])  # Explicit int conversion for class ID
                        coords = values[1:5].astype(np.float32)  # Ensure float32 for coords
                        
                        images.append(str(image_path))
                        bboxes.append(coords)
                        class_ids.append(class_id)

                    except ValueError as e:  # Catch specific ValueError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    except IndexError as e: # Catch potential IndexError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue

        return images, (np.array(class_ids, dtype=np.int8), np.array(bboxes, dtype=np.float32))
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Image data (NumPy array, shape (num_images, height, width, channels)).
                - Class IDs (NumPy array, dtype=np.int32).
                - Bounding boxes (NumPy array, shape (num_images, max_objects, 4), dtype=np.float32).
        """

        for file_name in tqdm(list(self.image_dir.iterdir())[:1000]):  # Removed [:100]
            if file_name.suffix.lower() in (".jpg", ".png", ".jpeg"): # Added .jpeg and lower() for robustness
                image_path = file_name
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                image = cv2.imread(str(image_path))
               
                if not label_file_path.exists():
                    print(f"Label file not found for image: {image_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"Label file is empty: {label_file_path}")
                    continue

                for line in lines:
                    try:
                        values = np.array([float(value) for value in line.split()]) # Explicit float conversion
                        class_id = int(values[0])  # Explicit int conversion for class ID
                        coords = values[1:5].astype(np.float32)  # Ensure float32 for coords
                        
                        self.images.append(image)
                        self.bboxes.append(coords)
                        self.class_ids.append(class_id)

                    except ValueError as e:  # Catch specific ValueError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    except IndexError as e: # Catch potential IndexError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue

        # self.images = np.array(self.images)
        self.bboxes = np.array(self.bboxes)
        self.class_ids = np.array(self.class_ids, dtype=np.int8)

        return self.images, self.class_ids, self.bboxes
    
    def prepare_dataset_augmented(self, plot=False):
        self.augmented_images = []
        self.augmented_bbxes = []
        self.augmented_class_ids = []
        # i = 0
        for img, id, bbx in zip(self.rebal_images, self.rebal_class_ids, self.rebal_bboxes):
            # print(bbx)
            x1, y1, x2, y2 = convert_coordinates_for_plot(img, bbx, plot=plot)
            # print(x1, y1, x2, y2)
            (new_image, new_bbx) = self.train_aug(image=img, bounding_boxes=ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=id))
            self.augmented_images.append(new_image)
            # print(new_bbx)
            self.augmented_bbxes.append(np.array([new_bbx.x1, 
                                        new_bbx.y1, 
                                        new_bbx.x2, 
                                        new_bbx.y2]))
            
            self.augmented_class_ids.append(new_bbx.label)
            # if i > 5:
            #     break
            # i += 1
        self.rebal_images.extend(self.augmented_images)
        self.rebal_class_ids.extend(self.augmented_class_ids)
        self.rebal_bboxes.extend(self.augmented_bbxes)




class AnnotationProcessor:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.df = pd.read_csv(str(self.annotation_file))  # Assumes CSV format
        self.images = []
        self.class_ids = []
        self.bboxes = []

    def process_annotations(self, image_dir, class_id_map):
        """
        Processes annotations and draws bounding boxes on images.

        Args:
            image_dir: The directory containing the images.

        Returns:
            A list of tuples, where each tuple contains:
                - The image with bounding boxes drawn.
                - A list of normalized bounding box coordinates for each object in the image.
        """
        uni_list = list(self.df['filename'].unique())
        for image_name in uni_list[:100]:  # Iterate over unique images
            image_path = image_dir/ image_name  # Construct full image path
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Warning: Image not found at {image_path}")
                    continue  # Skip to the next image

                img_height, img_width, _ = img.shape

                image_annotations = self.df[self.df['filename'] == image_name]  # Get annotations for this image
      
                for _, row in image_annotations.iterrows():
                    x_min = int(row['xmin'])
                    y_min = int(row['ymin'])
                    x_max = int(row['xmax'])
                    y_max = int(row['ymax'])
                    label = row['class']  # or 'label' depending on your CSV

                    # Normalize bounding box coordinates
                    x_min_norm = x_min / img_width
                    y_min_norm = y_min / img_height
                    x_max_norm = x_max / img_width
                    y_max_norm = y_max / img_height
                    # normalized_boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm, label])  # Include label
                    self.images.append(img)
                    self.class_ids.append(label)
                    self.bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])

                    # Draw bounding box (optional, for visualization)
                    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box
                    # cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

        return np.array(self.images), np.array([class_id_map[class_id] for class_id in self.class_ids]), np.array(self.bboxes)