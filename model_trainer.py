import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)

# # https://www.tensorflow.org/guide/mixed_precision
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# import the label map utility module
from object_detection.utils import label_map_util

# import module for reading and updating configuration files.
from object_detection.utils import config_util

# import module for visualization. use the alias `viz_utils`
from object_detection.utils import visualization_utils as viz_utils

# import module for building the detection model
from object_detection.builders import model_builder
from myutils.visualization_funcs import plot_random_images_bbox
from myutils.annotation_processor import AnnotationProcessor
from myutils.logs import get_logger
import logging
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from tqdm.notebook import tqdm
from pathlib import Path
from dotenv import load_dotenv
import random
from hydra import initialize, compose

# https://gist.github.com/bdsaglam/586704a98336a0cf0a65a6e7c247d248

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")

DATASET_DIRS = Path(cfg.DATASET.DATASET_DIR)
DATASET_DIRS.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = Path(cfg.DATASET_DIRS.TRAIN_DIR)
VALIDATION_DIR = Path(cfg.DATASET_DIRS.VALIDATION_DIR)
TEST_DIR = Path(cfg.DATASET_DIRS.TEST_DIR)

IMG_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE

TRAIN_ANNOT_FILE_PATH = TRAIN_DIR / cfg.DATASET.ANNOTATION_FILE_NAME
TEST_ANNOT_FILE_PATH = TEST_DIR / cfg.DATASET.ANNOTATION_FILE_NAME
VALID_ANNOT_FILE_PATH = VALIDATION_DIR / cfg.DATASET.ANNOTATION_FILE_NAME

CONFIG_PIPELINE_PATH = Path(cfg.OUTPUTS.CONFIG_PIPELINE_PATH)
PRETRAIN_MODEL_PATH  = Path(cfg.PRETRAIN_MODEL.PATH)
OUTPUTS_DIRS = Path(cfg.OUTPUTS.OUPUT_DIR)
EXPORTER_SCRIPT = Path(cfg.OUTPUTS.EXPORTER_SCRIPT)

CHECKPOINT_PATH = Path(cfg.OUTPUTS.CHECKPOINT_PATH)
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 640, 640, 3], dtype=tf.float32)])
def detect_fn(input_tensor):
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)

    # use the detection model's postprocess() method to get the the final detections
    detections = model.postprocess(prediction_dict, shapes)
    return detections

# decorate with @tf.function for faster training 
@tf.function
def train_step_fn(image_list,
                groundtruth_boxes_list,
                groundtruth_classes_list,
                model,
                optimizer,
                vars_to_fine_tune, 
                ):

    with tf.GradientTape() as tape:
        preprocessed_image_list = []
        true_shape_list = []

        # Preprocess the images
        for img in image_list:
          processed_img, true_shape = model.preprocess(img)
          preprocessed_image_list.append(processed_img)
          true_shape_list.append(true_shape)
   
        preprocessed_image_tensor =  tf.concat(preprocessed_image_list, axis=0)
        true_shape_tensor = tf.concat(true_shape_list, axis=0)

        # Make a prediction
        prediction_dict = model.predict(preprocessed_image_tensor, true_shape_tensor)

        # Provide the ground truth to the model
        model.provide_groundtruth(
                    groundtruth_boxes_list=groundtruth_boxes_list,
                    groundtruth_classes_list=groundtruth_classes_list)
        
        # Calculate the total loss (sum of both losses)
        losses_dict = model.loss(prediction_dict, true_shape_tensor)
            
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        # Calculate the gradients
        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        

        # Optimize the model's selected variables
        clipped_gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients]
        optimizer.apply_gradients(zip(clipped_gradients, vars_to_fine_tune))


    return total_loss

def build_model(num_classes:int, pretrain_model_path:Path):
    # define the path to the .config file for ssd resnet 50 v1 640x640
    pipeline_config = '/opt/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'

    # Load the configuration file into a dictionary
    configs = config_util.get_configs_from_pipeline_file(pipeline_config) 
    model_config = configs['model']
    # Modify the number of classes from its default of 90
    model_config.ssd.num_classes = num_classes

    # Freeze batch normalization
    model_config.ssd.freeze_batchnorm = True
    model = model_builder.build(model_config=model_config,is_training=True)
    tmp_box_predictor_checkpoint = tf.train.Checkpoint(
   _base_tower_layers_for_heads = model._box_predictor._base_tower_layers_for_heads,
   _box_prediction_head = model._box_predictor._box_prediction_head
    )
    tmp_model_checkpoint = tf.compat.v2.train.Checkpoint (
    _feature_extractor=model._feature_extractor,
    _box_predictor = tmp_box_predictor_checkpoint)
    checkpoint_path =  str(PRETRAIN_MODEL_PATH / 'checkpoint/ckpt-0')

    # Define a checkpoint that sets `model` to the temporary model checkpoint
    checkpoint = tf.train.Checkpoint(
        model=tmp_model_checkpoint
    )
    # Restore the checkpoint to the checkpoint path
    checkpoint.restore(save_path=checkpoint_path)
    return model

def main():
    load_dotenv()

    log = get_logger(__name__, log_level=logging.INFO)
    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.error("No GPU found")
        raise Exception("No GPU found")
    log.info(f'{found_gpu=}, {tf.__version__=}')

    label_map = {
        'License_Plate':1
    }

    prepare_train_dataset = AnnotationProcessor(annotation_file=TRAIN_ANNOT_FILE_PATH)

    train_images, train_class_ids, train_bboxes  = prepare_train_dataset.process_annotations(image_dir=TRAIN_DIR, label_map=label_map)
    prepare_valid_dataset = AnnotationProcessor(annotation_file=VALID_ANNOT_FILE_PATH)

    valid_images, valid_class_ids, valid_bboxes  = prepare_valid_dataset.process_annotations(image_dir=VALIDATION_DIR, label_map=label_map)
    train_images.extend(valid_images), train_class_ids.extend(valid_class_ids), train_bboxes.extend(valid_bboxes)

    # Assign the license plate class ID
    class_id = 1

    # define a dictionary describing license plate class
    category_index = {class_id :
    {'id'  : class_id,
    'name': 'License_Plate'}
    }

    # Specify the number of classes that the model will predict
    num_classes = 1
    label_id_offset = 1
    train_image_tensors = []

    # lists containing the one-hot encoded classes and ground truth boxes
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []

    for train_image, bbox in zip(train_images, train_bboxes):
        # convert training image to tensor, add batch dimension, and add to list
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image, dtype=tf.float32)/255., axis=0))
        
        # convert numpy array to tensor, then add to list
        gt_box_tensors.append(tf.convert_to_tensor(bbox, dtype=tf.float32))

        # apply offset to to have zero-indexed ground truth classes
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[bbox.shape[0]], dtype=np.int32) - label_id_offset
        )
        # do one-hot encoding to ground truth classes
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))


    model = build_model(num_classes, PRETRAIN_MODEL_PATH)
    # Run a dummy image to generate the model variables
    # use the detection model's `preprocess()` method and pass a dummy image
    dummy_img = tf.zeros([1,640,640,3])
    tmp_image, tmp_shapes = model.preprocess(dummy_img)

    # run a prediction with the preprocessed image and shapes
    tmp_prediction_dict = model.predict(tmp_image, tmp_shapes)

    # postprocess the predictions into final detections
    last_tune_layer=len(model.trainable_variables) //4 #30
    to_fine_tune = [model.trainable_variables[layer_num] for layer_num in range(last_tune_layer)]

    # # set the optimizer and pass in the learning_rate
    decay_steps = 50
    decay_rate = 0.96

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.TRAIN.LEARNING_RATE,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=cfg.TRAIN.LEARNING_RATE, 
        momentum=0.9,
    )

    with mlflow.start_run():
        mlflow.log_param("lr", cfg.TRAIN.LEARNING_RATE)
        mlflow.log_param("batch_size",cfg.TRAIN.BATCH_SIZE)
        mlflow.log_param("EPOCHS",cfg.TRAIN.NUM_EPOCHS)
        mlflow.log_param("optimizer", optimizer.get_config())
        mlflow.autolog()
        
    print('Start fine-tuning!', flush=True)
    train_loss_results = []
    tf.keras.backend.set_learning_phase(True)
    for _epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images)))
        random.shuffle(all_keys) 
        example_keys = all_keys[:cfg.TRAIN.BATCH_SIZE]

        # Get the ground truth
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]

        # get the images
        image_tensors = [train_image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors,
                                gt_boxes_list,
                                gt_classes_list,
                                model,
                                optimizer,
                                to_fine_tune)

        # if _epoch % 5 == 0:
        # Presentation
        _loss = total_loss.numpy()
        _lr = optimizer.learning_rate.numpy()

        train_loss_results.append(_loss)
        mlflow.log_metric("loss", _loss, step=_epoch)
        mlflow.log_metric("lr", _lr, step=_epoch)

        # Update learning rate.
        current_learning_rate = lr_schedule(_epoch)
        optimizer.learning_rate.assign(current_learning_rate)
        log.info(f'EPOCH {_epoch}/{cfg.TRAIN.NUM_EPOCHS} - {_loss=} - {_lr=}')
    print('Done fine-tuning!')
    # Save with signatures
    tf.saved_model.save(model, f'{OUTPUTS_DIRS}', signatures={"serving_default": detect_fn})

if __name__ == '__main__':
    main()