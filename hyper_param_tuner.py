import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
from utils.logs import get_logger
from pathlib import Path

# import module for reading and updating configuration files.
from object_detection.utils import config_util

# import module for building the detection model
from object_detection.builders import model_builder
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm
from dotenv import load_dotenv
from utils.annotation_processor import AnnotationProcessor
load_dotenv()
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

label_map = {
    'License_Plate':1
}

prepare_train_dataset = AnnotationProcessor(annotation_file=TRAIN_ANNOT_FILE_PATH)

def create_model():
    pass

def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["nadam", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    
    if optimizer_selected == "nadam":
        kwargs["learning_rate"] = trial.suggest_float(
            "nadam_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("nadam_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("nadam_momentum", 1e-5, 1e-1, log=True)

    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)

    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

def learn(model, optimizer, datasets, mode='eval'):
    pass

def objective(trail):
    train_images, train_class_ids, train_bboxes  = prepare_train_dataset.process_annotations(image_dir=TRAIN_DIR, label_map=label_map)
    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

def main():
    log = get_logger(__name__, log_level=logging.INFO)
    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.error("No GPU found")
        raise Exception("No GPU found")
    log.info(f'{found_gpu=}, {tf.__version__=}')



if __name__ == "__main__":
    main()