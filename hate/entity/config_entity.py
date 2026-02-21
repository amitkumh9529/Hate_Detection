from dataclasses import dataclass, field
from hate.constants import *
import os


# =====================================================
# Data Ingestion Config
# =====================================================

@dataclass
class DataIngestionConfig:
    data_dir: str = DATA_DIR
    raw_data_file: str = RAW_DATA_FILE
    imbalanced_data_file: str = IMBALANCED_DATA_FILE
    data_ingestion_dir: str = DATA_INGESTION_DIR
    ingested_raw_data_path: str = INGESTED_RAW_DATA_PATH
    ingested_imbalanced_data_path: str = INGESTED_IMBALANCED_DATA_PATH


# =====================================================
# Data Transformation Config
# =====================================================

@dataclass
class DataTransformationConfig:
    transformation_dir: str = os.path.join(os.getcwd(), DATA_TRANSFORMATION_DIR)
    transformed_file_path: str = TRANSFORMED_FILE_PATH
    tokenizer_path: str = TOKENIZER_FILE_PATH
    drop_columns: list = field(default_factory=lambda: DROP_COLUMNS.copy())
    label: str = LABEL
    tweet: str = TWEET
    class_column: str = CLASS


# =====================================================
# Model Trainer Config
# =====================================================

@dataclass
class ModelTrainerConfig:
    trainer_dir: str = os.path.join(os.getcwd(), MODEL_TRAINER_DIR)
    trained_model_path: str = TRAINED_MODEL_PATH
    x_test_path: str = X_TEST_PATH
    y_test_path: str = Y_TEST_PATH
    x_train_path: str = X_TRAIN_PATH

    max_words: int = MAX_WORDS
    max_len: int = MAX_LEN
    loss: str = LOSS
    metrics: list = field(default_factory=lambda: METRICS.copy())
    activation: str = ACTIVATION
    random_state: int = RANDOM_STATE
    epoch: int = EPOCH
    batch_size: int = BATCH_SIZE
    validation_split: float = VALIDATION_SPLIT


# =====================================================
# Model Evaluation Config
# =====================================================

@dataclass
class ModelEvaluationConfig:
    evaluation_dir: str = os.path.join(os.getcwd(), MODEL_EVALUATION_DIR)
    best_model_path: str = os.path.join(os.getcwd(), BEST_MODEL_PATH)


# =====================================================
# Model Pusher Config
# =====================================================

@dataclass
class ModelPusherConfig:
    best_model_path: str = os.path.join(os.getcwd(), BEST_MODEL_PATH)
    production_model_path: str = os.path.join(os.getcwd(), PRODUCTION_MODEL_PATH)