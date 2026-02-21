import os
from datetime import datetime

# =====================================================
# Common Constants
# =====================================================

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)

DATA_DIR = "data"   # Your actual folder (case-sensitive)
RAW_DATA_FILE = "raw_data.csv"
IMBALANCED_DATA_FILE = "imbalanced_data.csv"

LABEL = "label"
TWEET = "tweet"

# =====================================================
# Data Ingestion
# =====================================================

DATA_INGESTION_DIR = os.path.join(ARTIFACTS_DIR, "data_ingestion")
INGESTED_RAW_DATA_PATH = os.path.join(DATA_INGESTION_DIR, RAW_DATA_FILE)
INGESTED_IMBALANCED_DATA_PATH = os.path.join(DATA_INGESTION_DIR, IMBALANCED_DATA_FILE)

# =====================================================
# Data Transformation
# =====================================================

DATA_TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, "data_transformation")
TRANSFORMED_FILE_NAME = "final.csv"
TRANSFORMED_FILE_PATH = os.path.join(DATA_TRANSFORMATION_DIR, TRANSFORMED_FILE_NAME)
TOKENIZER_FILE_NAME = "tokenizer.pickle"
TOKENIZER_FILE_PATH = os.path.join(DATA_TRANSFORMATION_DIR, TOKENIZER_FILE_NAME)

DROP_COLUMNS = ['Unnamed: 0', 'id', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = "class"

# =====================================================
# Model Training
# =====================================================

MODEL_TRAINER_DIR = os.path.join(ARTIFACTS_DIR, "model_trainer")
TRAINED_MODEL_PATH = os.path.join(MODEL_TRAINER_DIR, "model.h5")

X_TEST_PATH = os.path.join(MODEL_TRAINER_DIR, "x_test.csv")
Y_TEST_PATH = os.path.join(MODEL_TRAINER_DIR, "y_test.csv")
X_TRAIN_PATH = os.path.join(MODEL_TRAINER_DIR, "x_train.csv")

RANDOM_STATE = 42
EPOCH = 1
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# =====================================================
# Model Architecture
# =====================================================

MAX_WORDS = 50000
MAX_LEN = 300
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
ACTIVATION = "sigmoid"

# =====================================================
# Model Evaluation
# =====================================================

MODEL_EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "model_evaluation")

BEST_MODEL_DIR = "artifacts/best_model"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "model.h5")

# =====================================================
# Model Pusher (Production)
# =====================================================

PRODUCTION_MODEL_DIR = "saved_models"
PRODUCTION_MODEL_PATH = os.path.join(PRODUCTION_MODEL_DIR, "model.h5")

# =====================================================
# App
# =====================================================

APP_HOST = "0.0.0.0"
APP_PORT = 8080