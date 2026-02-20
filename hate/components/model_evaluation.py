import os
import sys
import keras
import pickle
import shutil
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts


    def evaluate_model(self, model_path):
        try:
            logging.info("Evaluating model")

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            with open(self.data_transformation_artifacts.tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)

            model = keras.models.load_model(model_path)

            x_test = x_test["tweet"].astype(str).squeeze()
            y_test = y_test.squeeze()

            sequences = tokenizer.texts_to_sequences(x_test)
            padded = pad_sequences(sequences, maxlen=MAX_LEN)

            loss, accuracy = model.evaluate(padded, y_test, verbose=0)

            predictions = model.predict(padded)
            y_pred = (predictions > 0.5).astype(int)

            cm = confusion_matrix(y_test, y_pred)
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{cm}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("Starting model evaluation")

            trained_model_path = self.model_trainer_artifacts.trained_model_path
            best_model_path = self.model_evaluation_config.BEST_MODEL_PATH

            trained_accuracy = self.evaluate_model(trained_model_path)

            # If no best model exists → accept automatically
            if not os.path.exists(best_model_path):
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                shutil.copy(trained_model_path, best_model_path)

                logging.info("No existing best model. Current model accepted.")
                return ModelEvaluationArtifacts(is_model_accepted=True)

            # Compare with existing best model
            best_accuracy = self.evaluate_model(best_model_path)

            if trained_accuracy > best_accuracy:
                shutil.copy(trained_model_path, best_model_path)
                logging.info("New model is better. Replaced best model.")
                is_model_accepted = True
            else:
                logging.info("Existing best model is better.")
                is_model_accepted = False

            return ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)

        except Exception as e:
            raise CustomException(e, sys) from e