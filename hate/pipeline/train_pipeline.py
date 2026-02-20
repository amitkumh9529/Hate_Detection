import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.components.data_transformation import DataTransformation
from hate.components.model_trainer import ModelTrainer
from hate.components.model_evaluation import ModelEvaluation
from hate.components.model_pusher import ModelPusher

from hate.entity.config_entity import (DataIngestionConfig,
                                       DataTransformationConfig,
                                       ModelTrainerConfig,
                                       ModelEvaluationConfig,
                                       ModelPusherConfig)

from hate.entity.artifact_entity import (DataIngestionArtifacts,
                                         DataTransformationArtifacts,
                                         ModelTrainerArtifacts,
                                         ModelEvaluationArtifacts,
                                         ModelPusherArtifacts)



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()


    def start_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Starting data ingestion from local Data folder")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise CustomException(e, sys) from e


    def start_data_transformation(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise CustomException(e, sys) from e


    def start_model_trainer(
        self,
        data_transformation_artifacts: DataTransformationArtifacts
    ) -> ModelTrainerArtifacts:
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config=self.model_trainer_config
            )

            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise CustomException(e, sys) from e


    def start_model_evaluation(
        self,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts
    ) -> ModelEvaluationArtifacts:
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifacts=model_trainer_artifacts,
                data_transformation_artifacts=data_transformation_artifacts
            )

            return model_evaluation.initiate_model_evaluation()

        except Exception as e:
            raise CustomException(e, sys) from e


    def start_model_pusher(self) -> ModelPusherArtifacts:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config
            )

            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise CustomException(e, sys) from e


    def run_pipeline(self):

        try:
            logging.info("Pipeline execution started")

            data_ingestion_artifacts = self.start_data_ingestion()

            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts
            )

            model_trainer_artifacts = self.start_model_trainer(
                data_transformation_artifacts
            )

            model_evaluation_artifacts = self.start_model_evaluation(
                model_trainer_artifacts,
                data_transformation_artifacts
            )

            if not model_evaluation_artifacts.is_model_accepted:
                raise Exception("Trained model is not better than the best model")

            model_pusher_artifacts = self.start_model_pusher()

            logging.info("Pipeline execution completed")

            return model_pusher_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e