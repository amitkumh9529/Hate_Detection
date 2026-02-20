import os
import sys
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config


    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered initiate_model_pusher method")

        try:
            source_model_path = self.model_pusher_config.BEST_MODEL_PATH
            production_model_path = self.model_pusher_config.PRODUCTION_MODEL_PATH

            if not os.path.exists(source_model_path):
                raise Exception("Best model not found. Cannot push to production.")

            os.makedirs(os.path.dirname(production_model_path), exist_ok=True)

            shutil.copy(source_model_path, production_model_path)

            logging.info("Model successfully pushed to production directory.")

            model_pusher_artifact = ModelPusherArtifacts(
                production_model_path=production_model_path
            )

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e