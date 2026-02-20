import os
import sys
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config


    def get_data_from_local(self) -> None:
        try:
            logging.info("Entered the get_data_from_local method")

            # Create artifact directory
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            # Copy CSV files from local data folder to artifacts folder
            source_dir = self.data_ingestion_config.LOCAL_DATA_DIR
            destination_dir = self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR

            for file_name in os.listdir(source_dir):
                if file_name.endswith(".csv"):
                    shutil.copy(
                        os.path.join(source_dir, file_name),
                        os.path.join(destination_dir, file_name)
                    )

            logging.info("Local data copied successfully")

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered initiate_data_ingestion method")

        try:
            self.get_data_from_local()

            raw_data_file_path = self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=raw_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e