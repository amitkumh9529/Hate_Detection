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


    def get_data_from_local(self) -> DataIngestionArtifacts:
        try:
            os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)

            raw_source = os.path.join(
                self.data_ingestion_config.data_dir,
                self.data_ingestion_config.raw_data_file
            )

            imbalance_source = os.path.join(
                self.data_ingestion_config.data_dir,
                self.data_ingestion_config.imbalanced_data_file
            )

            shutil.copy(raw_source, self.data_ingestion_config.ingested_raw_data_path)
            shutil.copy(imbalance_source, self.data_ingestion_config.ingested_imbalanced_data_path)

            return DataIngestionArtifacts(
                raw_data_file_path=self.data_ingestion_config.ingested_raw_data_path,
                imbalance_data_file_path=self.data_ingestion_config.ingested_imbalanced_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self):
            return self.get_data_from_local()
    # def initiate_data_ingestion(self) -> DataIngestionArtifacts:
    #     logging.info("Entered initiate_data_ingestion method")

    #     try:
    #         self.get_data_from_local()

    #         raw_data_file_path = self.data_ingestion_config.data_ingestion_dir

    #         data_ingestion_artifacts = DataIngestionArtifacts(
    #             imbalance_data_file_path=raw_data_file_path,
    #             raw_data_file_path=raw_data_file_path
    #         )

    #         logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

    #         return data_ingestion_artifacts

    #     except Exception as e:
    #         raise CustomException(e, sys) from e