import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
from sklearn.model_selection import train_test_split
from hate.logger import logging 
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    

    def imbalanced_data_cleaning(self):

        try:
            logging.info("Entered into the imbalance_data_cleaning function")
            imbalanced_data=pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            cols_to_drop = [c for c in self.data_transformation_config.drop_columns if c in imbalanced_data.columns]
            imbalanced_data.drop(columns=cols_to_drop, inplace=True)
            logging.info(f"Exited the imbalance data_cleaning function and returned imbalance data {imbalanced_data}")
            return imbalanced_data 
        except Exception as e:
            raise CustomException(e,sys) from e 
        
    

    def raw_data_cleaning(self):
        
        try:
            logging.info("Entered into the raw_data_cleaning function")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            cols_to_drop = [c for c in self.data_transformation_config.drop_columns if c in raw_data.columns]
            raw_data.drop(columns=cols_to_drop, inplace=True)

            class_col = self.data_transformation_config.class_column
            raw_data.loc[raw_data[class_col] == 0, class_col] = 1
            # replace the value of 0 to 1, then 2 to 0
            raw_data[class_col] = raw_data[class_col].replace({0: 1, 2: 0})

            # Let's change the name of the 'class' to label
            raw_data.rename(columns={self.data_transformation_config.class_column:self.data_transformation_config.label},inplace =True)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {raw_data}")
            return raw_data

        except Exception as e:
            raise CustomException(e,sys) from e
        

    
    def concat_dataframe(self):

        try:
            logging.info("Entered into the concat_dataframe function")
            # Let's concatinate both the data into a single data frame.
            frame = [self.raw_data_cleaning(), self.imbalanced_data_cleaning()]
            df = pd.concat(frame)
            print(df.head())
            logging.info(f"returned the concatinated dataframe {df}")
            return df

        except Exception as e:
            raise CustomException(e, sys) from e
        
    

    def concat_data_cleaning(self, words):

        try:
            logging.info("Entered into the concat_data_cleaning function")
            # Let's apply stemming and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub(r'\[.*?\]', '', words)
            words = re.sub(r'https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub(r'\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            logging.info("Exited the concat_data_cleaning function")
            return words 

        except Exception as e:
            raise CustomException(e, sys) from e
        

    

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            self.imbalanced_data_cleaning()
            self.raw_data_cleaning()
            df = self.concat_dataframe()
            df[self.data_transformation_config.tweet]=df[self.data_transformation_config.tweet].apply(self.concat_data_cleaning)

            os.makedirs(self.data_transformation_config.transformation_dir, exist_ok=True)
            df.to_csv(self.data_transformation_config.transformed_file_path,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.transformed_file_path,
                tokenizer_path=self.data_transformation_config.tokenizer_path,
            )
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e