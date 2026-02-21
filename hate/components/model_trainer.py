import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts,DataTransformationArtifacts
from hate.ml.model import ModelArchitecture



class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,
                model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    
    def spliting_data(self,csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 42)
            print(len(x_train),len(y_train))
            print(len(x_test),len(y_test))
            print(type(x_train),type(y_train))
            logging.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test

        except Exception as e:
            raise CustomException(e, sys) from e
        

    
    def tokenizing(self,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.max_words)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.max_len)
            logging.info(f" The sequence matrix is: {sequences_matrix}")
            return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e
        

    

    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Entered the initiate_model_trainer function ")
            x_train,x_test,y_train,y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()   

            model = model_architecture.get_model()



            logging.info(f"Xtrain size is : {x_train.shape}")

            logging.info(f"Xtest size is : {x_test.shape}")

            sequences_matrix,tokenizer =self.tokenizing(x_train)


            logging.info("Entered into model training")
            model.fit(sequences_matrix, y_train,
                        batch_size=self.model_trainer_config.batch_size,
                        epochs=self.model_trainer_config.epoch,
                        validation_split=self.model_trainer_config.validation_split,
                        )
            logging.info("Model training finished")

            
            tokenizer_path = self.data_transformation_artifacts.tokenizer_path
            os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            logging.info("saving the model")
            model.save(self.model_trainer_config.trained_model_path)
            x_test.to_csv(self.model_trainer_config.x_test_path)
            y_test.to_csv(self.model_trainer_config.y_test_path)
            x_train.to_csv(self.model_trainer_config.x_train_path)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.trained_model_path,
                x_test_path=self.model_trainer_config.x_test_path,
                y_test_path=self.model_trainer_config.y_test_path,
                x_train_path=self.model_trainer_config.x_train_path,
            )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

