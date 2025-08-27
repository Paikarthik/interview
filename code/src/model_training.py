import pandas as pd
import torch 
from transformers import AutoTokenizer

from src.prepare_data import CustomDataset
from src.logger import get_logger
from src.custom_exception import CustomException
from src.training_class import Model
from config.paths_config import *
from config.model_params import *

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, 
                 train_path, 
                 test_path, 
                 val_path, 
                 model_output_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.model_output_path = model_output_path

    def run(self):
        try:
            model_class = Model(
                train_path = self.train_path,
                test_path = self.test_path,
                val_path = self.val_path,
                model_output_path = self.model_output_path
            )
            model_class.train_model()

        except Exception as e:
            logger.error(f"Error occurred during model training: {e}")
            raise CustomException("Error occurred during model training:", e)

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path = PROCESSED_TRAIN_DATA_PATH,
        test_path = PROCESSED_TEST_DATA_PATH,
        val_path = PROCESSED_VAL_DATA_PATH,
        model_output_path = MODEL_OUTPUT_PATH
        )
    trainer.run()