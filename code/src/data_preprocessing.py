import os 
import pandas as pd
from sklearn.utils import resample
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from src.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessing:

    def __init__(self, train_path, test_path, val_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.validation_path = val_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        
        
    def preprocess_data(self, df):
        """
        Pre-processes the data by:df['text'] = df['text'] = df['text'].str.split()df['text'].str.split()
         - Lower casing the text
         - Removing special characters
         - Tokenizing the text
        """
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(r'\W', ' ', regex=True)

        return df

    def balance_data(self, df):
        '''
        '''
        try: 
            logger.info("Handling imbalance data")
            class_counts = df['label'].value_counts()
            average_count = int(class_counts.mean())

            balanced_df = []

            for label, count in class_counts.items():
                class_subset = df[df['label'] == label]
                if count < average_count:
                
                    oversampled = resample(
                        class_subset,
                        replace=True,
                        n_samples=average_count,
                        random_state=42
                    )
                    balanced_df.append(oversampled)
                else:
                    
                    balanced_df.append(class_subset)

            result_df = pd.concat(balanced_df).sample(frac=1, random_state=42).reset_index(drop=True)
            return result_df
        except Exception as e :
            logger.error(f"Error during balancing data {e}")
            raise CustomException("Error while data balancing", e)
        

    def save_data(self, df, file_path):
        try:
            logger.info("Saving processed data in processed data")
            df.to_csv(file_path, index = False)
            logger.info(f"Data saved successfully in {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data {e}")
            raise CustomException("Error while saving processed data", e)
        

    def process(self):
        try:
            logger.info("Loading data from raw directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            val_df = load_data(self.validation_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            val_df = self.preprocess_data(val_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            val_df = self.balance_data(val_df)

            self.save_data(val_df, PROCESSED_VAL_DATA_PATH)
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully")
        except Exception as e:
            logger.error(f"Error during pre-processing pipeline execution {e}")
            raise CustomException("Error while data pre-processing pipeline execution", e)
        


if __name__ == "__main__":
    data_processor = DataPreprocessing(
        train_path=TRAIN_FILE_PATH, 
        test_path=TEST_FILE_PATH, 
        val_path=VALIDATION_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
        )
    data_processor.process()
             

            



                
