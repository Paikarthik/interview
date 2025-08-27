import traceback
import sys

import os
import pandas as pd

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    """
    Handles the ingestion of data from  source.
    """

    def __init__(self, config):
        self.config = config
        pass


    def split_data(self):
        """
        Splits the data into training,testing and valiadtion sets.
        """
        logger.info("Splitting the data into train, test and validation sets")
        pass 


    def download_csv_from_url(self):
        """
        Downloads the CSV file from the specified URL.
        """
        logger.info("Downloading the data from the source")
        pass


    def save_data(self):
        """
        Saves the ingested data to the specified directory.
        """
        logger.info("Saving the ingested data to the directory")
        pass


    def run(self):
        try:
            logger.info("Starting data ingestion")
            self.download_csv_from_url()
            logger.info("Downloaded the data successfully")
            self.split_data()
            logger.info("Data split into training, testing and validation sets successfully")
            self.save_data()
            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"Custom exception :{str(ce)}")

        finally:
            logger.info("Data ingestion completed")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()