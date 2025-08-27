import os 



RAW_DIR = "artifacts/raw_data"
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")
VALIDATION_FILE_PATH = os.path.join(RAW_DIR, "validation.csv")

CONFIG_PATH = "config/config.yaml"


# data processing 

PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")
PROCESSED_VAL_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_validation.csv")

# model training 

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"