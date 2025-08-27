import pandas as pd
import torch 
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from mlflow import log_artifacts, start_run, log_params, log_metrics

from src.prepare_data import CustomDataset
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from src.calculate_metrics import compute_metrics
from src.track_loss import LossTrackerCallback, plot_losses

logger = get_logger(__name__)

class Model:

    def __init__(self, 
                 train_path, 
                 test_path, 
                 val_path, 
                 model_output_path):
        
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.model_output_path = model_output_path


    def load_data(self):
        try:
            logger.info("Loading the pre-processed data for training ")
            logger.info(f"Loading training data from {self.train_path}")
            train_df = pd.read_csv(self.train_path)
            logger.info(f"Loading testing data from {self.test_path}")
            test_df = pd.read_csv(self.test_path)
            logger.info(f"Loading validation data from {self.val_path}")
            val_df = pd.read_csv(self.val_path)

            logger.info("Data loading completed successfully.")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

            train_data = CustomDataset(
                texts = train_df['text'].to_list(),
                labels = train_df['label'].to_list(),
                tokenizer=tokenizer,
                max_length=MAX_LENGTH
            )

            test_data = CustomDataset(
                texts = test_df['text'].to_list(),
                labels = test_df['label'].to_list(),
                tokenizer=tokenizer,
                max_length=MAX_LENGTH
            )

            val_data = CustomDataset(
                texts = val_df['text'].to_list(),
                labels = val_df['label'].to_list(),
                tokenizer=tokenizer,
                max_length=MAX_LENGTH
            )
            return train_data, test_data, val_data, tokenizer

        except Exception as e:
            logger.error(f"Error occurred while loading data: {e}")
            raise CustomException("Data loading failed")
        

    def train_model(self):
        try:
            train_data, test_data, val_data, tokenizer = self.load_data()
            logger.info("Model training process initiated.")

            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=MODEL_NAME,
                num_labels=NUM_LABELS
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

            training_args = TrainingArguments(**TRAINING_ARGS)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[
                    LossTrackerCallback(),
                    EarlyStoppingCallback(early_stopping_patience=PATIENCE)
                ]
            )

            with start_run():
                log_params({
                    "model_name": MODEL_NAME,
                    "num_labels": NUM_LABELS,
                    "max_length": MAX_LENGTH,
                    "epochs": TRAINING_ARGS.get("num_train_epochs"),
                    "batch_size": TRAINING_ARGS.get("per_device_train_batch_size"),
                    "learning_rate": TRAINING_ARGS.get("learning_rate"),
                    "patience": PATIENCE
                })

                trainer.train()

                # Log metrics from evaluation
                eval_results = trainer.evaluate(test_data)
                log_metrics(eval_results)
                logger.info(f"Evaluation results: {eval_results}")

                # Plot and log training losses
                loss_tracker = trainer.callback_handler.callbacks[0]  # Assuming this is LossTrackerCallback
                plot_path = plot_losses(loss_tracker.train_losses, loss_tracker.eval_losses)
                log_artifacts(plot_path)  # plot_path should be a directory or file path

                # Save and log the model
                trainer.save_model(self.model_output_path)
                log_artifacts(self.model_output_path)

            logger.info("Model training and MLflow logging completed successfully.")

        except CustomException as e:
            logger.error(f"Error occurred during model training: {e}")
            raise CustomException("Model training failed", e)

        

    
