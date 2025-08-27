import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import mlflow
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *

logger = get_logger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.model_name = "bert-base-uncased"
        self.max_len = 128
        self.num_labels = 6

    def load_data(self):
        try:
            logger.info("Loading and preprocessing data")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            train_dataset = TextDataset(
                texts=train_df["text"].tolist(),
                labels=train_df["label"].tolist(),
                tokenizer=tokenizer,
                max_len=self.max_len,
            )

            test_dataset = TextDataset(
                texts=test_df["text"].tolist(),
                labels=test_df["label"].tolist(),
                tokenizer=tokenizer,
                max_len=self.max_len,
            )

            return train_dataset, test_dataset, tokenizer
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise CustomException("Data loading failed", e)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        logger.info(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting BERT model training pipeline")

               
                train_dataset, test_dataset, tokenizer = self.load_data()

            
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

               
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=self.num_labels
                )

               
                training_args = TrainingArguments(
                    output_dir=self.model_output_path,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    num_train_epochs=3,
                    weight_decay=0.01,
                    logging_dir=os.path.join(self.model_output_path, "logs"),
                    logging_steps=10,
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    greater_is_better=True,
                    report_to="none",
                    
                )

                trainer = Trainer(
                    #accelerator="gpu",
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    compute_metrics=self.compute_metrics,
                )

                logger.info("Training the model...")
                trainer.train()

                logger.info("Evaluating the model...")
                metrics = trainer.evaluate()

                logger.info("Saving the final model and tokenizer")
                final_model_dir = os.path.join(self.model_output_path, "final_model")
                trainer.save_model(final_model_dir)
                tokenizer.save_pretrained(final_model_dir)

                logger.info("Logging model and metrics to MLflow")
                mlflow.log_metrics(metrics)
                mlflow.log_params(model.config.to_dict())
                mlflow.log_artifacts(final_model_dir, artifact_path="model")

                logger.info("Model training pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            raise CustomException("Model training pipeline failed", e)


if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
    )
    trainer.run()
