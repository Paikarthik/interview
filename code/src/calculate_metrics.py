from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
    }

    for label, scores in report.items():
        if label.isdigit():  
            metrics[f"precision_class_{label}"] = scores["precision"]
            metrics[f"recall_class_{label}"] = scores["recall"]
            metrics[f"f1_class_{label}"] = scores["f1-score"]

    return metrics
