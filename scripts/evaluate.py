import json
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load the labels from the labels.json file
labels_path = os.path.join(os.path.dirname(__file__), '../config/labels.json')
with open(labels_path, "r") as f:
    labels = json.load(f)

id2label = labels["id2label"]



def evaluate_model(trainer, test_dataset):
    print(test_dataset)
    predictions, labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)
    return compute_metrics(predictions, labels, id2label)




def compute_metrics(predictions, labels, id2label):
    true_labels = []
    pred_labels = []
    for label_seq, pred_seq in zip(labels, predictions):
        for label, pred in zip(label_seq, pred_seq):
            if label != -100:
                # label_int = int(label)
                # pred_int = int(pred)
                label_int = str(label)
                pred_int = str(pred)
                print(f"checking for {label} and {pred} - {label_int} and {pred_int}")
                true_labels.append(id2label[label_int])
                pred_labels.append(id2label[pred_int])

    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    # Generate classification report
    report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

def evaluate_test_model(test_labels,test_predictions):

    # print(f"Test_labels : {test_labels}")
    # print(f"predicted_labels : {test_predictions}")
    # Convert the labels into a binarized format
    mlb = MultiLabelBinarizer()

    # Fit and transform your multi-label sequences
    true_labels_bin = mlb.fit_transform(test_labels)
    pred_labels_bin = mlb.transform(test_predictions)

    # Now, evaluate
    print(classification_report(true_labels_bin, pred_labels_bin, target_names=mlb.classes_, digits=4))

    # Calculate precision, recall, and F1-score
    print("\nNER Model Performance:")
    print(classification_report(test_labels, test_predictions, digits=4))