import json
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from scripts.data_preprocessing import create_dataset, load_test_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model, evaluate_test_model
from scripts.inference import run_inference

print("Loading labels")
with open("config/labels.json", "r") as f:
    labels = json.load(f)

label2id = labels["label2id"]
id2label = labels["id2label"]

print("Instantiate the tokenizer")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

print("Load the model")
model = RobertaForTokenClassification.from_pretrained(
    "roberta-base", num_labels=len(label2id), label2id=label2id, id2label=id2label
)

# Load datasets
print("Loading datasets...")
train_data = create_dataset("data/train.txt", tokenizer, label2id)
val_data = create_dataset("data/val.txt", tokenizer, label2id)
test_data = load_test_data("data/test.txt")

# Check the structure of your data (optional)
print(train_data)
print(f"Train dataset size: {len(train_data)}")
print("Sample tokenized example:", tokenizer.encode("Lahari"))

# Ensure your dataset is in the expected format: {'input_ids', 'attention_mask', 'labels'}
for i, example in enumerate(train_data):
    print(f"Example {i}:")
    print(f"  Input IDs: {example['input_ids']}")
    print(f"  Attention Mask: {example['attention_mask']}")
    print(f"  Labels: {example['labels']}")
    print("=" * 50)
 

# Train the model
print("Training the model...")
trainer = train_model(model, train_data, val_data, tokenizer)

# Save the fine-tuned model
print("Saving the model...")
trainer.save_model("./ner_model")
tokenizer.save_pretrained("./ner_model")

# Evaluate on the validation dataset
print("Evaluating on the validation set...")
val_results = evaluate_model(trainer, val_data)
print("Validation Metrics:")
print(f"Precision: {val_results['precision']:.4f}")
print(f"Recall: {val_results['recall']:.4f}")
print(f"F1 Score: {val_results['f1']:.4f}")

# # Display the classification report
# print("\nValidation Classification Report:")
# for label, metrics in val_results["classification_report"].items():
#     if isinstance(metrics, dict):  # Skip "accuracy" key
#         print(f"{label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

# Test the model
print("\nTesting the model on the test set...")
test_predictions = []
test_labels = []

for tokens, true_tags in zip(test_data["tokens"], test_data["tags"]):
    text = " ".join(tokens)  # Convert token list to a single sentence
    
    predictions = run_inference(text, model_dir="./ner_model")
    #print(f"Predictions Output: {predictions}")

    predicted_entities = [pred["entity_group"] for pred in predictions]
    
    print("\nToken - True Label - Predicted Label")
    for token, true_label, predicted_label in zip(tokens, true_tags, predicted_entities):
        print(f"{token} -  {true_label} - {predicted_label}")

    test_predictions.append(predicted_entities)
    test_labels.append(true_tags)


