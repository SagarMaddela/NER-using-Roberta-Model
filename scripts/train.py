from transformers import Trainer, TrainingArguments

print("Started train.py")

def train_model(model, train_dataset, val_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./ner_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer

print("Ended Train.py")