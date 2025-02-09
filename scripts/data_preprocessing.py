import json
from transformers import RobertaTokenizerFast

def load_conll_data(file_path):
    tokens, tags = [], []
    with open(file_path, "r") as f:
        sentence, sentence_tags = [], []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    tokens.append(sentence)
                    tags.append(sentence_tags)
                    sentence, sentence_tags = [], []
            else:
                word, tag = line.split()
                sentence.append(word)
                sentence_tags.append(tag)
        if sentence:
            tokens.append(sentence)
            tags.append(sentence_tags)
    return tokens, tags

def tokenize_and_align_labels(examples, tokenizer, label2id, max_len=128):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=max_len
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label2id[label[word_id]] for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_dataset(file_path, tokenizer, label2id):
    tokens, tags = load_conll_data(file_path)
    dataset = {"tokens": tokens, "tags": tags}

    # Tokenizing the dataset and aligning the labels
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer, label2id)

    # Convert to a list of dictionaries for each sample
    final_dataset = []
    for i in range(len(tokenized_dataset["input_ids"])):
        final_dataset.append({
            "input_ids": tokenized_dataset["input_ids"][i],
            "attention_mask": tokenized_dataset["attention_mask"][i],
            "labels": tokenized_dataset["labels"][i]
        })

    return final_dataset

def load_test_data(file_path):
    test_data = {"tokens": [], "tags": []}
    tokens = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":  # Empty line indicates a new sentence
                if tokens:
                    test_data["tokens"].append(tokens)
                    test_data["tags"].append(tags)
                    tokens = []
                    tags = []
            else:
                parts = line.rsplit(" ", 1)  # Split into token and tag
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    tags.append(tag)
    
    # Append last sentence if not added
    if tokens:
        test_data["tokens"].append(tokens)
        test_data["tags"].append(tags)

    return test_data