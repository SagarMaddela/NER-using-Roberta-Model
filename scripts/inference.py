from transformers import pipeline

def run_inference(text, model_dir="./ner_model"):
    ner_pipeline = pipeline("token-classification", model=model_dir, tokenizer=model_dir, aggregation_strategy="simple")
    return ner_pipeline(text)
