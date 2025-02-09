from transformers import RobertaForTokenClassification

def load_model(label2id, id2label, pretrained_model="roberta-base"):
    model = RobertaForTokenClassification.from_pretrained(
        pretrained_model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    return model
