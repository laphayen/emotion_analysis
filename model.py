
# model.py

from transformers import BertForSequenceClassification

def load_model(num_labels=6):
    return BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=num_labels)