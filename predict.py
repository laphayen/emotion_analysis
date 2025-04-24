
# predict.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load
model = BertForSequenceClassification.from_pretrained("./kobert_emotion_model")
tokenizer = BertTokenizer.from_pretrained("./kobert_emotion_model")
labels = pd.read_csv("./kobert_emotion_model/labels.csv", header=None)[0].tolist()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

# 예시
print(predict_emotion("나 지금 기뻐"))  # → 슬픔 (예상)