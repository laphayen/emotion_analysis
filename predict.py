import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load
model = BertForSequenceClassification.from_pretrained("./kobert_emotion_model")
tokenizer = BertTokenizer.from_pretrained("./kobert_emotion_model")
labels = pd.read_csv("./kobert_emotion_model/labels.csv", header=None).squeeze().tolist()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]  # ✅ 인덱스를 라벨로 변환

# 예시
print("✅ 예측 시작")
print(f"🧠 감정 예측 결과: {predict_emotion('아니 이게 맞아??')}")  # 예: "기쁨"
