
# emotion_classifier.py
# 감정 분석을 위한 모델과 토크나이저를 불러오는 코드입니다.

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# 감정 레이블 (클래스 수는 모델에 따라 다를 수 있음)
LABELS = ['기쁨', '당황', '분노', '불안', '상처', '슬픔']

class EmotionClassifier:
    def __init__(self, model_name='klue/roberta-base'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS))
        self.model.eval()

    def predict(self, sentence: str) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            return LABELS[pred_idx]