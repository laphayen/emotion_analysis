
# train.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from dataset import EmotionDataset
from tqdm import tqdm

# 1. Load and preprocess data
df = pd.read_csv("emotion_dataset.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
train_texts, val_texts, train_labels, val_labels = train_test_split(df['sentence'], df['label'], test_size=0.2, random_state=42)

train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=64)
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 2. Load model and optimizer (resume or fresh)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "./kobert_emotion_model"
if os.path.exists(checkpoint_path):
    print("🔁 기존 모델에서 이어서 학습합니다.")
    model = BertForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    start_epoch = 6  # 지난 학습이 0~5였다면, 이어서 6부터
else:
    print("🆕 새로운 모델로 학습을 시작합니다.")
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=len(le.classes_)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    start_epoch = 0

loss_fn = torch.nn.CrossEntropyLoss()

# 3. Training loop (이어하기)
for epoch in range(start_epoch, start_epoch + 1):  # 예: 이어서 1에폭 더
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

    # Save after each epoch (선택)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    pd.Series(le.classes_).to_csv(f"{checkpoint_path}/labels.csv", index=False)
