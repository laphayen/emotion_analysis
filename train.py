
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
import matplotlib.pyplot as plt

# 1. Load and preprocess data
df = pd.read_csv("emotion_dataset.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)

train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=64)
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 2. Load model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "./kobert_emotion_model"

if os.path.exists(checkpoint_path):
    print("🔁 기존 모델에서 이어서 학습합니다.")
    model = BertForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
    start_epoch = 6  # 기존 학습한 에폭 이후부터
else:
    print("🆕 새로운 모델로 학습을 시작합니다.")
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=len(le.classes_)).to(device)
    start_epoch = 0

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 3. Training loop
train_losses = []
val_losses = []
total_epochs = 5  # 추가로 몇 에폭 더 학습할지 설정

for epoch in range(start_epoch, start_epoch + total_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"✅ Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    # Validation loss
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"🧪 Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

    # Save model safely (Windows 호환)
    model.save_pretrained(checkpoint_path, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_path)
    pd.Series(le.classes_).to_csv(f"{checkpoint_path}/labels.csv", index=False)

# 4. Plot training vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(start_epoch, start_epoch + total_epochs), train_losses, label='Train Loss', marker='o')
plt.plot(range(start_epoch, start_epoch + total_epochs), val_losses, label='Validation Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
