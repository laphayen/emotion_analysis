
# train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.optim import AdamW
from dataset import EmotionDataset
from model import load_model
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

# 2. Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(num_labels=len(le.classes_)).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 3. Training loop
for epoch in range(3):
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

# 4. Save model and label encoder
model.save_pretrained("./kobert_emotion_model")
tokenizer.save_pretrained("./kobert_emotion_model")
pd.Series(le.classes_).to_csv("kobert_emotion_model/labels.csv", index=False)