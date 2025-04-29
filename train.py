import os
import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from kobert import get_pytorch_kobert_model, get_tokenizer 
from gluonnlp.data import SentencepieceTokenizer
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# 데이터 로드 및 전처리
df = pd.read_csv("emotion_dataset1.csv", encoding='cp949', sep="\t")
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])

# Tokenizer 준비
tokenizer_path = get_tokenizer()
sp_tokenizer = SentencepieceTokenizer(tokenizer_path)

# Dataset 클래스
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text)
        token_ids = [self.vocab['[CLS]']] + [self.vocab[token] if token in self.vocab.token_to_idx else self.vocab['[UNK]'] for token in tokens] + [self.vocab['[SEP]']]

        token_ids = token_ids[:self.max_len]
        attention_mask = [1] * len(token_ids)
        pad_len = self.max_len - len(token_ids)
        token_ids += [self.vocab['[PAD]']] * pad_len
        attention_mask += [0] * pad_len

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# KoBERT 모델 로딩
model, vocab = get_pytorch_kobert_model()
print("✅ KoBERT 모델 로드 성공!")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)

train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist(), sp_tokenizer, vocab)
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist(), sp_tokenizer, vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델 classifier 설정
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size, len(le.classes_))
).to(device)

# Optimizer 및 Scheduler 설정
optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

checkpoint_path = "./kobert_emotion_model"
os.makedirs(checkpoint_path, exist_ok=True)
model_ckpt = os.path.join(checkpoint_path, "best_model.bin")
optimizer_ckpt = os.path.join(checkpoint_path, "optimizer.pt")
metadata_file = os.path.join(checkpoint_path, "metadata.json")
best_val_loss = float('inf')

if os.path.exists(metadata_file):
    print("🔁 기존 모델 + 옵티마이저 상태 복구하여 이어서 학습합니다.")
    if os.path.exists(model_ckpt):
        model.load_state_dict(torch.load(model_ckpt))
    if os.path.exists(optimizer_ckpt):
        optimizer.load_state_dict(torch.load(optimizer_ckpt))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    start_epoch = metadata.get("last_epoch", 0)
else:
    print("🆕 새로운 모델로 학습을 시작합니다.")
    start_epoch = 0

loss_fn = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
epochs = 1
patience = 4
early_stop_counter = 0

for epoch in range(start_epoch, start_epoch + epochs):
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = model.classifier(outputs[1])
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(epoch + batch_idx / len(train_loader))

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"✅ Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = model.classifier(outputs[1])
            loss = loss_fn(logits, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"🧪 Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_ckpt)
        torch.save(optimizer.state_dict(), optimizer_ckpt)
        pd.DataFrame(list(le.classes_)).to_csv(os.path.join(checkpoint_path, "labels.csv"), index=False, header=False)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({"last_epoch": epoch, "best_val_loss": best_val_loss}, f)
        print(f"🎉 모델 저장 완료 (Epoch: {epoch}, Val Loss: {best_val_loss:.4f})")

# Loss 시각화
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()


"""
epochs 0~7까지
optimizer =>  lr=1e-6, weight_decay=0.02
factor=0.5, patience=2
Dropout(0.6)
patience = 4

----- 

epochs 8~13까지
optimizer =>  lr=1e-6, weight_decay=0.03
factor=0.7, patience=3
Dropout(0.7)
patience = 4

------

epochs 14~19까지
optimizer =>  lr=1e-6, weight_decay=0.04
factor=0.5, patience=3
Dropout(0.7)
patience = 4


"""