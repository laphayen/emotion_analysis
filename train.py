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
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv("emotion_dataset1.csv", encoding='cp949', sep="\t")
le = LabelEncoder()
df['label'] = le.fit_transform(df['emotion'])

# KoBERT tokenizer
tokenizer_path = get_tokenizer()
sp_tokenizer = SentencepieceTokenizer(tokenizer_path)

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=64):
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

        # 토크나이징
        tokens = self.tokenizer(text)
        token_ids = [self.vocab['[CLS]']] + [self.vocab[token] for token in tokens] + [self.vocab['[SEP]']]

        # 패딩
        tokens = self.tokenizer(text)
        token_ids = [self.vocab['[CLS]']] + [self.vocab[token] for token in tokens] + [self.vocab['[SEP]']]
        
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

# 모델과 vocab 불러오기
model, vocab = get_pytorch_kobert_model()
print("✅ KoBERT 모델 로드 성공!")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'], df['label'], test_size=0.2, random_state=42
)

train_dataset = EmotionDataset(train_texts.tolist(), train_labels.tolist(), sp_tokenizer, vocab)
val_dataset = EmotionDataset(val_texts.tolist(), val_labels.tolist(), sp_tokenizer, vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 2. Load model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dropout 추가 및 classifier 설정 변경
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size, len(le.classes_))
).to(device)

checkpoint_path = "./kobert_emotion_model"
model_ckpt = os.path.join(checkpoint_path, "pytorch_model.bin")
optimizer_ckpt = os.path.join(checkpoint_path, "optimizer.pt")
metadata_file = os.path.join(checkpoint_path, "metadata.json")

# 학습률 조정
optimizer = AdamW(model.parameters(), lr=1e-5)

start_epoch = 0
best_val_loss = float('inf')

if os.path.exists(metadata_file):
    print("🔁 기존 모델 + 옵티마이저 상태 복구하여 이어서 학습합니다.")
    if os.path.exists(model_ckpt):
        model.load_state_dict(torch.load(model_ckpt))
    if os.path.exists(optimizer_ckpt):
        optimizer.load_state_dict(torch.load(optimizer_ckpt))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    start_epoch = metadata.get("last_epoch", 0) + 1
    best_val_loss = metadata.get("best_val_loss", float('inf'))
else:
    print("🆕 새로운 모델로 학습을 시작합니다.")
    os.makedirs(checkpoint_path, exist_ok=True)

loss_fn = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
epochs_to_run = 5

for epoch in range(start_epoch, start_epoch + epochs_to_run):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = model.classifier(outputs[1])
        loss = loss_fn(logits, labels)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    # Validation Loss가 최소일 때만 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_ckpt)
        torch.save(optimizer.state_dict(), optimizer_ckpt)
        pd.DataFrame(list(le.classes_)).to_csv(os.path.join(checkpoint_path, "labels.csv"), index=False, header=False)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({"last_epoch": epoch, "best_val_loss": best_val_loss}, f)
        print(f"🎉 모델 저장 완료 (Epoch: {epoch}, Val Loss: {best_val_loss:.4f})")

# Plotting loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(start_epoch, start_epoch + epochs_to_run), train_losses, label='Train Loss', marker='o')
plt.plot(range(start_epoch, start_epoch + epochs_to_run), val_losses, label='Validation Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
