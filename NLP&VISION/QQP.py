from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random

SEED = 124789
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm

dataset = load_dataset("glue", "qqp")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class QQPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q1 = item["question1"]
        q2 = item["question2"]
        label = int(item["label"])

        encoding = self.tokenizer(
            q1,
            q2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class CURAsformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        
        self.gate_fc = nn.Linear(dim, hidden_dim)
        self.residual_fc = nn.Linear(dim, hidden_dim)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ffn_out = nn.Linear(hidden_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.attn_norm(x + self.dropout(attn_output))
        gate = torch.sigmoid(self.gate_fc(x))
        residual = self.residual_fc(x)
        x_ffn = gate * residual + residual
        x_ffn = F.relu(self.relu_linear(x_ffn))
        x_ffn = self.conv(x_ffn.transpose(1, 2)).transpose(1, 2)
        x_ffn = self.ffn_out(x_ffn)
        return self.ffn_norm(x + self.dropout(x_ffn))

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

class LowRankEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, rank=32, max_len=128):
        super().__init__()
        self.E1 = nn.Embedding(vocab_size, rank)
        self.E2 = nn.Linear(rank, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim))

    def forward(self, input_ids):
        x = self.E2(self.E1(input_ids))
        x = x + self.pos_emb[:, :x.size(1)]
        return x

class CURAsformer(nn.Module): 
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=150, num_blocks=4, num_heads=8, dropout=0.2, low_rank_dim=32, max_len=128):
        super().__init__()
        self.embedding = LowRankEmbedding(vocab_size, embed_dim, rank=low_rank_dim, max_len=max_len)
        self.blocks = nn.ModuleList([
            CURAsformerBlock(embed_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.pooling = AttentionPooling(embed_dim)
        self.final_fc = nn.Linear(embed_dim, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        return self.final_fc(x)


train_dataset = QQPDataset(dataset["train"], tokenizer)
val_dataset = QQPDataset(dataset["validation"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CURAsformer(vocab_size=tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M")
def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

f1s = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = compute_f1(all_labels, all_preds)
    f1s.append(f1)
    print(f"Validation Accuracy: {acc:.4f} | F1: {f1:.4f}")

print(f"Best F1 Score: {max(f1s):.4f}")
