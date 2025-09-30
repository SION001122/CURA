from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
SEED = 1024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = load_dataset("hellaswag")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class HellaSwagDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.data = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ctx = item['ctx']
        choices = item['endings']
        label = int(item['label'])


        encoded = self.tokenizer(
            [ctx + " " + choice for choice in choices],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],      
            "attention_mask": encoded["attention_mask"],  
            "label": torch.tensor(label)
        }

train_dataset = HellaSwagDataset(dataset["train"], tokenizer)
val_dataset = HellaSwagDataset(dataset["validation"], tokenizer)

class CURAsformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=4, dropout=0.1):
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

        x = self.ffn_norm(x + self.dropout(x_ffn))
        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)

class LowRankEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, rank=32, max_len=128):
        super().__init__()
        self.E1 = nn.Embedding(vocab_size, rank)
        self.E2 = nn.Linear(rank, hidden_dim)

    def apply_rope(self, x):

        B, L, D = x.size()
        assert D % 2 == 0, "Embedding dim must be even for RoPE"

        half = D // 2
        x1 = x[..., :half]  
        x2 = x[..., half:]


        device = x.device
        dim_t = torch.arange(0, half, 2, device=device).float()
        freqs = 1.0 / (10000 ** (dim_t / half))  

        pos = torch.arange(L, device=device).float().unsqueeze(1) 
        angles = pos * freqs.unsqueeze(0)  

        sin = angles.sin().repeat_interleave(2, dim=1)  
        cos = angles.cos().repeat_interleave(2, dim=1) 

        sin = sin.unsqueeze(0).expand(B, -1, -1) 
        cos = cos.unsqueeze(0).expand(B, -1, -1) 

        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        return torch.cat([x1_new, x2_new], dim=-1)


    def forward(self, input_ids):
        x = self.E2(self.E1(input_ids))  
        x = self.apply_rope(x)        
        return x

    
    
class CURAsformerForMultipleChoice(nn.Module): 
    def __init__(self, vocab_size, embed_dim=192, hidden_dim=270, num_blocks=6, num_heads=6, dropout=0.15, low_rank_dim=32, max_len=128):
        super().__init__()
        self.embedding = LowRankEmbedding(vocab_size, embed_dim, rank=low_rank_dim, max_len=max_len)
        self.blocks = nn.ModuleList([
            CURAsformerBlock(embed_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.pooling = AttentionPooling(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask=None):

        B, num_choices, L = input_ids.shape
        input_ids = input_ids.view(-1, L) 
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, L)

        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)

        x = self.pooling(x)
        logits = self.classifier(x)
        return logits.view(B, num_choices)



from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CURAsformerForMultipleChoice(vocab_size=tokenizer.vocab_size).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {num_params}")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

from sklearn.metrics import f1_score
def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

f1s = []
for epoch in range(10): 
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)       
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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

    f1s.append(compute_f1(all_labels, all_preds))
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1s[-1]:.4f}")

print(f"Best F1 Score: {max(f1s):.4f}")
