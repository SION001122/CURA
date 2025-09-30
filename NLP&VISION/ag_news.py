from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class AGNewsDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_length=128):
        self.data = split_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }

train_dataset = AGNewsDataset(dataset["train"], tokenizer)
val_dataset = AGNewsDataset(dataset["test"], tokenizer)


sample = train_dataset[0]
sample["input_ids"].shape, sample["attention_mask"].shape, sample["label"]



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
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim))

    def forward(self, input_ids):
        x = self.E2(self.E1(input_ids))
        x = x + self.pos_emb[:, :x.size(1)]
        return x




class CURAsformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=180, num_blocks=3, num_heads=2, dropout=0.1, low_rank_dim=32, max_len=128):
        super().__init__()
        self.embedding = LowRankEmbedding(vocab_size, embed_dim, rank=low_rank_dim, max_len=max_len)
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            CURAsformerBlock(embed_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pooling = AttentionPooling(embed_dim)
        self.final_fc = nn.Linear(embed_dim, 4)  

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) 
        for block in self.blocks:
            x = block(x)

        x = self.pooling(x)
        logits = self.final_fc(x)  
        return logits


from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = CURAsformer(vocab_size=tokenizer.vocab_size).to(device)
from ptflops import get_model_complexity_info


dummy_input_shape = (128,)

def forward_hook(model, input_res):

    return dict(input_ids=torch.randint(0, tokenizer.vocab_size, (1, *input_res)))

def forward_hook(input_res):
    seq_len = input_res[0]
    return {
        "input_ids": torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
    }

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model,
        input_res=(128,), 
        input_constructor=forward_hook,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"MACs: {macs}")
    print(f"Params: {params}")


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=1800, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1800)
from sklearn.metrics import f1_score
f1s = []
def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).long()

        logits = model(input_ids, attention_mask) 
        loss = criterion(logits, labels)     

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")


    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).long()

            logits = model(input_ids, attention_mask)  
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    f1s.append(compute_f1(all_labels, all_preds))
    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1s[-1]:.4f}")
    
best_f1 = max(f1s)
print(f"Best F1 Score: {best_f1:.4f}")
