import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = yf.download('^GSPC', start='2010-01-01', end='2024-12-31')
df = df[['Close']].dropna()


scaler = StandardScaler()
scaled_close = scaler.fit_transform(df[['Close']].values)


def create_sequences(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_close, window_size=20)
print("시퀀스 생성 완료:", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).squeeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class CURA_CORE(nn.Module):
    def __init__(self, seq_len, hidden_dim, output_dim=1):
        super().__init__()
        self.gate_fc = nn.Linear(seq_len, hidden_dim)
        self.residual_fc = nn.Linear(seq_len, hidden_dim)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.output = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        gate = torch.sigmoid(self.gate_fc(x))         
        residual = self.residual_fc(x)                
        x = gate * residual + residual                          
        x = F.relu(self.relu_linear(x))               
        x_cnn = self.conv(x.unsqueeze(1)).squeeze(1)  
        out = self.output(x_cnn)                      
        return out.squeeze()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CURA_CORE(seq_len=20, hidden_dim=13, output_dim=1).to(device)

print("모델 파라미터 수:", sum(p.numel() for p in model.parameters() if p.requires_grad))
import torch
print(torch.cuda.memory_allocated() / 1024**2, "memory_allocated MB")
print(torch.cuda.memory_reserved() / 1024**2, "memory_reserved MB")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
for epoch in range(110):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1:02d} - Train Loss: {avg_loss:.4f}")


model.eval()
with torch.no_grad():
    preds = []
    y_true = []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).cpu().numpy()
        preds.extend(pred)
        y_true.extend(y_batch.numpy())


preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
y_true_inv = scaler.inverse_transform(np.array(y_true).reshape(-1, 1))


r2 = r2_score(y_true_inv, preds_inv)
print(f" R² Score on Test Set: {r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_true_inv, label='True')
plt.plot(preds_inv, label='Predicted')
plt.title(f'S&P 500 Close Price Prediction (CURA_CORE)\nR² Score: {r2:.4f}')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
