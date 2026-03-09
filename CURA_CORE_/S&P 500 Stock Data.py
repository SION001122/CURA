import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

scaler = StandardScaler()
train_scaled_values = scaler.fit_transform(train_df[['Close']].values)
test_scaled_values = scaler.transform(test_df[['Close']].values)

def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled_values, window_size=20)
X_test, y_test = create_sequences(test_scaled_values, window_size=20)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).squeeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=64)

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
model = CURA_CORE(seq_len=20, hidden_dim=13).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("학습 시작...")
for epoch in range(110):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/110]")

# 5. 평가 및 시각화
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
mae = mean_absolute_error(y_true_inv, preds_inv)
mse = mean_squared_error(y_true_inv, preds_inv)

print(f"\n--- Test Set Evaluation ---")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
