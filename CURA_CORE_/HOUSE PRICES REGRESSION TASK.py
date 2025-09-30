import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score
import random
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


df = pd.read_csv(r'train2.csv')
x = df.drop(['SalePrice', 'Id', 'Utilities'], axis=1).values
y = df['SalePrice'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


class HousePriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HousePriceDataset(x_train, y_train)
test_dataset = HousePriceDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

def evaluate(model, loader):
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).view(-1)
            rmse = torch.sqrt(torch.mean((y - pred) ** 2)).item()
            total_loss += rmse * len(y)
            count += len(y)
    return total_loss / count



model = CURA_CORE(x_train.shape[1], 5, 1).to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
val_losses = []
best_loss = float("inf")
early_stop_count = 0
print("모델 파라미터 수:", sum(p.numel() for p in model.parameters() if p.requires_grad))
import torch
print(torch.cuda.memory_allocated() / 1024**2, "memory_allocated MB")
print(torch.cuda.memory_reserved() / 1024**2, "memory_reserved MB")
for epoch in range(10000):
    model.train()
    epoch_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X).view(-1)
        loss = criterion(pred, y) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    val_loss = evaluate(model, test_loader)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val RMSE: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_count = 0
        
        model.eval()
        with torch.no_grad():
            df_test = pd.read_csv(r'test2.csv')
            X_test = torch.tensor(df_test.drop(['Id', 'Utilities'], axis=1).values, dtype=torch.float32).to(device)
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_pred = np.expm1(y_pred)  
    else:
        early_stop_count += 1
        if early_stop_count >= 50:
            print("성능 개선 없어서 중단")
            break
    if val_loss < 0.1:
        print("목표 성능 도달")
        break


model.eval()

y_preds = []
y_trues = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch).view(-1)
        y_preds.extend(pred.cpu().numpy())
        y_trues.extend(y_batch.cpu().numpy())


y_preds = np.expm1(y_preds)
y_trues = np.expm1(y_trues)


r2 = r2_score(y_trues, y_preds)
print(f"R² Score on Test Set: {r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_trues, label='True Prices')
plt.plot(y_preds, label='Predicted Prices')
plt.title(f'House Price Prediction (CURA)\nR² Score: {r2:.4f}')
plt.xlabel('Sample Index')
plt.ylabel('Sale Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
