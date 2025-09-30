import sys
import os
import torch
from torch import nn, optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

SEED = 210
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from CURA import CURA_CORE
from utils import load_ettdataset
from datadownload import download_ettm1
download_ettm1()
seq_len, pred_len = 96, 24
input_dim = 7
hidden_dim = 1

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  


csv_path = os.path.join(base_dir,"run", "data", "ETT-small", "ETTm1.csv")

train_loader, val_loader, test_loader, mean, std = load_ettdataset(
    csv_path,
    seq_len, pred_len, batch_size=64, feature_type="s", target="OT"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CURA_CORE(
    seq_len=seq_len * input_dim,
    hidden_dim=hidden_dim,
    output_dim=pred_len,
    num_cores=1
).to(device)
print(f"Model: {model.__class__.__name__}, Total params: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def evaluate(loader, mean, std):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            B, T, C = x.shape
            x, y = x.to(device), y.to(device)
            x_flat = x.view(B, T * C)
            y_target = y[:, :, -1]           
            pred = model(x_flat)              
            if pred.dim() == 3:
                pred = pred.squeeze(-1)
            pred_real = pred * std[0] + mean[0]
            y_real = y_target * std[0] + mean[0]
            all_preds.extend(pred_real.cpu().numpy().flatten())
            all_targets.extend(y_real.cpu().numpy().flatten())
            total_loss += criterion(pred, y_target).item()
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return total_loss / len(loader), mae, r2, mse, all_targets, all_preds


train_losses, val_losses = [], []
val_maes, val_r2s, val_mses = [], [], []
val_losses, val_maes, val_r2s, val_mses = [], [], [], []
best_model_state_CURA = None
best_epoch_CURA = 0
best_val_r2_CURA = float('-inf')
best_val_mae_CURA = float('inf')
best_val_mse_CURA = float('inf')
best_model_params_CURA = sum(p.numel() for p in model.parameters())

patience = 5  
patience_counter = 0

for epoch in range(100):
    model.train()
    total_train_loss = 0
    for x, y in train_loader:
        B, T, C = x.shape
        x, y = x.to(device), y.to(device)
        x_flat = x.view(B, T * C)
        y_target = y[:, :, -1]  
        pred = model(x_flat)
        loss = criterion(pred, y_target)
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    val_loss, val_mae, val_r2, val_mse, _, _ = evaluate(val_loader, mean, std)
    val_losses.append(val_loss)
    val_maes.append(val_mae)
    val_r2s.append(val_r2)
    val_mses.append(val_mse)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f} | MSE: {val_mse:.4f}")


    if val_r2 > best_val_r2_CURA:
        best_model_state_CURA = model.state_dict()
        best_val_r2_CURA = val_r2
        best_val_mae_CURA = val_mae
        best_val_mse_CURA = val_mse
        best_epoch_CURA = epoch + 1
        patience_counter = 0
        print(f"Best model updated at epoch {epoch + 1}")
    else:
        patience_counter += 1


    if patience_counter >= patience:
        print("Early stopping triggered")
        break


model.load_state_dict(best_model_state_CURA)
test_loss, test_mae, test_r2, test_mse, all_targets, all_preds = evaluate(test_loader, mean, std)

print("\n========== Final Results ==========")
print(f"Best Val R²: {best_val_r2_CURA:.4f} at Epoch {best_epoch_CURA}")
print(f"Best Val MAE: {best_val_mae_CURA:.4f} at Epoch {best_epoch_CURA}")
print(f"Best Val MSE: {best_val_mse_CURA:.4f} at Epoch {best_epoch_CURA}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Total parameters: {best_model_params_CURA}")
