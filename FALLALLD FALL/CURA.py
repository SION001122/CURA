
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def load_and_window_csv_with_label(path, window_size=1800, stride=200):
    df = pd.read_csv(path)
    signal = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    raw_labels = df['label'].to_numpy()

    windows = []
    labels = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i + window_size]
        label_window = raw_labels[i:i + window_size]
        label = 1 if np.mean(label_window) > 0.5 else 0
        windows.append(window)
        labels.append(label)
    return np.array(windows), np.array(labels)

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



import torch.nn as nn
import torch.nn.functional as F

class CURA_CORE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=1):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, hidden_dim)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C)
        gate = torch.sigmoid(self.gate_fc(x))
        residual = self.residual_fc(x)
        x = gate * residual + residual
        x = F.relu(self.relu_linear(x))
        x = x.view(B, T, -1).transpose(1, 2)
        x_cnn = self.conv(x).mean(dim=2)
        return self.output(x_cnn).squeeze()

import random
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_and_window_csv_with_label("fall_waist_with_gyro.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(FallDataset(X_test, y_test), batch_size=64)

    model = CURA_CORE().to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Model parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Train label count:", torch.bincount(torch.tensor(y_train).int()))
    print("Test label count:", torch.bincount(torch.tensor(y_test).int()))


    loss_list, acc_list, f1_list, sensitivity_list, specificity_list, epoch_list = [], [], [], [], [], []

    for epoch in range(1, 100):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.6).int().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            print(f"[Epoch {epoch}] Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

            epoch_list.append(epoch)
            loss_list.append(total_loss)
            acc_list.append(acc)
            f1_list.append(f1)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            if f1_list[-1] > 0.8:
                print("Early stopping at epoch", epoch)
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, acc_list, marker='o', label='Accuracy')
    plt.plot(epoch_list, f1_list, marker='s', label='F1-score')
    plt.title("Accuracy & F1-score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
