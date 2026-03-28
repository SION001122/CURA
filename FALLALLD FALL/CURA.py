
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

def load_and_window_csv_with_label(path, window_size=1800, stride=200):

    df = pd.read_csv(path)


    df['subject_id'] = df['source_file'].apply(lambda x: int(str(x).split('_')[0][1:]))

    df['activity_id'] = df['source_file'].apply(lambda x: int(str(x).split('_')[2][1:]))

    df['label'] = (df['activity_id'] >= 50).astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['subject_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)


    train_signal_raw = train_df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
    train_mean = np.mean(train_signal_raw, axis=0)
    train_std = np.std(train_signal_raw, axis=0) + 1e-7

    def extract_windows(data_df, mean, std, desc):
        signal = data_df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
        signal = (signal - mean) / std
        
        target_labels = data_df['label'].to_numpy()
        
        windows = []
        labels = []
        

        for i in tqdm(range(0, len(signal) - window_size + 1, stride), desc=desc):
            window = signal[i:i + window_size]

            label_mean = np.mean(target_labels[i:i + window_size])
            label = 1 if label_mean > 0.5 else 0
            
            windows.append(window)
            labels.append(label)
            
        return np.array(windows), np.array(labels)

    X_train, y_train = extract_windows(train_df, train_mean, train_std, "Extracting Train Windows")
    X_test, y_test = extract_windows(test_df, train_mean, train_std, "Extracting Test Windows")


    return X_train, y_train, X_test, y_test

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
import random
from sklearn.metrics import accuracy_score, f1_score

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = alpha * (1-pt)**gamma * BCE_loss
            return F_loss.mean()


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
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
if __name__ == "__main__":
        SEED = 34069  

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        csv_path = r"fall_waist_with_gyro.csv"
        X_train, y_train, X_test, y_test = load_and_window_csv_with_label(csv_path, window_size=1800, stride=200)

        train_loader = DataLoader(FallDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(FallDataset(X_test, y_test), batch_size=64)
        model = CURA_CORE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        print("Model parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        stopping_counter = 0
        maxcount = 10
        best_f1 = 0
        loss_list, acc_list, f1_list, sensitivity_list, specificity_list, epoch_list = [], [], [], [], [], []

        for epoch in tqdm(range(1, 100), desc="Training"):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = focal_loss(output.unsqueeze(1), y_batch.unsqueeze(1).float(), alpha=0.25, gamma=2.0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(device)
                        logits = model(X_batch)
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.4).int().cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(y_batch.numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)


            epoch_list.append(epoch)
            loss_list.append(total_loss)
            acc_list.append(acc)
            f1_list.append(f1)
            if f1 > best_f1:
                best_f1 = f1
                stopping_counter = 0
            else:
                stopping_counter += 1
                if stopping_counter >= maxcount:
                    print(f"Early stopping at epoch {epoch} with best F1: {best_f1:.4f}")
                    break
