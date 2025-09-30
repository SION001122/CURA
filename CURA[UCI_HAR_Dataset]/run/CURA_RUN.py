import torch
from torch import nn, optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CURA_MODEL import CURASTACK
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import requests
import zipfile
import random
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def download_ucihar(dest_folder="data/UCI_HAR_Dataset"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = os.path.join(dest_folder, "UCI_HAR_Dataset.zip")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if not os.path.exists(zip_path):
        print(" Downloading UCI HAR Dataset...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    print(" Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    print(" Done.")


download_ucihar()


def load_har_dataset(batch_size=64):
    base_path = r"data\UCI_HAR_Dataset\UCI HAR Dataset"


    X_train = np.loadtxt(f"{base_path}/train/X_train.txt")
    y_train = np.loadtxt(f"{base_path}/train/y_train.txt") - 1
    X_test = np.loadtxt(f"{base_path}/test/X_test.txt")
    y_test = np.loadtxt(f"{base_path}/test/y_test.txt") - 1


    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 51, 11)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 51, 11)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = load_har_dataset(batch_size=64)


model = CURASTACK(seq_len=51*11, hidden_dim=4, output_dim=6, num_cores=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(f"Model: {model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
import torch
print(torch.cuda.memory_allocated() / 1024**2, "memory_allocated MB")
print(torch.cuda.memory_reserved() / 1024**2, "memory_reserved MB")
def evaluate(loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).long()
            x = x.view(x.size(0), -1)
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())


    try:
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")
    except Exception as e:
        print("F1 계산 실패:", e)
        acc = 0
        f1 = 0

    print(f"Total Loss: {total_loss / len(loader):.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return total_loss / len(loader), acc, f1

epochs = list(range(1, 50 + 1))  
train_losses, test_losses, accs, f1s = [], [], [], []
for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1) 
        logits = model(x) 
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    test_loss, acc, f1 = evaluate(test_loader)
    train_losses.append(loss.item())
    test_losses.append(test_loss)
    accs.append(acc)
    f1s.append(f1)  

import matplotlib.pyplot as plt


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.plot(epochs, accs, label='Accuracy')
plt.plot(epochs, f1s, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
best_f1 = max(f1s)
plt.title(f'Loss per Epoch (Best F1: {best_f1:.4f})')
plt.legend()
plt.grid(True)
plt.show()
