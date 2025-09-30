import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score
import random
import numpy as np
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CURA_CORE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gate_fc = nn.Linear(input_dim, hidden_dim)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
        self.relu_linear = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_fc(x))
        residual = self.residual_fc(x)
        x = gate * residual+residual
        x = F.relu(self.relu_linear(x))
        x_cnn = self.conv(x.unsqueeze(1)).squeeze(1)
        out = self.output(x_cnn)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 50


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


input_dim = 28 * 28   
hidden_dim = 4
output_dim = 10

model = CURA_CORE(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.view(images.size(0), -1).to(device) 
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")


model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = 100 * correct / total


f1 = f1_score(all_labels, all_preds, average='macro')

print(f" Test Accuracy: {accuracy:.2f}%")
print(f" Macro F1-score: {f1:.4f}")





accuracy = 100 * correct / total


f1 = f1_score(all_labels, all_preds, average='macro')

print(f" Test Accuracy: {accuracy:.2f}%")
print(f"Macro F1-score: {f1:.4f}")


import matplotlib.pyplot as plt


metrics = ['Accuracy', 'F1 Score (Macro)']
values = [accuracy / 100, f1]  

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen'])
plt.ylim(0, 1.0)
plt.title('CURA on MNIST: Accuracy & F1 Macro Score')
plt.ylabel('Score')
plt.grid(axis='y')


for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

plt.tight_layout()
plt.show()
