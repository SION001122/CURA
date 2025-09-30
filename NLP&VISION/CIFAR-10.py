import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CURA_CORE(nn.Module):
    def __init__(self, hidden_dim1=32, output_dim=10):
        super().__init__()

        self.conv_core = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten_dim = 256 * 4 * 4



        self.gate_fc1 = nn.Linear(self.flatten_dim, hidden_dim1)
        self.residual_fc1 = nn.Linear(self.flatten_dim, hidden_dim1)
        self.relu_linear1 = nn.Linear(hidden_dim1, hidden_dim1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(hidden_dim1)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim1, output_dim)
        )

    def forward(self, x):
        x = self.conv_core(x)       
        x = x.view(x.size(0), -1) 

        gate1 = torch.sigmoid(self.gate_fc1(x))
        residual1 = self.residual_fc1(x)
        x1 = gate1 * residual1 + residual1
        x1 = F.relu(self.relu_linear1(x1))

        x1 = self.conv1(x1.unsqueeze(1)).squeeze(1)
        x1 = self.norm(x1)
        x1 = self.dropout(x1)

        return self.output(x1)



from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2470, 0.2435, 0.2616)) 
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_dim = 3 * 32 * 32  
model = CURA_CORE(hidden_dim1=32, output_dim=10).to(device)
from ptflops import get_model_complexity_info

model2 = CURA_CORE(hidden_dim1=32, output_dim=10)

with torch.cuda.device(0): 
    macs, params = get_model_complexity_info(
        model2, input_res=(3, 32, 32),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
print(f"FLOPs (MACs): {macs}")
print(f"Parameters: {params}")


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5, amsgrad=True)

train_losses = []
test_accuracies = []
num_epochs = 130

print("모델 파라미터 수:", sum(p.numel() for p in model.parameters() if p.requires_grad))

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    test_accuracies.append(acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_losses[-1]:.4f} - Acc: {acc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Enhanced CURA_CORE on CIFAR-10 (Accuracy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
