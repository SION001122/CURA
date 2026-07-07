if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from tqdm import tqdm
    import numpy as np, random, os
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
            x = gate * residual + residual
            x = F.relu(self.relu_linear(x))
            x_cnn = self.conv(x.unsqueeze(1)).squeeze(1)
            out = self.output(x_cnn)
            return out


    class CURA_ViTStack(nn.Module):
        def __init__(self, num_classes=10, image_size=64, patch_size=8, hidden_dim=128, num_cores=3):
            super().__init__()
            self.patch_size = patch_size
            num_patches = (image_size // patch_size) ** 2
            patch_dim = 3 * patch_size * patch_size
            self.patch_embed = nn.Linear(patch_dim, hidden_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            layers = [CURA_CORE(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_cores)]
            self.cores = nn.ModuleList(layers)
            self.norm = nn.LayerNorm(hidden_dim)
            self.fc = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            B, C, H, W = x.shape
            P = self.patch_size
            patches = x.unfold(2, P, P).unfold(3, P, P)
            patches = patches.contiguous().view(B, C, -1, P, P)
            patches = patches.permute(0, 2, 1, 3, 4).reshape(B, -1, 3 * P * P)
            tokens = self.patch_embed(patches) + self.pos_embed
            for core in self.cores:
                out_list = []
                for i in range(tokens.size(1)): 
                    patch_feat = tokens[:, i, :]  
                    patch_out = core(patch_feat) 
                    out_list.append(patch_out.unsqueeze(1))
                tokens = torch.cat(out_list, dim=1)

            pooled = tokens.mean(dim=1)
            out = self.fc(self.norm(pooled))
            return out
    data_dir = r".\tiny-imagenet-200"
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),      
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                            (0.2302, 0.2265, 0.2262))])
    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                            (0.2302, 0.2265, 0.2262))])
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True)
    def get_model(name, num_classes):
        if name == "CURA":
            model = CURA_ViTStack(
                num_classes=num_classes,
                image_size=64,
                patch_size=8,
                hidden_dim=512,
                num_cores=2
            )
            print("CURA Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
            return model
        else:
            raise ValueError("Unknown model name")
    model_names = ["CURA"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_epochs = 10000
    results = {}
    for name in model_names:
        print(f"\n Training {name} ...")
        model = get_model(name, len(train_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        train_losses, val_accs, val_f1s = [], [], []
        best_acc = 0.0 
        maxcount = 20
        count = 0
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{num_epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            model.eval()
            correct, total = 0, 0
            preds_all, labels_all = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    preds_all.extend(preds.cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())
            acc = correct / total
            f1 = f1_score(labels_all, preds_all, average="macro")
            if acc > best_acc:
                best_acc = acc
                val_accs.append(acc)
                val_f1s.append(f1)
                count = 0
            else:
                print(f"not best acc")
                count += 1
                print(f"count: {count}/{maxcount}")
                if count >= maxcount:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print(f"[{name}] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

        results[name] = {
            "Acc": max(val_accs),
            "F1": max(val_f1s),
            "Params(M)": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        }
    print("\nSummary Results (Tiny-ImageNet)")
    print(f"{'Model':<20} {'Params(M)':<12} {'Acc(%)':<10} {'F1(%)':<10}")
    print("-" * 55)
    for name, res in results.items():
        acc_pct = res['Acc'] * 100
        f1_pct = res['F1'] * 100
        print(f"{name:<20} {res['Params(M)']:<12.2f} {acc_pct:<10.2f} {f1_pct:<10.2f}")
