"""
ST-GCN 分类器训练模块。

可从命令行运行或作为函数调用。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.perception.st_gcn import STGCN
from src.data.dataset import get_dataloader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.ndim == 5:
            x = x.squeeze(-1)

        optimizer.zero_grad()
        logits = model(x)                      # (N, C, T)
        N, C, T = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(N * T, C)
        targets = y.unsqueeze(1).expand(-1, T).reshape(-1)
        loss = criterion(logits_flat, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits_flat.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if x.ndim == 5:
            x = x.squeeze(-1)

        logits = model(x)
        N, C, T = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(N * T, C)
        targets = y.unsqueeze(1).expand(-1, T).reshape(-1)
        loss = criterion(logits_flat, targets)

        total_loss += loss.item()
        preds = logits_flat.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def train_model(data_dir: str, train_csv: str, test_csv: str = None,
                window_size: int = 32, batch_size: int = 32, epochs: int = 50,
                lr: float = 0.001, save_path: str = "outputs/models/best_model.pt",
                device: str = None):
    """完整训练流程，返回最佳验证准确率。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_dataloader(data_dir, train_csv, batch_size, "train", window_size)
    val_csv = test_csv if test_csv else train_csv
    val_loader = get_dataloader(data_dir, val_csv, batch_size, "test", window_size)

    model = STGCN(num_classes=7, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved ({best_acc:.4f})")

    print(f"Training complete. Best val acc: {best_acc:.4f}")
    return best_acc
