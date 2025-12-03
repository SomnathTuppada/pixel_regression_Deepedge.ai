"""
src/train.py

Full training pipeline:
- Load data
- Train MLP or CNN
- Save best model
- Plot loss curves
- Visualize predictions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from models import MLPBaseline, CNNModel


# ---------------------------------------------------------------
# CHOOSE MODEL HERE
# ---------------------------------------------------------------
USE_CNN = True  # True = CNNModel, False = MLPBaseline


# ---------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{epochs}]  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(" ✔ Saved best model")

    return train_losses, val_losses


# ---------------------------------------------------------------
# VISUALIZE TRAINING CURVES
# ---------------------------------------------------------------
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid()
    plt.show()


# ---------------------------------------------------------------
# VISUALIZE PREDICTIONS vs GROUND TRUTH
# ---------------------------------------------------------------
def visualize_predictions(model, loader, device, n=10):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n].to(device), labels[:n].to(device)

    with torch.no_grad():
        preds = model(imgs)

    imgs = imgs.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    plt.figure(figsize=(6, 6))
    for i in range(n):
        plt.subplot(5, 2, i + 1)
        plt.imshow(imgs[i][0], cmap="gray")
        true_x = labels[i][0].item() * 49
        true_y = labels[i][1].item() * 49
        pred_x = preds[i][0].item() * 49
        pred_y = preds[i][1].item() * 49

        plt.scatter([true_x], [true_y], c="green", label="True")
        plt.scatter([pred_x], [pred_y], c="red", label="Pred")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    model = CNNModel().to(device) if USE_CNN else MLPBaseline().to(device)
    print("Training model:", model.__class__.__name__)

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=10,
        lr=1e-3
    )

    plot_losses(train_losses, val_losses)

    print("Visualizing Predictions...")
    visualize_predictions(model, test_loader, device)
