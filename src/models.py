"""
src/models.py

Contains 2 models:
1. MLPBaseline  - Simple fully connected regression network
2. CNNModel     - Convolutional regression network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# 1) MLP BASELINE MODEL
# ---------------------------------------------------
class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(50 * 50, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)  # predicts (x, y) normalized

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))  # ensure outputs are in [0,1]
        return x


# ---------------------------------------------------
# 2) CNN MODEL (BETTER)
# ---------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # reduces H,W by half

        # After 3 conv + pool layers, size becomes 50 → 25 → 12 → 6
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.out(x))
        return x


# ---------------------------------------------------
# TEST CODE
# ---------------------------------------------------
if __name__ == "__main__":
    print("Testing models...")

    dummy = torch.randn(4, 1, 50, 50)

    mlp = MLPBaseline()
    cnn = CNNModel()

    print("MLP output:", mlp(dummy).shape)
    print("CNN output:", cnn(dummy).shape)
