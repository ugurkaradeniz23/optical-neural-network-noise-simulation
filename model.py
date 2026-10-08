# model.py
# Adım 1: Temiz CNN Modeli (Gürültüsüz)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # --- Evrişim Katmanları ---
        # Giriş: 1x28x28 (MNIST: gri tonlamalı, 28x28 piksel)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Çıkış: 32x28x28

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Çıkış: 64x14x14 (MaxPool sonrası)

        # --- Tam Bağlantılı Katmanlar ---
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 sınıf (0-9 rakamları)

        # --- Düzenlileştirme ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))  # 32x14x14

        # Conv2 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # 64x7x7

        # Düzleştir
        x = x.view(-1, 64 * 7 * 7)

        # FC1 → ReLU → Dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # FC2 → Çıkış (10 sınıf)
        x = self.fc2(x)

        return x