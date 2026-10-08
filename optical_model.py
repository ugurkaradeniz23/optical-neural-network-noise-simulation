# optical_model.py
# Temiz CNN'in FC katmanlarını CustomOpticalLayer ile değiştir

import torch
import torch.nn as nn
import torch.nn.functional as F
from optical_layer import CustomOpticalLayer

class OpticalCNN(nn.Module):
    def __init__(self):
        super(OpticalCNN, self).__init__()

        # Conv katmanları aynı kalıyor
        # Bunlar optik çipte değil, dijital ön işlemcide çalışıyor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)

        # FC katmanları CustomOpticalLayer ile değiştirildi
        # Bunlar optik çipte MZI'larla hesaplanıyor
        self.fc1 = CustomOpticalLayer(64 * 7 * 7, 128)
        self.fc2 = CustomOpticalLayer(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x