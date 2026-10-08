# train.py
# CNN modelini MNIST üzerinde eğit ve kaydet

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# --- CİHAZ AYARI ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Kullanılan cihaz: {device}')

# --- VERİ YÜKLEME ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST ortalama ve std
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- MODEL ---
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- EĞİTİM FONKSİYONU ---
def train_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 200 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f}')

    accuracy = 100. * correct / total
    print(f'--- Epoch {epoch} Eğitim Doğruluğu: %{accuracy:.2f} ---')


# --- TEST FONKSİYONU ---
def test(label='Temiz Model'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    print(f'\n[{label}] Test Doğruluğu: %{accuracy:.2f}\n')
    return accuracy


# --- ANA EĞİTİM DÖNGÜSÜ ---
N_EPOCHS = 5

for epoch in range(1, N_EPOCHS + 1):
    train_epoch(epoch)
    test()

# --- MODELİ KAYDET ---
torch.save(model.state_dict(), 'clean_model.pth')
print('Model kaydedildi: clean_model.pth')