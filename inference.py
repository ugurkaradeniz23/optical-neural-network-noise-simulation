# inference.py
# Senaryo A vs Senaryo B karşılaştırması

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
from optical_model import OpticalCNN
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- VERİ ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False,
                               download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- TEST FONKSİYONU ---
def test_model(model, label):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred   = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    accuracy = 100. * correct / total
    print(f'[{label}] Test Doğruluğu: %{accuracy:.2f}')
    return accuracy

# --- SENARYO A: Temiz Dijital Model ---
clean_model = CNN().to(device)
clean_model.load_state_dict(torch.load('clean_model.pth', map_location=device))
acc_clean = test_model(clean_model, 'Senaryo A - Temiz Dijital Model')

# --- SENARYO B: Gürültülü Optik Model ---
# Eğitilmiş ağırlıkları yükle, sonra optik katmana geçir
optical_model = OpticalCNN().to(device)

# Conv katmanlarının ağırlıklarını temiz modelden al
optical_model.conv1.load_state_dict(clean_model.conv1.state_dict())
optical_model.conv2.load_state_dict(clean_model.conv2.state_dict())

# FC katmanları CustomOpticalLayer — zaten __init__'de gürültü eklendi
# Önce temiz ağırlıkları yükle
optical_model.fc1.weight.data = clean_model.fc1.weight.data.clone()
optical_model.fc2.weight.data = clean_model.fc2.weight.data.clone()

# Sonra statik gürültüyü manuel ekle
with torch.no_grad():
    from optical_layer import get_sigma_static, get_sigma_dynamic

    sigma_s1 = get_sigma_static(optical_model.fc1.weight).to(device)
    optical_model.fc1.weight.add_(torch.randn_like(optical_model.fc1.weight) * sigma_s1)

    sigma_s2 = get_sigma_static(optical_model.fc2.weight).to(device)
    optical_model.fc2.weight.add_(torch.randn_like(optical_model.fc2.weight) * sigma_s2)

acc_optical = test_model(optical_model, 'Senaryo B - Gürültülü Optik Model')

# --- SONUÇ ---
print(f'\nDoğruluk Düşüşü: %{acc_clean - acc_optical:.2f}')

# --- GRAFİK ---
plt.figure(figsize=(8, 5))
bars = plt.bar(['Senaryo A\nTemiz Dijital Model',
                'Senaryo B\nOptik Hızlandırıcı'],
               [acc_clean, acc_optical],
               color=['#2196F3', '#FF5722'], width=0.4)

for bar, acc in zip(bars, [acc_clean, acc_optical]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 0.5,
             f'%{acc:.2f}', ha='center', va='top',
             color='white', fontweight='bold', fontsize=12)

plt.ylim([90, 100])
plt.ylabel('Test Doğruluğu (%)')
plt.title('ONN vs Dijital Model — Doğruluk Karşılaştırması')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
#plt.show()
print('Grafik kaydedildi: accuracy_comparison.png')