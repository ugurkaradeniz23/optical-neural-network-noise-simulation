# optical_layer.py
# CustomOpticalLayer: MATLAB'den gelen sigma değerleriyle gürültü enjeksiyonu

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# --- LOOKUP TABLE YÜKLEME ---
lut = pd.read_csv('mzi_lookup_table.csv')

def get_sigma_static(weight_val):
    """Ağırlık değerine karşılık gelen sigma_static'i CSV'den interpolasyonla bul."""
    weight_np = weight_val.detach().cpu().numpy().flatten()
    sigma = np.interp(weight_np,
                      lut['mu_weight'].values,
                      lut['sigma_static'].values)
    return torch.tensor(sigma, dtype=torch.float32).reshape(weight_val.shape)

def get_sigma_dynamic(weight_val):
    """Ağırlık değerine karşılık gelen sigma_dynamic'i CSV'den interpolasyonla bul."""
    weight_np = weight_val.detach().cpu().numpy().flatten()
    sigma = np.interp(weight_np,
                      lut['mu_weight'].values,
                      lut['sigma_dynamic'].values)
    return torch.tensor(sigma, dtype=torch.float32).reshape(weight_val.shape)


class CustomOpticalLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomOpticalLayer, self).__init__(in_features, out_features, bias)

        # --- STATİK GÜRÜLTÜ: Çip üretildi, MZI'ların kaderi belli ---
        # __init__'de bir kez uygulanır, bir daha değişmez
        with torch.no_grad():
            sigma_s = get_sigma_static(self.weight)
            static_noise = torch.randn_like(self.weight) * sigma_s
            self.weight.add_(static_noise)

    def forward(self, x):
        # --- DİNAMİK GÜRÜLTÜ: Her çıkarımda anlık termal dalgalanma ---
        # Her forward pass'te yeniden örneklenir
        sigma_d = get_sigma_dynamic(self.weight).to(x.device)
        dynamic_noise = torch.randn_like(self.weight) * sigma_d
        noisy_weight = self.weight + dynamic_noise

        return F.linear(x, noisy_weight, self.bias)