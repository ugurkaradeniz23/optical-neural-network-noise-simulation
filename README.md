# Optical Neural Network Hardware Noise Simulation

A simulation study of how hardware imperfections affect the classification accuracy of optical neural networks (ONNs). The optical layers are built on Mach-Zehnder interferometer (MZI) behavior, characterized through a lookup table, and the networks are trained and evaluated on the MNIST handwritten digit dataset.

The repository contains two ONN versions that share the same optical layer implementation but use different network architectures, so their robustness to hardware noise can be compared.

## Repository Structure

```
.
├── MZI/                 # MZI characterization (MATLAB / Simulink)
│   ├── untitled.slx
│   ├── untitled5.mlx
│   └── mzi_lookup_table.csv
├── onn_project/         # ONN version 1
│   ├── model.py
│   ├── optical_layer.py
│   ├── optical_model.py
│   ├── train.py
│   ├── inference.py
│   ├── clean_model.pth
│   └── accuracy_comparison.png
└── onn_project_cnn/     # ONN version 2 (CNN-based)
    ├── model.py
    ├── optical_layer.py
    ├── optical_model.py
    ├── train.py
    ├── inference.py
    ├── clean_model.pth
    └── accuracy_comparison.png
```

## How It Works

1. **MZI characterization:** The MZI behavior is modeled in MATLAB/Simulink and exported as `mzi_lookup_table.csv`.
2. **Optical layer:** `optical_layer.py` implements the optical layer using the MZI lookup table.
3. **Training:** `train.py` trains a clean (noise-free) model, saved as `clean_model.pth`.
4. **Noise simulation:** `inference.py` evaluates the trained model with hardware noise applied to the optical layer and compares the resulting accuracy against the clean baseline.
5. **Comparison:** The results are plotted in `accuracy_comparison.png`.

## Requirements

- Python 3.10+
- PyTorch and torchvision
- NumPy
- Matplotlib

```bash
pip install torch torchvision numpy matplotlib
```

MATLAB with Simulink is only needed if you want to open or modify the files in `MZI/`. The generated lookup table is already included.

## Usage

The MNIST dataset is downloaded automatically on the first run, so an internet connection is needed once.

```bash
# Version 1
cd onn_project
python train.py
python inference.py

# Version 2 (CNN)
cd onn_project_cnn
python train.py
python inference.py
```

Pretrained clean weights (`clean_model.pth`) are included, so you can run `inference.py` directly without retraining.

## Results

Accuracy comparison between the clean model and the model under simulated hardware noise:

**Version 1**

<img width="650" height="340" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/ac7950f9-7131-4af7-b1b7-3011a2e3dcb0" />


**Version 2 (CNN)**

<img width="650" height="340" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/658dfee1-5705-4435-b345-aab92e81298d" />




| Model | Clean accuracy | Noisy accuracy |
|-------|----------------|----------------|
| Version 1 | 98.7% | 97.07% |
| Version 2 (CNN) | 99.23% | 96.28% |

## Tech Stack

Python, PyTorch, MATLAB / Simulink

## Author

Uğur Karadeniz — [GitHub](https://github.com/ugurkaradeniz23)
