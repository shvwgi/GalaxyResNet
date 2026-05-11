# Galaxy Zoo 2 Classification

Deep learning model for morphological classification of galaxies using the Galaxy Zoo 2 dataset.

## Overview

This project implements a convolutional neural network to classify galaxy morphologies based on citizen science voting data from Galaxy Zoo 2. The model predicts probability distributions across three primary morphological classes: smooth, featured/disk, and artifact.

## Dataset

- **Source**: Galaxy Zoo 2 (GZ2)
- **Size**: 167,434 galaxy images
- **Image Resolution**: 224×224 pixels (RGB)
- **Labels**: Crowd-sourced vote fractions across morphological features

## Model Architecture

**GalaxyCNN**: Custom 4-block convolutional architecture
- **Parameters**: 51.8M
- **Input**: 224×224×3 normalized images
- **Output**: 3-class probability distribution
- **Loss Function**: KL Divergence (vote fraction matching)

### Architecture Details
```
Block 1: Conv(3→32) → BatchNorm → ReLU → MaxPool  [224→112]
Block 2: Conv(32→64) → BatchNorm → ReLU → MaxPool  [112→56]
Block 3: Conv(64→128) → BatchNorm → ReLU → MaxPool [56→28]
Block 4: Conv(128→256) → BatchNorm → ReLU → MaxPool [28→14]
Classifier: Flatten → Linear(50176→1024) → ReLU → Dropout(0.5) → Linear(1024→3)
```

## Results

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.1264 |
| Final Training Loss | 0.0898 |
| Training Samples | 1,600 |
| Validation Samples | 400 |
| Training Time | ~6 minutes |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU training)
- See `requirements.txt` for full dependencies

## Installation

```bash
# Clone repository
cd GZ2

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
# Open and run the notebook
jupyter notebook GalaxyZoo2Model.ipynb
```

The notebook will:
1. Download the GZ2 dataset to `./data/`
2. Train the model on CUDA if available
3. Save checkpoints to `./models/`
4. Generate visualizations in `./plots/`

### Inference

```python
import torch
from galaxy_datasets.pytorch import GZ2

# Load trained model
checkpoint = torch.load('./models/best_model.pth')
model = GalaxyCNN(num_classes=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Project Structure

```
GZ2/
├── GalaxyZoo2Model.ipynb    # Main training notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Dataset (auto-downloaded)
│   ├── images_gz2/
│   ├── gz2_train_catalog.parquet
│   └── gz2_test_catalog.parquet
├── models/                   # Saved model checkpoints
│   ├── best_model.pth
│   └── final_model.pth
└── plots/                    # Training visualizations
    ├── learning_curve.png
    └── predictions.png
```

## Hardware

Tested on:
- **GPU**: NVIDIA GeForce RTX 3060 Laptop (6.44 GB)
- **CUDA**: 12.6
- **Training Speed**: ~37 seconds/epoch (pilot dataset)

## Optimizations

- **Data Loading**: 4 workers with pin_memory for optimal GPU utilization
- **Mixed Precision**: cuDNN benchmark mode enabled
- **Checkpointing**: Auto-saves best model based on validation loss

## License

Dataset provided by Galaxy Zoo under their data release policy.

## References

- Galaxy Zoo 2: [Willett et al. 2013](https://academic.oup.com/mnras/article/435/4/2835/1079261)
- Dataset: [galaxy-datasets](https://github.com/mwalmsley/galaxy-datasets)

## Citation

```bibtex
@article{willett2013galaxy,
  title={Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey},
  author={Willett, Kyle W and others},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={435},
  number={4},
  pages={2835--2860},
  year={2013}
}
```
