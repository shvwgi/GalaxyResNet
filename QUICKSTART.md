# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 10GB+ disk space for dataset

## Installation

```bash
# Navigate to project
cd GZ2

# Install dependencies
pip install -r requirements.txt
```

## Training

Open and run the Jupyter notebook:

```bash
jupyter notebook GalaxyZoo2Model.ipynb
```

Execute all cells sequentially. The notebook will:
1. Download Galaxy Zoo 2 dataset (~2.5GB)
2. Preprocess images
3. Train GalaxyCNN model
4. Save checkpoints and visualizations

**Training time**: ~6 minutes on RTX 3060

## Inference

Use the trained model to classify new galaxies:

```python
python inference.py
```

Or programmatically:

```python
from inference import load_model, preprocess_image, predict

# Load model
model = load_model('./models/best_model.pth')

# Classify galaxy
image = preprocess_image('path/to/galaxy.jpg')
probs = predict(model, image)

# Results: [smooth_prob, featured_prob, artifact_prob]
print(probs)
```

## Outputs

After training, you'll find:

- `models/best_model.pth` - Best performing model
- `models/final_model.pth` - Final model with training history
- `plots/` - Training visualizations

## Model Performance

| Metric | Value |
|--------|-------|
| Validation Loss | 0.1264 |
| Smooth MAE | 0.1463 |
| Featured/Disk MAE | 0.1467 |
| Artifact MAE | 0.0269 |

## Hardware Requirements

**Minimum**:
- CPU: Modern x86_64
- RAM: 8GB
- GPU: Optional (CPU training supported but slow)

**Recommended**:
- GPU: NVIDIA RTX 3060 or better
- VRAM: 6GB+
- CUDA: 12.0+

## Troubleshooting

**Out of memory**: Reduce batch size in training cell
**Slow training**: Ensure CUDA is properly installed
**Import errors**: Run `pip install -r requirements.txt`

## Citation

```bibtex
@article{willett2013galaxy,
  title={Galaxy Zoo 2: detailed morphological classifications},
  author={Willett, Kyle W and others},
  journal={MNRAS},
  year={2013}
}
```
