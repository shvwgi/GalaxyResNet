# Changelog

All notable changes to the Galaxy Zoo 2 Classification project.

## [1.0.0] - 2026-03-09

### Added
- Initial release of GalaxyCNN model
- Custom 4-block convolutional architecture for galaxy classification
- CUDA optimization with pin_memory and cuDNN benchmark
- Automatic model checkpointing based on validation loss
- Training visualizations (learning curves, predictions, scatter plots)
- Vote distribution analysis
- Per-class MAE metrics
- Comprehensive documentation and README
- Inference script for model deployment
- Requirements.txt for dependency management
- MIT License

### Model Performance
- Best validation loss: 0.1264
- Smooth class MAE: 0.1463
- Featured/Disk class MAE: 0.1467
- Artifact class MAE: 0.0269
- Training time: ~6 minutes on RTX 3060

### Data
- Galaxy Zoo 2 dataset with 167,434 galaxies
- Automatic download and preprocessing
- 80/20 train/validation split (pilot dataset: 2,000 samples)

### Hardware
- Tested on NVIDIA GeForce RTX 3060 Laptop GPU
- CUDA 12.6 support
- CPU fallback available
