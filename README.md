# Hybrid CNN-Transformer Model for Multi-Dataset Image Classification

This project implements a hybrid deep learning model combining ConvNeXt and Transformer architectures for image classification across multiple datasets. The model is trained on three different datasets: ImageWoof, FGVC Aircraft, and Flowers102.

## Project Structure

- `cnnHybrid.py`: Defines the hybrid CNN-Transformer architecture combining ConvNeXt and Transformer models
- `utils.py`: Contains utility functions for data loading, seeding, and training/validation loops
- `task_one.py`: Training script for Flowers102 dataset
- `task_two.py`: Training script for ImageWoof dataset
- `task_three.py`: Training script for combined multi-dataset classification
- `Assignment1_Dataloader.ipynb`: Jupyter notebook demonstrating dataset loading and preprocessing

## Model Architecture

The hybrid model (`ConvNeXtWithTransformer`) consists of:
- ConvNeXt base model (pretrained on ImageNet) as the feature extractor
- Projection layer to adjust feature dimensions
- Transformer encoder for sequence modeling
- Final classification layer

Key parameters:
- Embedding dimension: 512
- Number of attention heads: 16
- Number of transformer layers: 1

## Datasets

The project uses three datasets:

1. **Flowers102**
   - 102 flower categories
   - Used for fine-grained flower classification

2. **ImageWoof**
   - 10 dog breed categories
   - A subset of ImageNet focusing on dog breeds

3. **FGVC Aircraft**
   - Aircraft classification dataset
   - Fine-grained visual classification

## Training Process

Common training parameters across all tasks:
- Batch size: 64 (training) / 512 (testing)
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate scheduler: StepLR (step_size=5, gamma=0.1)
- Number of epochs: 10

Data augmentation techniques:
- Random rotation (30 degrees)
- Random horizontal flip
- Random resized crop
- Resize to 224x224

## Usage Instructions

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the datasets:
```python
# Datasets will be downloaded when running the Assignment1_Dataloader.ipynb
# Default paths:
# - ./data/flowers-102
# - ./data/imagewoof2-160
# - ./data/fgvc-aircraft
```

3. Run individual tasks:
```bash
# For Flowers102 dataset
python task_one.py

# For ImageWoof dataset
python task_two.py

# For combined multi-dataset training
python task_three.py
```

## Features

- **Deterministic Training**: Implements seed fixing for reproducible results
- **Efficient Data Loading**: Optimized data loading with pinned memory and worker initialization
- **Multi-GPU Support**: Automatic device selection (CUDA if available, CPU otherwise)
- **Progress Tracking**: Training progress visualization using tqdm
- **Label Remapping**: Automatic label remapping for multi-dataset training

## Model Performance

The model provides:
- Real-time loss and accuracy metrics during training
- Separate validation phase after each epoch
- Training time measurement
- Memory-efficient training with proper CUDA memory management

## Requirements

- Python 3.12.4
- PyTorch 2.4.1
- torchvision 0.19.1
- numpy 1.26.4
- tqdm 4.66.5
