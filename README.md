# Traffic Sign Recognition for Autonomous Driving

A comprehensive implementation of traffic sign recognition using lightweight CNN architectures with PyTorch for autonomous driving applications.

## Overview

This repository contains an efficient implementation for traffic sign classification focused on real-time recognition in autonomous driving scenarios. The system processes images to identify and classify key traffic signs including speed limits, stop signs, yield signs, crosswalk signs, and warning signs through image classification.

## Features

- **TinyTrafficSignNet Architecture**: Ultra-lightweight CNN optimized for edge deployment
- **Multi-Dataset Training**: Support for GTSRB and custom SEAME datasets
- **Weighted Sampling**: Balances data from different sources for better generalization
- **Advanced Data Augmentation**: Enhances robustness through augmentation techniques
- **Dataset-Specific Normalization**: Calculates mean/std for optimal performance
- **ONNX Export**: Easy model conversion for deployment on edge devices
- **Real-time Inference**: Optimized detection and classification pipeline

## Model Architecture

The model architecture consists of:

- **TinyTrafficSignNet**: Lightweight CNN with only ~15K parameters
- **Efficient Design**: 3 convolutional blocks with batch normalization
- **Global Average Pooling**: Reduces parameters while maintaining performance
- **Dropout Regularization**: Prevents overfitting on small datasets
- **Memory-Efficient Design**: Optimized for deployment on autonomous vehicles

## Directory Structure

```
.
├── assets/                 # Video files for testing
├── Models/                 # Saved model weights
│   ├── traffic_signs/      # Traffic sign classification models
│   └── onnx/              # ONNX format models
├── src/                   # Source code
│   ├── augmentation.py    # Data augmentation implementations
│   ├── classificationNet.py # CNN model implementations
│   ├── CombinedDataset.py # Combined dataset for multi-dataset training
│   ├── GTSRBDataset.py    # GTSRB dataset loader with CSV test support
│   ├── normstd.py         # Dataset statistics calculation
│   ├── SEAMEDataset.py    # SEAME custom dataset loader
│   └── train.py           # Training functions with validation
├── convert.py             # Script for converting PyTorch model to ONNX
├── inference.py           # Script for real-time traffic sign detection
├── main.py                # Main training script
├── example_combined_usage.py # Example usage of combined datasets
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SEAME-pt/Team02-TrafficSignRecognition.git
cd Team02-TrafficSignRecognition
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

To train the traffic sign recognition model:

```bash
python main.py
```

The training script will:
- Load GTSRB and SEAME datasets
- Calculate dataset-specific normalization statistics
- Train with validation monitoring
- Save best model based on validation accuracy

### Inference

To run traffic sign detection and classification on a video file:

```bash
python inference.py
```

Controls during inference:
- Press 'q' to quit
- Press 'd' to toggle debug mode (shows detection areas)

### Converting to ONNX

To convert a trained PyTorch model to ONNX format for deployment:

```bash
python convert.py
```

### Converting to .engine
To convert an ONNX model to TensorRT engine format optimized for the target GPU, run:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=onnx/traffic_sign_model.onnx --saveEngine=engine/traffic_sign_detection_model.engine --fp16
```


## Datasets

The system supports multiple datasets:

- **GTSRB**: German Traffic Sign Recognition Benchmark with 43 classes
  - **Training**: Organized in class folders (00000/, 00001/, etc.)
  - **Testing**: All images in Test/ folder with CSV labels
- **SEAME**: Custom dataset for Southeast Asian traffic signs
  - **Format**: Folders named like "0_Speed_50km_h", "3_Stop"
  - **Validation**: Uses subset sampling for small datasets

## Traffic Sign Classes

The model recognizes 7 key traffic sign categories:

1. **Speed 50km/h** - Speed limit signs
2. **Speed 80km/h** - Speed limit signs
3. **Yield** - Yield/Give way signs
4. **Stop** - Stop signs
5. **Danger** - Warning/Caution signs
6. **Crosswalk** - Pedestrian crossing signs
7. **Unknown** - Other traffic signs (for rejection)

## Dataset Structure

### GTSRB Structure
```
StreetSigns/
├── Train/
│   ├── 00000/    # Class folders with .ppm images
│   ├── 00001/
│   └── ...
├── Test/
│   ├── 00000.ppm  # All test images in one folder
│   ├── 00001.ppm
│   └── ...
└── Test.csv      # CSV with columns: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
```

### SEAME Structure
```
SEAMEsignals/
├── 0_Speed_50km_h/    # Class folders with descriptive names
├── 1_Speed_80km_h/
├── 2_Yield/
├── 3_Stop/
├── 4_Danger/
├── 5_Crosswalk/
└── 6_Unknown/
```

## Key Features

### Smart Detection Pipeline
- **Color-based detection**: Identifies potential sign regions using HSV color filtering
- **Geometric filtering**: Filters by area, aspect ratio, and shape complexity
- **Confidence thresholding**: Only displays high-confidence predictions (>75%)
- **Unknown rejection**: Automatically filters out unknown/irrelevant detections

### Training Optimizations
- **Weighted sampling**: Balances GTSRB and SEAME contributions
- **Validation monitoring**: Tracks both accuracy and loss
- **Best model saving**: Automatically saves best performing model
- **Dataset-specific normalization**: Calculates optimal mean/std for each dataset

### Deployment Ready
- **Lightweight model**: Only ~15K parameters for fast inference
- **ONNX conversion**: Ready for edge device deployment
- **Real-time processing**: Optimized for video stream processing
- **Adjustable thresholds**: Easy to tune for different scenarios

## Model Performance

- **Input Size**: 30x30 pixels (optimized for speed)
- **Parameters**: ~15,000 (ultra-lightweight)
- **Model Size**: ~60KB (perfect for edge devices)
- **Inference Speed**: Real-time on CPU
- **Classes**: 7 traffic sign categories

## Configuration

Key parameters can be adjusted in the code:

```python
# Detection sensitivity
min_area = 800          # Minimum detection area
confidence_threshold = 0.75  # Minimum confidence for display

# Training parameters
batch_size = 16         # Training batch size
learning_rate = 1.5e-4  # Adam optimizer learning rate
epochs = 50             # Training epochs

# Dataset balance
GTSRB_weight = 0.5      # GTSRB dataset contribution
SEAME_weight = 0.5      # SEAME dataset contribution
```

## Optimization

For deployment on edge devices, the model can be:
- Quantized to FP16 precision for faster inference
- Converted to TensorRT for NVIDIA hardware acceleration
- Optimized with ONNX Runtime for cross-platform deployment