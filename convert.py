import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.classificationNet import TinyTrafficSignNet, ClassificationNet

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the trained model
model = ClassificationNet(num_classes=10).to(device)
model.load_state_dict(torch.load('Models/traffic_signs/best_model_SEAME2_epoch_100.pth', map_location=device))
model.eval()

dummy_input = torch.randn(1, 3, 80, 80).to(device)  # Updated for 30x30 traffic sign input

onnx_file_path = "Models/onnx/traffic_sign_model3.onnx"
torch.onnx.export(
    model,                       # PyTorch model instance
    dummy_input,                 # Input to the model
    onnx_file_path,             # Output file path
    export_params=True,         # Store the trained parameter weights inside the model file
    opset_version=12,           # ONNX opset version
    do_constant_folding=True,   # Optimization: fold constant ops into initializers
    input_names=['input'],      # Names for the input tensors
    output_names=['output'],    # Names for the output tensors
    dynamic_axes={
        'input': {0: 'batch_size'},   # Variable batch size
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to {onnx_file_path}")

# Verify the ONNX model
try:
    import onnx
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
except ImportError:
    print("ONNX package not found. Install with: pip install onnx")
    print("Skipping model validation.")
