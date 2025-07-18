import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.train import train_model
from src.classificationNet import TinyTrafficSignNet
import os
import numpy as np

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_size = (30, 30)

    GTSRB_config = {
        'img_dir': '/media/jorge/ea212f76-1f87-43c4-9983-ca15701c0381/StreetSigns',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True
    }

    SEAME_config = {
        'img_dir': '/media/jorge/ea212f76-1f87-43c4-9983-ca15701c0381/Team02-Dataset/SEAME/SEAMEsignals',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True
    }
    
    # # Create the combined dataset with built-in train/val split
    combined_dataset = CombinedLaneDataset(
        GTSRB_config=GTSRB_config, 
        SEAME_config=SEAME_config, 
        carla_config=None,
        val_split=0.2  # Add validation split
    )
    
    # Get train and val datasets
    train_dataset = combined_dataset.get_train_dataset()
    val_dataset = combined_dataset.get_val_dataset()

    # Create weights array for TRAINING data only
    train_GTSRB_size = len(train_dataset.GTSRB_train_dataset) if train_dataset.GTSRB_train_dataset else 0
    train_SEAME_size = len(train_dataset.SEAME_train_dataset) if train_dataset.SEAME_train_dataset else 0
    train_carla_size = len(train_dataset.carla_dataset) if train_dataset.carla_dataset else 0
    
    total_train_size = len(train_dataset)
    weights = np.zeros(total_train_size)

    # Calculate weights for equal contribution (adjust percentages as needed)
    total_samples = train_GTSRB_size + train_SEAME_size + train_carla_size
    GTSRB_weight = 0.5 / (train_GTSRB_size / total_samples) if train_GTSRB_size > 0 else 0
    SEAME_weight = 0.5 / (train_SEAME_size / total_samples) if train_SEAME_size > 0 else 0
    carla_weight = 0.0 / (train_carla_size / total_samples) if train_carla_size > 0 else 0

    # Apply weights to all samples
    current_idx = 0
    if train_GTSRB_size > 0:
        weights[current_idx:current_idx + train_GTSRB_size] = GTSRB_weight
        current_idx += train_GTSRB_size
    if train_SEAME_size > 0:
        weights[current_idx:current_idx + train_SEAME_size] = SEAME_weight
        current_idx += train_SEAME_size
    if train_carla_size > 0:
        weights[current_idx:current_idx + train_carla_size] = carla_weight

    # Create sampler for TRAINING only
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Created weighted sampler: GTSRB={GTSRB_weight:.4f}, SEAME={SEAME_weight:.4f}, Carla={carla_weight:.4f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        sampler=sampler,
        num_workers=os.cpu_count() // 2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,  # Don't shuffle validation
        num_workers=os.cpu_count() // 2
    )
    
    # Initialize model
    model = TinyTrafficSignNet(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
    
    # Train model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50)

if __name__ == '__main__':
    main()