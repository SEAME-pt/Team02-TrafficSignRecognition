import os
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from src.normstd import get_mean_std

class SEAMEDataset(Dataset):
    def __init__(self, img_dir, width=30, height=30, is_train=True, val_split=0.2):
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        self.val_split = val_split
        
        # Class names for your custom dataset (based on folder structure)
        # Expected folders: "0_Speed_50km_h", "1_Speed_80km_h", "2_Yield", "3_Stop", etc.
        self.class_names = [
            "Speed 50km/h",      # 0
            "Speed 80km/h",      # 1  
            "Yield",             # 2
            "Stop",              # 3
            "Danger",            # 4
            "Crosswalk",         # 5
            "Unknown"            # 6
        ]


        # Load images and labels
        self.all_images = []
        self.all_labels = []
        self._load_dataset()
        
        # For small datasets: use all images for training, subset for validation
        if is_train:
            # Use ALL images for training
            self.images = self.all_images.copy()
            self.labels = self.all_labels.copy()
        else:
            # Use a subset for validation
            self._create_validation_subset()
        
        # Calculate stats using all training data
        mean, std = self._calculate_dataset_stats()
        self.mean = mean.tolist()
        self.std = std.tolist()
        
        print(f"Using mean: {self.mean}")
        print(f"Using std: {self.std}")
        
        # Augmentation for training
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
            
    def _load_dataset(self):
        """Load custom dataset images and labels from folders like '0_Speed_50km_h', '3_Stop'"""
        for folder_name in os.listdir(self.img_dir):
            folder_path = os.path.join(self.img_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    # Parse folder name format: "class_id_description"
                    # Examples: "0_Speed_50km_h", "3_Stop", "5_Crosswalk"
                    if '_' in folder_name:
                        class_id = int(folder_name.split('_')[0])
                        
                        # Validate class_id is within expected range
                        if 0 <= class_id < len(self.class_names):
                            # Get all images in this class folder
                            for img_file in os.listdir(folder_path):
                                if img_file.lower().endswith(('.ppm', '.jpg', '.jpeg', '.png', '.bmp')):
                                    img_path = os.path.join(folder_path, img_file)
                                    self.all_images.append(img_path)
                                    self.all_labels.append(class_id)
                        else:
                            print(f"Warning: Skipping folder '{folder_name}' - class_id {class_id} out of range")
                    else:
                        print(f"Warning: Skipping folder '{folder_name}' - doesn't match expected format 'class_description'")
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping folder '{folder_name}' - cannot parse class_id: {e}")
                    continue
        
        # Print class distribution
        if self.all_images:
            unique_labels, counts = np.unique(self.all_labels, return_counts=True)
            print(f"Loaded {len(self.all_images)} images from {len(unique_labels)} classes:")
            for label, count in zip(unique_labels, counts):
                if label < len(self.class_names):
                    print(f"  Class {label} ({self.class_names[label]}): {count} images")
                else:
                    print(f"  Class {label} (Unknown): {count} images")
        else:
            print("Warning: No images found! Check your dataset directory structure.")
    
    def _create_validation_subset(self):
        """Create validation subset from all images (stratified sampling)"""
        if not self.all_images:
            self.images = []
            self.labels = []
            return
        
        # Convert to numpy arrays for easier manipulation
        all_images = np.array(self.all_images)
        all_labels = np.array(self.all_labels)
        
        # Create validation subset with stratified sampling
        val_images = []
        val_labels = []
        
        # Sample from each class separately to maintain class balance
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            class_mask = all_labels == label
            class_images = all_images[class_mask]
            class_labels = all_labels[class_mask]
            
            # Calculate validation size for this class
            n_samples = len(class_images)
            n_val = max(1, int(n_samples * self.val_split))  # At least 1 sample per class
            
            # Randomly sample validation images
            indices = np.random.RandomState(42).choice(n_samples, size=n_val, replace=False)
            
            val_images.extend(class_images[indices])
            val_labels.extend(class_labels[indices])
        
        self.images = val_images
        self.labels = val_labels
        
        print(f"Using validation subset: {len(self.images)} images ({self.val_split*100:.1f}% of total)")
        
        # Print validation class distribution
        unique_val_labels, val_counts = np.unique(self.labels, return_counts=True)
        print("Validation class distribution:")
        for label, count in zip(unique_val_labels, val_counts):
            if label < len(self.class_names):
                print(f"  {self.class_names[label]}: {count} images")
    
    def _calculate_dataset_stats(self):
        """Calculate mean and std for this specific dataset"""
        # Create a temporary dataset with basic transforms for stats calculation
        temp_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])
        
        temp_dataset = SEAMETempDataset(self.all_images, temp_transform)
        temp_loader = torch.utils.data.DataLoader(
            temp_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=2
        )
        
        return get_mean_std(temp_loader)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, label


class SEAMETempDataset(torch.utils.data.Dataset):
    """Temporary dataset for calculating statistics"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Dummy label for stats calculation
    