import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from src.normstd import get_mean_std


# GTSRB - German Traffic Sign Recognition Benchmark
class GTSRBDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, width=30, height=30, is_train=True, val_split=0.0):
        """
        GTSRB Dataset
        Args:
            img_dir: Path to GTSRB root directory (should contain 'Train' and 'Test' folders)
            width: Image width
            height: Image height
            is_train: If True, use Train folder; if False, use Test folder
            val_split: For SEAME compatibility (not used in GTSRB since it has separate test folder)
        """
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        self.val_split = val_split
        
        # Define which GTSRB classes to use (0-42 are available in GTSRB)
        # You can specify which classes you want to keep

        self.selected_classes = [
            2,   # Speed limit (50km/h)
            5,   # Speed limit (80km/h)
            13,  # Yield
            14,  # Stop
            18,  # General caution (danger)
            27,  # Pedestrians (crosswalk)
            28,  # Children crossing (crosswalk variant)
            43,  # Green
            44,  # Red
            45,  # Yellow
            0,   # Speed limit (20km/h) - as unknown
            1,   # Speed limit (30km/h) - as unknown
            3,   # Speed limit (60km/h) - as unknown
            4,   # Speed limit (70km/h) - as unknown
            7,   # Speed limit (100km/h) - as unknown
            8,   # Speed limit (120km/h) - as unknown
            9,   # No passing - as unknown
            10,  # No passing vehicles over 3.5 tons - as unknown
            11,  # Right-of-way at intersection - as unknown
            12,  # Priority road - as unknown
        ]
        
        # Map classes to your specific categories
        self.class_mapping = {
            2: 0,   # Speed limit (50km/h)
            5: 1,   # Speed limit (80km/h)
            13: 2,  # Yield
            14: 3,  # Stop
            18: 4,  # General caution (danger)
            27: 5,  # Pedestrians (crosswalk)
            28: 5,  # Children crossing (also crosswalk) - same as pedestrians
            43: 6,  # Traffic Green
            44: 7,  # Traffic Red
            45: 8,  # Traffic Yellow
            0: 9, 
            1: 9,
            3: 9,
            4: 9,
            7: 9,
            8: 9,
            9: 9,
            10: 9,
            11: 9,
            12: 9,  # Unknown
        }
        
        # Class names for reference
        self.class_names = [
            "Speed 50km/h",      # 0
            "Speed 80km/h",      # 1  
            "Yield",             # 2
            "Stop",              # 3
            "Danger",            # 4
            "Crosswalk",         # 5
            "Traffic Green",     # 6
            "Traffic Red",       # 7
            "Traffic Yellow",     # 8
            "Unknown",           # 9
        ]


        # Load images and labels
        self.images = []
        self.labels = []
        self._load_dataset()
        
        # mean, std = self._calculate_dataset_stats()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
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
        """Load GTSRB dataset images and labels"""
        if self.is_train:
            # Training data: Load from folder structure
            self._load_train_data()
        else:
            # Test data: Load from CSV file
            self._load_test_data()
    
    def _load_train_data(self):
        """Load training data from folder structure"""
        data_path = os.path.join(self.img_dir, "Train")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Train folder not found: {data_path}")
        
        # GTSRB has folders named by class (00000, 00001, etc.)
        for class_folder in os.listdir(data_path):
            class_path = os.path.join(data_path, class_folder)
            if os.path.isdir(class_path):
                try:
                    class_id = int(class_folder)
                    # Only include selected classes
                    if class_id in self.selected_classes:
                        # Map to your specific categories
                        mapped_class = self.class_mapping[class_id]
                        
                        # Get all images in this class folder
                        for img_file in os.listdir(class_path):
                            if img_file.endswith('.ppm') or img_file.endswith('.jpg') or img_file.endswith('.png'):
                                img_path = os.path.join(class_path, img_file)
                                self.images.append(img_path)
                                self.labels.append(mapped_class)
                except ValueError:
                    # Skip non-numeric folder names
                    continue
        
        # Print class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Loaded {len(self.images)} training images from {len(unique_labels)} classes:")
        for label, count in zip(unique_labels, counts):
            print(f"  {self.class_names[label]}: {count} images")
    
    def _load_test_data(self):
        """Load test data from CSV file"""
        test_dir = os.path.join(self.img_dir, "Test")
        csv_file = os.path.join(self.img_dir, "Test.csv")
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test folder not found: {test_dir}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Test CSV file not found: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"Found {len(df)} test images in CSV")
        
        # Process each row in the CSV
        for _, row in df.iterrows():
            # Extract information from CSV
            # CSV format: Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
            class_id = int(row['ClassId'])
            img_path = row['Path']
            
            # Only include selected classes
            if class_id in self.selected_classes:
                # Map to your specific categories
                mapped_class = self.class_mapping[class_id]
                
                # Full path to image
                full_img_path = os.path.join(self.img_dir, img_path)
                
                # Check if image exists
                if os.path.exists(full_img_path):
                    self.images.append(full_img_path)
                    self.labels.append(mapped_class)
                else:
                    print(f"Warning: Image not found: {full_img_path}")
        
        # Print class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Loaded {len(self.images)} test images from {len(unique_labels)} classes:")
        for label, count in zip(unique_labels, counts):
            print(f"  {self.class_names[label]}: {count} images")
            
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