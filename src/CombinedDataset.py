import torch
import random
from torch.utils.data import Dataset
from src.SEAMEDataset import SEAMEDataset
from src.GTSRBDataset import GTSRBDataset

class CombinedLaneDataset(Dataset):
    def __init__(self, GTSRB_config=None, SEAME_config=None, carla_config=None, val_split=0.2, seed=42):
        """
        Combined dataset that includes GTSRB, SEAME, Carla datasets
        
        For GTSRB: Uses separate Train/Test folders (val_split ignored)
        For SEAME: Uses val_split to create train/validation from single folder
        
        Args:
            GTSRB_config: Dictionary with GTSRB dataset config or None to skip
            SEAME_config: Dictionary with SEAME dataset config or None to skip
            carla_config: Dictionary with Carla dataset config or None to skip
            val_split: Fraction of data to use for validation (used for SEAME, ignored for GTSRB)
            seed: Random seed for reproducible splits
        """
        self.val_split = val_split
        self.seed = seed
        random.seed(seed)
        
        # Initialize dataset variables
        self.GTSRB_train_dataset = None
        self.GTSRB_test_dataset = None
        self.SEAME_train_dataset = None
        self.SEAME_val_dataset = None
        self.carla_dataset = None
        
        # Create GTSRB datasets if config is provided
        if GTSRB_config:
            self.GTSRB_train_dataset = GTSRBDataset(
                img_dir=GTSRB_config['img_dir'],
                width=GTSRB_config.get('width', 30),
                height=GTSRB_config.get('height', 30),
                is_train=True,  # Use Train folder
                val_split=0.0   # Not used for GTSRB
            )
            
            self.GTSRB_test_dataset = GTSRBDataset(
                img_dir=GTSRB_config['img_dir'],
                width=GTSRB_config.get('width', 30),
                height=GTSRB_config.get('height', 30),
                is_train=False,  # Use Test folder
                val_split=0.0    # Not used for GTSRB
            )
        
        # Create SEAME datasets if config is provided
        if SEAME_config:
            self.SEAME_train_dataset = SEAMEDataset(
                img_dir=SEAME_config['img_dir'],
                width=SEAME_config.get('width', 30),
                height=SEAME_config.get('height', 30),
                is_train=True,   # Training split
                val_split=val_split
            )
            
            self.SEAME_val_dataset = SEAMEDataset(
                img_dir=SEAME_config['img_dir'],
                width=SEAME_config.get('width', 30),
                height=SEAME_config.get('height', 30),
                is_train=False,  # Validation split
                val_split=val_split
            )
        
        # Calculate total sizes
        self._calculate_sizes()
        
        # Default to training mode
        self.is_validation = False
    
    def _calculate_sizes(self):
        """Calculate dataset sizes for each split"""
        # Training sizes
        self.GTSRB_train_size = len(self.GTSRB_train_dataset) if self.GTSRB_train_dataset else 0
        self.SEAME_train_size = len(self.SEAME_train_dataset) if self.SEAME_train_dataset else 0
        self.carla_train_size = len(self.carla_dataset) if self.carla_dataset else 0
        self.train_size = self.GTSRB_train_size + self.SEAME_train_size + self.carla_train_size
        
        # Validation sizes  
        self.GTSRB_val_size = len(self.GTSRB_test_dataset) if self.GTSRB_test_dataset else 0
        self.SEAME_val_size = len(self.SEAME_val_dataset) if self.SEAME_val_dataset else 0
        self.carla_val_size = 0  # Assuming no separate validation for carla
        self.val_size = self.GTSRB_val_size + self.SEAME_val_size + self.carla_val_size
        
        # Print dataset summary
        print(f"Combined dataset created:")
        if self.GTSRB_train_size > 0 or self.GTSRB_val_size > 0:
            print(f"GTSRB: {self.GTSRB_train_size} train, {self.GTSRB_val_size} test")
        if self.SEAME_train_size > 0 or self.SEAME_val_size > 0:
            print(f"SEAME: {self.SEAME_train_size} train, {self.SEAME_val_size} validation")
        if self.carla_train_size > 0:
            print(f"Carla: {self.carla_train_size} train, {self.carla_val_size} validation")
        print(f"Total: {self.train_size} train, {self.val_size} validation")
    
    def set_validation(self, is_validation=True):
        """Set whether to return validation or training samples"""
        self.is_validation = is_validation
        return self
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.is_validation:
            return self.val_size
        else:
            return self.train_size
    
    def __getitem__(self, idx):
        """Get a sample from either training or validation set"""
        if self.is_validation:
            # Getting validation sample
            if idx < self.GTSRB_val_size:
                # Get GTSRB test sample
                return self.GTSRB_test_dataset[idx]
            elif idx < self.GTSRB_val_size + self.SEAME_val_size:
                # Get SEAME validation sample
                seame_idx = idx - self.GTSRB_val_size
                return self.SEAME_val_dataset[seame_idx]
            else:
                # Get Carla validation sample (if implemented)
                carla_idx = idx - self.GTSRB_val_size - self.SEAME_val_size
                return self.carla_dataset[carla_idx]
        else:
            # Getting training sample
            if idx < self.GTSRB_train_size:
                # Get GTSRB training sample
                return self.GTSRB_train_dataset[idx]
            elif idx < self.GTSRB_train_size + self.SEAME_train_size:
                # Get SEAME training sample
                seame_idx = idx - self.GTSRB_train_size
                return self.SEAME_train_dataset[seame_idx]
            else:
                # Get Carla training sample
                carla_idx = idx - self.GTSRB_train_size - self.SEAME_train_size
                return self.carla_dataset[carla_idx]

    def get_train_dataset(self):
        """Return a reference to this dataset in training mode"""
        return self.set_validation(False)
    
    def get_val_dataset(self):
        """Return a reference to this dataset in validation mode"""
        return self.set_validation(True)