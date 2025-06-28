import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTrafficSignNet(nn.Module):
    """Ultra-lightweight for edge deployment"""
    def __init__(self, num_classes=7, input_channels=3):
        super(TinyTrafficSignNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 30x30 -> 15x15
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 15x15 -> 7x7
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ClassificationNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=3):  # Fixed parameter name
        super(ClassificationNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise separable blocks
        self.dw_conv1 = self._make_dw_block(32, 64, stride=1)
        self.dw_conv2 = self._make_dw_block(64, 128, stride=2)
        self.dw_conv3 = self._make_dw_block(128, 256, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_dw_block(self, in_channels, out_channels, stride=1):
        """Depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class LightweightTrafficSignNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=3):
        super(LightweightTrafficSignNet, self).__init__()
        
        # First block - Basic feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 30x30 -> 15x15
        
        # Second block - More features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 15x15 -> 7x7 (with padding)
        
        # Third block - Higher level features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3
        
        # Global Average Pooling (much lighter than fully connected)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 3x3 -> 1x1
        
        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Classification
        x = self.global_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 128)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage and comparison
if __name__ == "__main__":
    # Test the model
    model = TinyTrafficSignNet(num_classes=7)
    dummy_input = torch.randn(1, 3, 30, 30)
    
    params = count_parameters(model)
    output = model(dummy_input)
    print(f"TinyTrafficSignNet:")
    print(f"  Parameters: {params:,}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model size: {params * 4 / 1024 / 1024:.2f} MB")