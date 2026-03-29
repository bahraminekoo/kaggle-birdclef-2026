"""Model definitions for BirdCLEF 2026."""

import torch
import torch.nn as nn
import timm


class BirdCLEFModel(nn.Module):
    """
    BirdCLEF classification model using pretrained CNN backbone.
    """
    
    def __init__(self, model_name: str = 'efficientnet_b0', num_classes: int = 234, pretrained: bool = True, dropout: float = 0.3):
        """
        Args:
            model_name: Name of the timm model to use as backbone
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''
        )
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 313)  # Approximate mel-spec shape
            features = self.backbone(dummy_input)
            n_features = features.shape[1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


def create_model(model_name: str, num_classes: int = 234, pretrained: bool = True, dropout: float = 0.3):
    """
    Factory function to create a model.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        
    Returns:
        Model instance
    """
    return BirdCLEFModel(model_name, num_classes, pretrained, dropout)
