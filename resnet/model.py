"""
Module for defining the architecture of ResNet18 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BasicBlock(nn.Module):
    """
    Basic block for ResNet18
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # First conv + batch norm layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)

        # Second conv + batch norm layer
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=out_channels)

        # Check if there is a downsample to protect shape
        self.downsample_conv = None
        self.downsample_bn = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(num_features=out_channels)

    
    def forward(self, x):
        """
        Forward propagation
        """

        # Store the original input for skip connection
        identity = x

        # Main network
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.batch_norm2(out)

        # If there is a shape mismatch downsample the identity
        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        # Skip Connection: Add original output with block output
        out += identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    """
    Reproduced ResNet18 model architecture based on Resnet paper.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Starter layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),    # stride = 1 for CIFAR-10 dataset, normally 2

            # First layer
            BasicBlock(in_channels=64, out_channels=64, stride=1),
            BasicBlock(in_channels=64, out_channels=64, stride=1),

            # Second layer
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            BasicBlock(in_channels=128, out_channels=128, stride=1),

            # Third layer
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            BasicBlock(in_channels=256, out_channels=256, stride=1),

            # Fourth layer
            BasicBlock(in_channels=256, out_channels=512, stride=2),
            BasicBlock(in_channels=512, out_channels=512, stride=1),

            # Average adaptive pooling layer
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            # Flatten layer
            nn.Flatten(),

            # Output layer
            nn.Linear(in_features=512, out_features=10),
        )


    def forward(self, x):
        """
        Forward propagation.
        """

        return self.network(x)