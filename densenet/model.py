"""
Module for defining the architecture of DenseNet121 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class DenseLayer(nn.Module):
    """
    Dense layer for DenseNet121
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Dense layer architecture
        self.bn_1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):
        """
        Forward propagation.
        """

        # Network
        out = self.bn_1(x)
        out = F.relu(out)
        out = self.conv_1(out)

        # Concatinate the output
        return torch.cat([x, out], dim=1)
    

class DenseBlock(nn.Module):
    """
    Dense block for DenseNet121
    """

    def __init__(self, num_layers, in_channels):
        super().__init__()

        # Block
        self.block = nn.ModuleList()

        # Add dense layers dynamically
        for i in range(num_layers):
            self.block.add_module(f'dense_layer_{i}', module=DenseLayer(in_channels=in_channels + i * config.K, out_channels=config.K))
    

    def forward(self, x):
        """
        Forward propagation.
        """

        for layer in self.block:
            x = layer(x)

        return x


class TransitionLayer(nn.Module):
    """
    Transition layer for DenseNet121
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition_layer = nn.Sequential(
            # Transition layer archtecture
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    
    def forward(self, x):
        """
        Forward propagation
        """

        return self.transition_layer(x)


class DenseNet121(nn.Module):
    """
    Reproduced DenseNet121 model architecture based on Densenet paper.
    """

    def __init__(self):
        super().__init__()
        
        # Define the network
        
        # Input layer
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=1, padding=1)     # out_channels=64 and kernel_size=7 for Imagenet
        self.bn_1 = nn.BatchNorm2d(num_features=24)
        self.relu_1 = nn.ReLU()
        # Additional MaxPool for Imagenet

        # DenseBlock(s) + TransitionLayer(s)
        # Block 1
        self.db_1 = DenseBlock(num_layers=6, in_channels=24)
        self.trns_1 = TransitionLayer(in_channels=96, out_channels=int(96 * config.THETA))      # -> 48

        # Block 2
        self.db_2 = DenseBlock(num_layers=12, in_channels=48)
        self.trns_2 = TransitionLayer(in_channels=192, out_channels=int(192 * config.THETA))    # -> 96

        # Block 3
        self.db_3 = DenseBlock(num_layers=24, in_channels=96)
        self.trns_3 = TransitionLayer(in_channels=384, out_channels=int(384 * config.THETA))    # -> 192

        # Block 4
        self.db_4 = DenseBlock(num_layers=16, in_channels=192)                                  # -> 384

        # Output layer
        self.bn_final = nn.BatchNorm2d(num_features=384)
        self.relu_final = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=384, out_features=10)


    def forward(self, x):
        """
        Forward propagation.
        """

        # Network
        # STEM
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        # DenseLayers + TransitionLayers
        x = self.db_1(x)
        x = self.trns_1(x)

        x = self.db_2(x)
        x = self.trns_2(x)

        x = self.db_3(x)
        x = self.trns_3(x)

        x = self.db_4(x)
        
        # Output layer
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x