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
    Reproduced DenseNet121 model architecture based on VggNet paper.
    """

    def __init__(self):
        super().__init__()
        
        # Define the network
        


    def forward(self, x):
        """
        Forward propagation.
        """

        return self.network(x)