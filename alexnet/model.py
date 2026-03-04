"""
Module for defining the architecture of AlexNet model.
"""

import torch
import torch.nn as nn

import config
from utils import LRN


class AlexNet(nn.Module):
    """
    Reproduced AlexNet model architecture based on AlexNet paper.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # Input size 227 ==> 55
            nn.ReLU(),
            LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),                                              # Input size 55 ==> 27

            # Second convolutional layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # Input size 27 ==> 27
            nn.ReLU(),
            LRN(),
            nn.MaxPool2d(kernel_size=3, stride=2),                                              # Input size 27 ==> 13

            # Third convolutional layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # Input size 13 ==> 13
            nn.ReLU(),

            # Fourth convolutional layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # Input size 13 ==> 13
            nn.ReLU(),

            # Fifth convolutional layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # Input size 13 ==> 13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                                              # Input size 13 ==> 6

            # Flatten layer
            nn.Flatten(),                                                                       # Input size is -> 256 channels * 6 * 6 ==> 9216

            # First fully-connected layer
            nn.Linear(in_features=9216, out_features=4096),                                     # Input size 9216 ==> 4096
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT),

            # Second fully connected layer
            nn.Linear(in_features=4096, out_features=4096),                                     # Input size 4096 ==> 4096
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT),

            # Third fully connected layer (output layer)
            nn.Linear(in_features=4096, out_features=10),                                       # Input size 4096 ==> 10
        )
    

    def forward(self, x):
        """
        Forward propagation.
        """

        return self.network(x)