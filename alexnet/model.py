"""
Module for defining the architecture of AlexNet model.
"""

import torch
import torch.nn as nn

from utils import LRN

class AlexNet(nn.Module):
    """
    Reproduced AlexNet model architecture based on AlexNet paper.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential([
            # First convolutional layer

        ])