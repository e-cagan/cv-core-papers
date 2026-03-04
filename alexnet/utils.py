"""
Module that contains utilities for AlexNet.
"""

import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import config

from dataset import COVARIANCE_MATRIX, EIGENVALUES, EIGENVECTORS


# Local Rate Normalization
class LRN(nn.Module):
    """
    Class for local rate normalization to normalize inputs within the AlexNet paper.
    """

    def __init__(self):
        super().__init__()
        
        # Hyperparameters
        self.local_size = config.LOCAL_SIZE
        self.alpha = config.ALPHA
        self.beta = config.BETA
        self.k = config.K

    def forward(self, x):
        """
        Forward propagation function for local rate normalization.
        """

        # Take the squares
        squared = x ** 2

        # padding along channel (e.g: if n=5 then 2 left, 2 right)
        padding = (self.local_size - 1) // 2
        squared_avg = torch.nn.functional.avg_pool3d(
            squared.unsqueeze(1),  # [N, 1, C, H, W]
            kernel_size=(self.local_size, 1, 1),
            stride=1,
            padding=(padding, 0, 0)
        ).squeeze(1)

        # normalization
        s = squared_avg * self.local_size
        denom = (self.k + self.alpha * s).pow(self.beta)
        return x / denom
    

# Wrapper dataset class to avoid loading the dataset twice while splitting it
class WrapperDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset class which takes dataset and transforms (if any) to perform.
    """

    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Take image and label on a particular index
        img, label = self.dataset[index]

        # Check if there is any transform (apply if any)
        if self.transform:
            img = self.transform(img)
        else:
            pass

        return img, label


# PCA Augmentation
class PCA:
    """
    A class for PCA data augmentation to use on training transforms.
    """

    def __init__(self, covariance_matrix=COVARIANCE_MATRIX, eigenvales=EIGENVALUES, eigenvectors=EIGENVECTORS):
        # Convert numpy matrixes to pytorch tensors to avoid conflicts between pytorch and numpy
        self.covariance_matrix = torch.tensor(covariance_matrix, dtype=torch.float32)
        self.eigenvales = torch.tensor(eigenvales, dtype=torch.float32)
        self.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)

    def __call__(self, img: torch.Tensor):
        # Create a random generated tensor for alpha values
        alphas = torch.randn(3) * 0.1                           # mean = 0, standard deviation = 0.1

        # Calculate noise strength
        noises = alphas * torch.sqrt(self.eigenvales)           # Took square root to keep it at the same scale with standard deviation

        # Calculate deflections based on color channels
        deltas = torch.matmul(self.eigenvectors, noises)        # Delta vectors

        # Reshape deltas to match with the shape of image
        deltas = deltas.unsqueeze(1)
        deltas = deltas.unsqueeze(1)

        # Apply deltas to image
        img += deltas
        
        return img.clamp(min=0, max=1)                          # clamp the output between 0 and 1