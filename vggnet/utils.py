"""
Module that contains utilities for VggNet-11 model.
"""

import torch
import torch.nn as nn
import numpy as np
import config

    
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


# Define a function to compute PCA to avoid import errors
def compute_pca(dataset, indicies):
    """
    Function to compute PCA.
    """

    # Calculate covariance matrix, eigenvalues and eigenvectors to use it on PCA augmentation
    imgs = dataset.data[indicies]       # Only 45k training images
    imgs = imgs.astype(np.float32)
    imgs /= 255                         # Normalize images

    # Reshape them flatten except channels
    flattened = imgs.reshape(-1, 3)     # 3 channels, flatten (multiply) rest of them automatically except channels

    # Calculate covariance matrix, eigenvalues and eigenvectors
    convariance_matrix = np.cov(flattened, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(convariance_matrix)

    return convariance_matrix, eigenvalues, eigenvectors


# PCA Augmentation
class PCA:
    """
    A class for PCA data augmentation to use on training transforms.
    """

    def __init__(self, covariance_matrix, eigenvales, eigenvectors):
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
    

# Functions for saving and loading model checkpoints
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch, 
                    path=config.MODEL_PATH):
    """
    Function for saving model checkpoint.
    """

    # Create the state dict
    state_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    # Save the checkpoint
    torch.save(obj=state_dict, f=path)
    print(f"Checkpoint saved to {path} successfully!")
    return


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, path=config.MODEL_PATH):
    """
    Function for loading model checkpoint.
    """

    # Load the checkpoint from file
    checkpoint = torch.load(f=path, map_location=config.DEVICE)

    # Fill out the parameters using loaded checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Return the epoch
    return checkpoint['epoch']