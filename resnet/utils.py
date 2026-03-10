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