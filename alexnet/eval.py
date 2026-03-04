"""
Module for providing the evaluation function for training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


# Define the evaluate function
def evaluate(model: nn.Module, dataloader: DataLoader, loss: nn.CrossEntropyLoss, device=config.DEVICE):
    """
    Evaluate function for model which works with model, dataloader, loss function and device configuration.
    """
    
    # Accuracy calculation variables
    running_val_loss = 0.0
    true, total = 0, 0

    # Setup model mode as evaluation
    model.eval()

    # Start evaluatin on no_grad mode since we aren't training
    with torch.no_grad():

        # Iterate trough batches
        for i, data in enumerate(dataloader):
            # Take images and labels
            imgs, labels = data

            # Carry them to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Take the outputs and predictions
            outputs = model(imgs)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate the validation loss then add it to running validation loss
            val_loss = loss(outputs, labels).item()
            running_val_loss += val_loss

            # Take total and true
            total += labels.size(0)
            true += (predictions == labels).sum().item()
    
    # Calculate average validation loss
    avg_val_loss = running_val_loss / len(dataloader) # BATCH SIZE

    # Calculate validation accuracy
    val_acc = true / total

    return avg_val_loss, val_acc