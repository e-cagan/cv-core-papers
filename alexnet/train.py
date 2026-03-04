"""
Module for training the model.
"""

import torch
import torch.nn as nn
import torchvision as tv
import numpy as np

import config
from dataset import train_dataloader, val_dataloader, test_dataloader
from eval import evaluate
from model import AlexNet


# Define important variables to use in training
EPOCHS = config.EPOCHS
model = AlexNet().to(device=config.DEVICE)
loss = nn.CrossEntropyLoss()

# SGD + momentum optimizer
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY
)


# Define a function for training the model
def train(model=model, dataloader=train_dataloader, loss=loss):
    """
    Train function for model which works with model, dataloader, loss function and device configuration.
    """

    


# Test the training function
if __name__ == '__main__':
    # TODO start implementing train after implementing evaluation.
    pass