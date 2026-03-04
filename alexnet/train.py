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
device = config.DEVICE
model = AlexNet().to(device=device)
loss = nn.CrossEntropyLoss()

# SGD + momentum optimizer
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=config.LEARNING_RATE,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',         # 'min' for loss, 'max' for accuracy
    factor=config.LR_FACTOR,
    patience=5,
    min_lr=1e-6,
    threshold=0.001,
    cooldown=0
)


# Define a function for training the model
def train(model=model, dataloader=train_dataloader, loss=loss):
    """
    Train function for model which works with model, dataloader, loss function and device configuration.
    """

    # Setup the model mode as training mode
    model.train(mode=True)

    # Iterate trough batches in range of epochs
    for epoch in range(EPOCHS):
        
        # Running loss and variables to calculate accuracy
        running_train_loss = 0.0
        true, total = 0, 0
        
        for i, data in enumerate(dataloader):
            # Take images and labels
            imgs, labels = data

            # Convert them to device
            imgs, labels = imgs.to(device), labels.to(device)

            # Zero the gradients for every batch
            optimizer.zero_grad()

            # Forward propagation to predict
            outputs = model(imgs)
            predictions = torch.argmax(outputs, dim=1)

            # Calculate the loss and backward propagation
            train_loss = loss(outputs, labels)
            train_loss.backward()
            running_train_loss += train_loss.item()

            # Adjust learning weights
            optimizer.step()

            # Take total and true to calculate train accuracy afterwards
            total += labels.size(0)
            true += (predictions == labels).sum().item()

        # Take average validation loss and validation accuracy from evaluate function
        average_val_loss, val_acc = evaluate(model=model, dataloader=val_dataloader, loss=loss, device=device)

        # Calculate average training loss and training accuracy
        average_train_loss = running_train_loss / len(dataloader) # BATCH SIZE
        train_acc = (true / total) * 100

        # Return model back to train mode since evaluate function adjusts the model mode as eval
        model.train(mode=True)

        # Log the info to display metrics
        print("="*60)
        print(f"Epoch: {epoch + 1}")
        print(f"Average train loss: {average_train_loss}")
        print(f"Train accuracy: {train_acc}")
        print(f"Average validation loss: {average_val_loss}")
        print(f"Validation accuracy: {val_acc}")
        print("="*60)

        # Schedule the learning rate
        lr_scheduler.step(average_val_loss)
        
    return


# Test the training function
if __name__ == '__main__':
    train(
        model=model,
        dataloader=train_dataloader,
        loss=loss
    )