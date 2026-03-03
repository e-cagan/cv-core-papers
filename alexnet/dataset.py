"""
Module for datasets, augmentations and dataloaders.
"""

import torch
import torchvision as tv
import config


# Define transforms for train, val (train == val) and test
train_transform = tv.transforms.Compose([
    tv.transforms.RandomCrop(size=config.INPUT_SIZE, padding=config.PADDING),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=config.MEAN, std=config.STD),
])

test_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=config.MEAN, std=config.STD),
])


# Load the datasets (split the train dataset to train and validaton datasets)
train_dataset = tv.datasets.CIFAR10(
    root=config.DATASET_PATH,
    train=True,
    transform=train_transform,
    download=True
)

val_dataset = None

test_dataset = tv.datasets.CIFAR10(
    root=config.DATASET_PATH,
    train=False,
    transform=test_transform,
    download=True
)