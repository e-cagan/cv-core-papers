"""
Module for datasets, augmentations and dataloaders.
"""

import torch
import torchvision as tv
import numpy as np
import config
from utils import PCA, WrapperDataset


# Define transforms for train, val (train == val) and test
train_transform = tv.transforms.Compose([
    tv.transforms.RandomCrop(size=config.INPUT_SIZE, padding=config.PADDING),
    tv.transforms.ToTensor(),
    PCA(),
    tv.transforms.Normalize(mean=config.MEAN, std=config.STD),
])

test_transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=config.MEAN, std=config.STD),
])

# Load the datasets (split the train dataset to train and validaton datasets)
base_dataset = tv.datasets.CIFAR10(
    root=config.DATASET_PATH,
    train=True,
    transform=None,
    download=True
)

# Generate random subsets for split
train_subset, val_subset = torch.utils.data.random_split(
    base_dataset,
    lengths=config.SPLIT_LENGTHS,
    generator=torch.Generator(device=config.DEVICE).manual_seed(config.SEED)
)

# Split dataset to train and validation datasets (45k train 5k validation on plain train images) using wrapper dataset class implemented in utils
train_dataset = WrapperDataset(
    dataset=train_subset,
    transform=train_transform
)

val_dataset = WrapperDataset(
    dataset=val_subset,
    transform=test_transform
)

# Load test dataset
test_dataset = tv.datasets.CIFAR10(
    root=config.DATASET_PATH,
    train=False,
    transform=test_transform,
    download=True
)

# Calculate covariance matrix, eigenvalues and eigenvectors to use it on PCA augmentation
train_imgs = base_dataset.data[train_subset.indices]    # Only 45k training images
train_imgs = train_imgs.astype(np.float32)
train_imgs /= 255                                       # Normalize images

# Reshape them flatten except channels
flattened = train_imgs.reshape(-1, 3)                   # 3 channels, flatten (multiply) rest of them automatically except channels

# Calculate covariance matrix, eigenvalues and eigenvectors
COVARIANCE_MATRIX = np.cov(flattened, rowvar=False)
EIGENVALUES, EIGENVECTORS = np.linalg.eigh(COVARIANCE_MATRIX)