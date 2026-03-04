"""
Module for datasets, augmentations and dataloaders.
"""

import torch
import torchvision as tv
import numpy as np
import config
from utils import PCA, WrapperDataset, compute_pca


# Define transforms for train, val (train == val) and test
test_transform = tv.transforms.Compose([
    tv.transforms.Resize(size=(config.RESIZE_SIZE)),
    tv.transforms.CenterCrop(size=(config.INPUT_SIZE)),
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
    generator=torch.Generator(device='cpu').manual_seed(config.SEED) # Always cpu for generator
)

# Split dataset to train and validation datasets (45k train 5k validation on plain train images) using wrapper dataset class implemented in utils
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

# Take covariance matrix, eigenvalues and eigenvectors that needed for PCA
convariance_matrix, eigenvalues, eigenvectors = compute_pca(dataset=base_dataset, indicies=train_subset.indices)

train_transform = tv.transforms.Compose([
    tv.transforms.Resize(size=(config.RESIZE_SIZE)),
    tv.transforms.RandomCrop(size=config.INPUT_SIZE, padding=config.PADDING),
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.ToTensor(),
    PCA(covariance_matrix=convariance_matrix, eigenvales=eigenvalues, eigenvectors=eigenvectors),
    tv.transforms.Normalize(mean=config.MEAN, std=config.STD),
])

train_dataset = WrapperDataset(
    dataset=train_subset,
    transform=train_transform
)

# Load data using dataloaders for each split
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if config.DEVICE == 'cuda' else False,
    drop_last=True      # Drops last batch if is less than intended batch size
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if config.DEVICE == 'cuda' else False,
    drop_last=False
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True if config.DEVICE == 'cuda' else False,
    drop_last=False
)