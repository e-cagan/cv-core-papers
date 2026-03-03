"""
Module for configuring hyperparameters.
"""

import os
import torch


# Additional constants
DATASET_PATH = "cifar10_dataset/"
SPLIT_LENGTHS = [45000, 5000]       # 45k train - 5k validation images 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# Training hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 128
DROPOUT = 0.5

# Transform hyperparameters
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
INPUT_SIZE = 32
PADDING = 4

# LRN hyperparameters
LOCAL_SIZE = 5
ALPHA = 0.0001
BETA = 0.75
K = 2