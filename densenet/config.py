"""
Module for configuring hyperparameters.
"""

import os
import torch


# Additional constants
DATASET_PATH = "cifar10_dataset/"
MODEL_PATH = "models/densenet121_model.pt"
SPLIT_LENGTHS = [45000, 5000]       # 45k train - 5k validation images 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
NUM_WORKERS = 4

# Training hyperparameters
LEARNING_RATE = 0.1
LR_FACTOR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
DROPOUT = 0.5
EPOCHS = 300
K = 12              # Growth rate
THETA = 0.5         # Compression

# Transform hyperparameters
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
RESIZE_SIZE = 32
INPUT_SIZE = 32
PADDING = 4