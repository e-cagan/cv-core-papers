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


if __name__ == '__main__':
    # TODO start implementing train after implementing evaluation.
    pass