"""
Module to test out the model.
"""

import torch
import config

from eval import evaluate
from model import AlexNet
from dataset import test_dataloader


# Test out the model
if __name__ == '__main__':
    
    # Load model checkpoint
    checkpoint = torch.load(f=config.MODEL_PATH, map_location=config.DEVICE)

    # Create model instance and load state dictionary of the model
    model = AlexNet().to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate the model
    avg_test_loss, test_acc = evaluate(model=model, dataloader=test_dataloader, loss=torch.nn.CrossEntropyLoss(), device=config.DEVICE)

    # Print out the results
    print("="*60)
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Test Accuracy: {test_acc}")
    print("="*60)