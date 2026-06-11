"""
engine/evaluator.py
--------------------
Evaluates a trained model on the test DataLoader.

Kept separate from the trainer so you can:
  - Re-evaluate a saved checkpoint without retraining
  - Run evaluation at any epoch checkpoint interval
  - Swap in different metrics without touching training logic
"""

import torch
from torch.utils.data import DataLoader

from models.shufflenet import ShuffleNetV2


@torch.no_grad()
def evaluate(model: ShuffleNetV2, test_loader: DataLoader) -> float:
    """
    Computes top-1 accuracy on the test split.

    Args:
        model:       Trained ShuffleNetV2 (already on the correct device).
        test_loader: DataLoader for the test split.

    Returns:
        Accuracy as a percentage (0–100), e.g. 87.3
    """
    model.eval()
    device  = next(model.parameters()).device
    correct = 0
    total   = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        outputs    = model(images)
        _, predicted = outputs.max(1)

        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)

    accuracy = 100.0 * correct / total
    print(f"  Test Accuracy: {accuracy:.2f}%")
    return accuracy
