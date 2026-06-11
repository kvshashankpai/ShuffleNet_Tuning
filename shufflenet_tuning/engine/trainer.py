"""
engine/trainer.py
-----------------
Training loop for one ExperimentConfig.

Responsibilities:
  - Build MedMNIST DataLoaders with the correct input_size transform
  - Instantiate the model with the config's hyperparameters
  - Run the training loop with Adam + CosineAnnealingLR
  - Return the trained model (for subsequent evaluation)

The trainer is intentionally stateless — call train() once per config.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.experiment_config import ExperimentConfig
from models.shufflenet import ShuffleNetV2


def build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """
    Builds train and test DataLoaders for PathMNIST.

    The resize transform is driven by cfg.input_size, implementing
    the 4th hyperparameter (input resolution) cleanly at the data level.
    """
    try:
        import medmnist
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError(
            "medmnist is required. Install with: pip install medmnist"
        )

    transform = transforms.Compose([
        transforms.Resize((cfg.input_size, cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = PathMNIST(split="train", transform=transform, download=True)
    test_dataset  = PathMNIST(split="test",  transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,   # CPU-only — pin_memory has no benefit
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    return train_loader, test_loader


def build_model(cfg: ExperimentConfig) -> ShuffleNetV2:
    """Instantiates a ShuffleNetV2 from a config."""
    return ShuffleNetV2(
        width_multiplier=cfg.width_multiplier,
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        intra_op_threads=cfg.intra_op_threads,
    )


def train(cfg: ExperimentConfig) -> tuple[ShuffleNetV2, DataLoader, DataLoader]:
    """
    Trains a ShuffleNetV2 for the given config.

    Args:
        cfg: Fully populated ExperimentConfig for this run.

    Returns:
        Tuple of (trained_model, train_loader, test_loader).
        Loaders are returned so the evaluator can reuse them.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {cfg}")
    print(f"{'='*60}")

    device = torch.device("cpu")
    model  = build_model(cfg).to(device)

    print(f"  Parameters: {model.count_parameters():,}")

    train_loader, test_loader = build_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc  = 100.0 * correct / total
        print(
            f"  Epoch [{epoch:2d}/{cfg.num_epochs}]  "
            f"loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%"
        )

    return model, train_loader, test_loader
