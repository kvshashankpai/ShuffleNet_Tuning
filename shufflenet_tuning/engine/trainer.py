"""
engine/trainer.py
-----------------
CPU-only training loop for one ExperimentConfig.

Responsibilities:
  - Build MedMNIST DataLoaders with the correct input_size transform
  - Instantiate the model with the config's hyperparameters
  - Build optimizer (Adam / SGD / RMSprop) and LR scheduler
    (CosineAnnealingLR / StepLR / OneCycleLR) from config
  - Run the training loop with CrossEntropyLoss (+ optional label smoothing)
  - Return the trained model (for subsequent evaluation)

NOTE: All training is CPU-only. GPU/CUDA paths have been removed to ensure
      consistent, reproducible CPU energy and latency profiling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.experiment_config import ExperimentConfig
from models.shufflenet import QuantizableShuffleNetV2

# Always CPU — no CUDA paths
DEVICE = torch.device("cpu")


def build_dataloaders(cfg: ExperimentConfig, device: torch.device = None) -> tuple[DataLoader, DataLoader]:
    """
    Builds train and test DataLoaders for PathMNIST.

    The resize transform is driven by cfg.input_size, implementing
    the spatial resolution hyperparameter cleanly at the data level.

    Note: `device` argument is accepted for API compatibility but ignored —
    all data stays on CPU.
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

    # Conservative data loading for shared/remote CPU hosts — avoids
    # pin-memory thread failures and file-descriptor pressure during
    # parallel Optuna trials.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, test_loader


def build_model(cfg: ExperimentConfig) -> QuantizableShuffleNetV2:
    """Instantiates a QuantizableShuffleNetV2 from a config."""
    model = QuantizableShuffleNetV2(
        width_multiplier=cfg.width_multiplier,
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        intra_op_threads=cfg.intra_op_threads,
    )
    model.dropout.p = cfg.dropout
    return model


def build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> optim.Optimizer:
    """
    Builds the optimizer from cfg.optimizer_name.

    Supported:
      - "adam"    → Adam (momentum-free, ignores cfg.momentum)
      - "sgd"     → SGD with Nesterov momentum
      - "rmsprop" → RMSprop with momentum
    """
    name = cfg.optimizer_name.lower()
    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    elif name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer_name}'. "
            f"Choose from: 'adam', 'sgd', 'rmsprop'."
        )


def build_scheduler(
    cfg: ExperimentConfig,
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """
    Builds the LR scheduler from cfg.scheduler_name.

    Supported:
      - "cosine"   → CosineAnnealingLR over full training
      - "step"     → StepLR — decays by 0.5 every 3 epochs
      - "onecycle" → OneCycleLR — fast convergence, good for short runs
    """
    name = cfg.scheduler_name.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs
        )
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.num_epochs // 3), gamma=0.5
        )
    elif name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.num_epochs,
            pct_start=0.3,
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{cfg.scheduler_name}'. "
            f"Choose from: 'cosine', 'step', 'onecycle'."
        )


def train(cfg: ExperimentConfig, device: torch.device = None) -> tuple[QuantizableShuffleNetV2, DataLoader, DataLoader]:
    """
    Trains a QuantizableShuffleNetV2 on CPU for the given config.

    Args:
        cfg:    Fully populated ExperimentConfig for this run.
        device: Accepted for API compatibility but ignored — always CPU.

    Returns:
        Tuple of (trained_model, train_loader, test_loader).
        Loaders are returned so the evaluator can reuse them.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {cfg}")
    print(f"  Device: CPU (forced)")
    print(f"{'='*60}")

    # CPU-only: set MKL-DNN optimisations and thread count
    torch.set_num_threads(cfg.intra_op_threads)
    torch.backends.mkldnn.enabled = True

    model = build_model(cfg).to(DEVICE)
    print(f"  Parameters: {model.count_parameters():,}")

    train_loader, test_loader = build_dataloaders(cfg, device=DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = build_optimizer(cfg, model)

    # OneCycleLR is per-step; others are per-epoch
    is_per_step = cfg.scheduler_name.lower() == "onecycle"
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if is_per_step:
                scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += images.size(0)

        if not is_per_step:
            scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc  = 100.0 * correct / total
        print(
            f"  Epoch [{epoch:2d}/{cfg.num_epochs}]  "
            f"loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%"
        )

    return model, train_loader, test_loader
