"""
engine/trainer.py
-----------------
CPU-only training loop for one ExperimentConfig.

Responsibilities:
  - Build MedMNIST DataLoaders with the correct input_size transform
  - Instantiate the model with the config's hyperparameters
  - Build optimizer (Adam / SGD) and LR scheduler
    (CosineAnnealingLR / StepLR / OneCycleLR) from config
  - Build criterion (CrossEntropy / KLDivergence / FocalLoss) from config
  - Run the training loop
  - Return the trained model (for subsequent evaluation)

NOTE: All training is CPU-only. GPU/CUDA paths have been removed to ensure
      consistent, reproducible CPU energy and latency profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.experiment_config import ExperimentConfig
from models.shufflenet import QuantizableShuffleNetV2

# Always CPU — no CUDA paths
DEVICE = torch.device("cpu")


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification (Lin et al., 2017).

    Down-weights easy examples and focuses training on hard-to-classify samples.
    Particularly useful for imbalanced medical imaging datasets.

    Args:
        gamma:           Focusing parameter. Higher gamma = more focus on hard examples.
        label_smoothing: Label smoothing epsilon applied before focal weighting.
        reduction:       'mean' or 'sum'.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ── Builder functions ─────────────────────────────────────────────────────────

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
        stage_repeats=cfg.resolved_stage_repeats,
        fc_hidden_dim=cfg.fc_hidden_dim,
    )
    model.dropout.p = cfg.dropout
    return model


def build_criterion(cfg: ExperimentConfig) -> nn.Module:
    """
    Builds the loss function from cfg.loss_name.

    Supported:
      - "cross_entropy"  → CrossEntropyLoss with label smoothing
      - "kl_divergence"  → KLDivLoss (with log-softmax on model output, smoothed targets)
      - "focal"          → FocalLoss (gamma=2.0, focuses on hard examples)
    """
    name = cfg.loss_name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    elif name == "kl_divergence":
        # KLDivLoss will be used with a wrapper in the training loop
        return nn.KLDivLoss(reduction="batchmean")
    elif name == "focal":
        return FocalLoss(gamma=2.0, label_smoothing=cfg.label_smoothing)
    else:
        raise ValueError(
            f"Unknown loss '{cfg.loss_name}'. "
            f"Choose from: 'cross_entropy', 'kl_divergence', 'focal'."
        )


def build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> optim.Optimizer:
    """
    Builds the optimizer from cfg.optimizer_name.

    Supported:
      - "adam"    → Adam (momentum-free, ignores cfg.momentum)
      - "sgd"     → SGD with Nesterov momentum
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
    else:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer_name}'. "
            f"Choose from: 'adam', 'sgd'."
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


def _compute_kl_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    """
    Computes KL Divergence loss with smoothed one-hot targets.

    KLDivLoss expects log-probabilities as input and probability targets.
    We apply label smoothing to the one-hot target distribution.
    """
    log_probs = F.log_softmax(outputs, dim=1)

    # Build smoothed one-hot targets
    with torch.no_grad():
        targets = torch.zeros_like(log_probs)
        targets.fill_(label_smoothing / (num_classes - 1))
        targets.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)

    return criterion(log_probs, targets)


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

    criterion = build_criterion(cfg)
    is_kl = cfg.loss_name.lower() == "kl_divergence"

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

            if is_kl:
                loss = _compute_kl_loss(criterion, outputs, labels, cfg.num_classes, cfg.label_smoothing)
            else:
                loss = criterion(outputs, labels)

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
