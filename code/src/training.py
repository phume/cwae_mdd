"""Training utilities for CWAE-MMD and baselines."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import time


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train for one epoch.

    Returns:
        Dictionary of average metrics
    """
    model.train()
    total_metrics = {}
    n_batches = 0

    for x, c, _ in dataloader:
        x = x.to(device)
        c = c.to(device)

        optimizer.zero_grad()
        loss, metrics = model.compute_loss(x, c)

        # Check for NaN
        if torch.isnan(loss):
            return {"loss": float("nan"), "nan_detected": True}

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        for k, v in metrics.items():
            if k not in total_metrics:
                total_metrics[k] = 0.0
            total_metrics[k] += v
        n_batches += 1

    # Average
    for k in total_metrics:
        total_metrics[k] /= n_batches

    return total_metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Validate model.

    Returns:
        Dictionary of average metrics
    """
    model.eval()
    total_metrics = {}
    n_batches = 0

    with torch.no_grad():
        for x, c, _ in dataloader:
            x = x.to(device)
            c = c.to(device)

            _, metrics = model.compute_loss(x, c)

            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v
            n_batches += 1

    for k in total_metrics:
        total_metrics[k] /= n_batches

    return total_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model with early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Maximum epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: Device to use
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for epoch in iterator:
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Check for NaN
        if np.isnan(train_metrics.get("loss", 0)):
            if verbose:
                print(f"\nNaN detected at epoch {epoch + 1}, stopping training")
            break

        history["train_loss"].append(train_metrics["loss"])

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device)
            history["val_loss"].append(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if verbose:
            msg = f"Epoch {epoch + 1}: train_loss={train_metrics['loss']:.4f}"
            if val_loader is not None:
                msg += f", val_loss={val_metrics['loss']:.4f}"
            iterator.set_postfix_str(msg)

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
