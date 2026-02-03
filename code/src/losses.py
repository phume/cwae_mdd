"""Alternative loss functions for CWAE-MMD."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def crps_gaussian(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute CRPS assuming Gaussian predictive distribution.

    CRPS (Continuous Ranked Probability Score) measures the integral of squared
    differences between the predicted CDF and the empirical CDF. For a Gaussian
    predictive distribution N(mu, sigma^2), the closed-form CRPS is:

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    where z = (y - mu) / sigma, Phi is standard normal CDF, phi is standard normal PDF.

    Args:
        y_true: True values [B, D]
        y_pred: Predicted mean (reconstruction) [B, D]
        sigma: Predictive std. If None, estimated from residuals.

    Returns:
        Mean CRPS loss (scalar)
    """
    # Standard normal CDF and PDF
    def phi(x):
        """Standard normal PDF."""
        return torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def Phi(x):
        """Standard normal CDF."""
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))

    # Estimate sigma from residuals if not provided
    if sigma is None:
        # Use batch std as estimate, with minimum for stability
        residuals = y_true - y_pred
        sigma = residuals.std(dim=0, keepdim=True).clamp(min=1e-6)

    # Standardized residual
    z = (y_true - y_pred) / sigma

    # CRPS formula for Gaussian
    crps = sigma * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / np.sqrt(np.pi))

    return crps.mean()


def crps_empirical(
    y_true: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    """Compute empirical CRPS from samples.

    CRPS = E[|Y - X|] - 0.5 * E[|X - X'|]

    where Y is true value, X and X' are independent samples from predictive dist.

    Args:
        y_true: True values [B, D]
        samples: Samples from predictive distribution [B, N_samples, D]

    Returns:
        Mean CRPS loss (scalar)
    """
    B, N, D = samples.shape

    # Term 1: E[|Y - X|]
    # y_true: [B, D] -> [B, 1, D]
    y_expanded = y_true.unsqueeze(1)
    term1 = torch.abs(y_expanded - samples).mean(dim=1)  # [B, D]

    # Term 2: E[|X - X'|] (using all pairs)
    # This is O(N^2), for efficiency use random pairs if N is large
    if N > 100:
        # Random pairs
        idx1 = torch.randint(0, N, (N,), device=samples.device)
        idx2 = torch.randint(0, N, (N,), device=samples.device)
        term2 = torch.abs(samples[:, idx1] - samples[:, idx2]).mean(dim=1)
    else:
        # All pairs
        samples_i = samples.unsqueeze(2)  # [B, N, 1, D]
        samples_j = samples.unsqueeze(1)  # [B, 1, N, D]
        term2 = torch.abs(samples_i - samples_j).mean(dim=(1, 2))  # [B, D]

    crps = term1 - 0.5 * term2
    return crps.mean()


def huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss - smooth combination of L1 and L2.

    Less sensitive to outliers than MSE.

    Args:
        y_true: True values
        y_pred: Predictions
        delta: Threshold for switching from L2 to L1

    Returns:
        Mean Huber loss
    """
    residual = torch.abs(y_true - y_pred)
    quadratic = torch.clamp(residual, max=delta)
    linear = residual - quadratic
    return (0.5 * quadratic**2 + delta * linear).mean()


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """Quantile (pinball) loss.

    Args:
        y_true: True values
        y_pred: Predictions
        quantile: Target quantile (0.5 = median = L1 loss)

    Returns:
        Mean quantile loss
    """
    residual = y_true - y_pred
    return torch.max(quantile * residual, (quantile - 1) * residual).mean()


class CRPSLoss(nn.Module):
    """CRPS loss module for use in training."""

    def __init__(self, sigma: Optional[float] = None):
        """
        Args:
            sigma: Fixed predictive std. If None, estimated per batch.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        sigma = torch.tensor(self.sigma) if self.sigma else None
        return crps_gaussian(y_true, y_pred, sigma)


class HuberLoss(nn.Module):
    """Huber loss module."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return huber_loss(y_true, y_pred, self.delta)
