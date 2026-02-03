"""CWAE-MMD: Conditional Wasserstein Autoencoder with MMD regularization."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple, List, Literal, Callable


class Encoder(nn.Module):
    """Deterministic encoder: maps (x, c) -> z."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = input_dim + context_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Encode input and context to latent code."""
        xc = torch.cat([x, c], dim=-1)
        return self.net(xc)


class Decoder(nn.Module):
    """Decoder: maps (z, c) -> x_hat."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        layers = []
        in_dim = latent_dim + context_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent code and context to reconstruction."""
        zc = torch.cat([z, c], dim=-1)
        return self.net(zc)


def compute_mmd(
    z: torch.Tensor,
    z_prior: torch.Tensor,
    kernel: str = "imq",
    scales: List[float] = None,
) -> torch.Tensor:
    """Compute MMD^2 between encoded samples and prior samples.

    Args:
        z: Encoded samples [B, latent_dim]
        z_prior: Prior samples [B, latent_dim]
        kernel: Kernel type ("imq" or "rbf")
        scales: Bandwidth scale multipliers for multi-scale kernel

    Returns:
        MMD^2 scalar
    """
    if scales is None:
        scales = [0.2, 0.5, 1.0, 2.0, 5.0]

    B = z.size(0)
    latent_dim = z.size(1)

    # Base bandwidth: expected squared distance between Gaussian samples
    # E[||z - z'||^2] = 2 * latent_dim for z, z' ~ N(0, I)
    base_bandwidth = 2.0 * latent_dim

    mmd = torch.tensor(0.0, device=z.device)

    for scale in scales:
        C = scale * base_bandwidth

        if kernel == "imq":
            # Inverse multiquadrics: k(x, y) = C / (C + ||x - y||^2)
            # k(z, z')
            zz_dist = torch.cdist(z, z, p=2).pow(2)
            k_zz = C / (C + zz_dist)

            # k(z_prior, z_prior')
            pp_dist = torch.cdist(z_prior, z_prior, p=2).pow(2)
            k_pp = C / (C + pp_dist)

            # k(z, z_prior)
            zp_dist = torch.cdist(z, z_prior, p=2).pow(2)
            k_zp = C / (C + zp_dist)

        elif kernel == "rbf":
            # RBF: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            sigma_sq = C / 2

            zz_dist = torch.cdist(z, z, p=2).pow(2)
            k_zz = torch.exp(-zz_dist / (2 * sigma_sq))

            pp_dist = torch.cdist(z_prior, z_prior, p=2).pow(2)
            k_pp = torch.exp(-pp_dist / (2 * sigma_sq))

            zp_dist = torch.cdist(z, z_prior, p=2).pow(2)
            k_zp = torch.exp(-zp_dist / (2 * sigma_sq))

        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Unbiased MMD^2 estimate
        # Remove diagonal for unbiased estimate
        mask = 1.0 - torch.eye(B, device=z.device)

        term1 = (k_zz * mask).sum() / (B * (B - 1))
        term2 = (k_pp * mask).sum() / (B * (B - 1))
        term3 = k_zp.mean()

        mmd = mmd + term1 + term2 - 2 * term3

    return mmd / len(scales)


class CWAEMMD(nn.Module):
    """Conditional Wasserstein Autoencoder with MMD regularization.

    Variants:
    - CWAE-MMD: Base model with MSE reconstruction loss
    - CWAE-MMD-CRPS: Uses CRPS loss (more robust to outliers)
    - CWAE-MMD-Huber: Uses Huber loss (smooth L1/L2 blend)
    - CWAE-MMD-IMQ-Extreme: Multi-scale IMQ with extra tail scales
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        mmd_weight: float = 1.0,
        kernel: str = "imq",
        kernel_scales: List[float] = None,
        recon_loss: Literal["mse", "crps", "huber"] = "mse",
        huber_delta: float = 1.0,
        monitor_mmd: bool = False,
    ):
        """Initialize CWAE-MMD.

        Args:
            input_dim: Behavioral feature dimension
            context_dim: Context feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer sizes for encoder/decoder
            mmd_weight: Weight for MMD regularization (lambda)
            kernel: MMD kernel type ("imq" or "rbf")
            kernel_scales: Bandwidth scales for multi-scale kernel
            recon_loss: Reconstruction loss type ("mse", "crps", "huber")
            huber_delta: Delta parameter for Huber loss
            monitor_mmd: Track MMD history for collapse detection
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]
        if kernel_scales is None:
            kernel_scales = [0.2, 0.5, 1.0, 2.0, 5.0]

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.mmd_weight = mmd_weight
        self.kernel = kernel
        self.kernel_scales = kernel_scales
        self.recon_loss_type = recon_loss
        self.huber_delta = huber_delta
        self.monitor_mmd = monitor_mmd

        self.encoder = Encoder(input_dim, context_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, context_dim, input_dim, hidden_dims[::-1])

        # For anomaly scoring
        self.sigma_train: Optional[torch.Tensor] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.isolation_forest_latent: Optional[IsolationForest] = None

        # MMD monitoring for posterior collapse detection
        self.mmd_history: List[float] = []
        self._collapse_threshold = 1e-4

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            x_hat: Reconstruction
            z: Latent code
            residual: x - x_hat
        """
        z = self.encoder(x, c)
        x_hat = self.decoder(z, c)
        residual = x - x_hat
        return x_hat, z, residual

    def _compute_recon_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss based on configured type."""
        if self.recon_loss_type == "mse":
            return ((x - x_hat) ** 2).mean()

        elif self.recon_loss_type == "crps":
            # CRPS for Gaussian: more robust to outliers
            # Import here to avoid circular imports
            from ..losses import crps_gaussian
            return crps_gaussian(x, x_hat)

        elif self.recon_loss_type == "huber":
            # Huber loss: L2 for small errors, L1 for large
            residual = torch.abs(x - x_hat)
            quadratic = torch.clamp(residual, max=self.huber_delta)
            linear = residual - quadratic
            return (0.5 * quadratic**2 + self.huber_delta * linear).mean()

        else:
            raise ValueError(f"Unknown recon_loss type: {self.recon_loss_type}")

    def compute_loss(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute training loss.

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        x_hat, z, _ = self(x, c)

        # Reconstruction loss (configurable)
        recon_loss = self._compute_recon_loss(x, x_hat)

        # MMD loss
        z_prior = torch.randn_like(z)
        mmd_loss = compute_mmd(z, z_prior, self.kernel, self.kernel_scales)

        # Total loss
        loss = recon_loss + self.mmd_weight * mmd_loss

        # Monitor MMD for posterior collapse detection
        mmd_value = mmd_loss.item()
        if self.monitor_mmd:
            self.mmd_history.append(mmd_value)

        metrics = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "mmd_loss": mmd_value,
        }

        # Add collapse warning if MMD is suspiciously low
        if mmd_value < self._collapse_threshold:
            metrics["mmd_collapse_warning"] = True

        return loss, metrics

    def check_mmd_collapse(self, window: int = 10) -> dict:
        """Check for potential MMD/posterior collapse.

        Args:
            window: Number of recent epochs to analyze

        Returns:
            Dictionary with collapse diagnostics
        """
        if len(self.mmd_history) < window:
            return {"status": "insufficient_data", "n_samples": len(self.mmd_history)}

        recent = self.mmd_history[-window:]
        mean_mmd = np.mean(recent)
        std_mmd = np.std(recent)

        result = {
            "mean_mmd": mean_mmd,
            "std_mmd": std_mmd,
            "min_mmd": np.min(recent),
            "max_mmd": np.max(recent),
        }

        # Collapse if MMD is very low and stable
        if mean_mmd < self._collapse_threshold and std_mmd < self._collapse_threshold:
            result["status"] = "collapsed"
            result["warning"] = "MMD near zero - encoder may be ignoring input"
        # Healthy if MMD has some variance
        elif std_mmd > 0.01 * mean_mmd:
            result["status"] = "healthy"
        else:
            result["status"] = "stable"

        return result

    def reset_mmd_history(self):
        """Clear MMD history."""
        self.mmd_history = []

    def fit_scorer(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        n_estimators: int = 100,
        contamination: float = 0.05,
        fit_latent: bool = False,
    ):
        """Fit Isolation Forest on training residuals.

        Args:
            x: Training features
            c: Training context
            n_estimators: Number of IF trees
            contamination: Expected anomaly proportion
            fit_latent: Also fit IF on latent space (for dual-space scoring)
        """
        self.eval()
        with torch.no_grad():
            x_hat, z, residual = self(x, c)

        # Compute training residual statistics
        self.sigma_train = residual.std(dim=0).clamp(min=1e-8)

        # Normalize residuals
        residual_norm = (residual / self.sigma_train).cpu().numpy()

        # Fit IF on residuals
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(residual_norm)

        # Optionally fit IF on latent space
        if fit_latent:
            self.isolation_forest_latent = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.isolation_forest_latent.fit(z.cpu().numpy())

    def score(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        dual_space: bool = False,
    ) -> np.ndarray:
        """Compute anomaly scores.

        Args:
            x: Features to score
            c: Context
            dual_space: Use both latent and residual space scoring

        Returns:
            Anomaly scores in [0, 1] (higher = more anomalous)
        """
        if self.isolation_forest is None:
            raise RuntimeError("Must call fit_scorer before scoring")

        self.eval()
        with torch.no_grad():
            x_hat, z, residual = self(x, c)

        # Normalize residuals
        residual_norm = (residual / self.sigma_train).cpu().numpy()

        # IF score (negative of decision function, so higher = more anomalous)
        scores_residual = -self.isolation_forest.decision_function(residual_norm)

        if dual_space and self.isolation_forest_latent is not None:
            scores_latent = -self.isolation_forest_latent.decision_function(
                z.cpu().numpy()
            )
            # Max aggregation
            scores = np.maximum(scores_residual, scores_latent)
        else:
            scores = scores_residual

        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores

    def get_residuals(
        self, x: torch.Tensor, c: torch.Tensor, normalize: bool = True
    ) -> np.ndarray:
        """Get reconstruction residuals.

        Args:
            x: Features
            c: Context
            normalize: Divide by training sigma

        Returns:
            Residuals array
        """
        self.eval()
        with torch.no_grad():
            _, _, residual = self(x, c)

        if normalize and self.sigma_train is not None:
            residual = residual / self.sigma_train

        return residual.cpu().numpy()
