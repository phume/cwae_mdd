"""Baseline models for comparison."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple, List


class CVAEEncoder(nn.Module):
    """Stochastic encoder for CVAE: maps (x, c) -> (mu, log_var)."""

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

        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_log_var = nn.Linear(in_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to mean and log variance."""
        xc = torch.cat([x, c], dim=-1)
        h = self.net(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mu, log_var


class CVAEDecoder(nn.Module):
    """Decoder for CVAE: maps (z, c) -> (x_mu, x_log_var)."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        learn_variance: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        self.learn_variance = learn_variance

        layers = []
        in_dim = latent_dim + context_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, output_dim)

        if learn_variance:
            self.fc_log_var = nn.Linear(in_dim, output_dim)

    def forward(
        self, z: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode to reconstruction mean (and variance if learned)."""
        zc = torch.cat([z, c], dim=-1)
        h = self.net(zc)
        mu = self.fc_mu(h)

        if self.learn_variance:
            log_var = self.fc_log_var(h)
            log_var = torch.clamp(log_var, min=-10, max=10)
            return mu, log_var
        return mu, None


class CVAE(nn.Module):
    """Conditional Variational Autoencoder baseline."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        beta: float = 1.0,
        learn_variance: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.learn_variance = learn_variance

        self.encoder = CVAEEncoder(input_dim, context_dim, latent_dim, hidden_dims)
        self.decoder = CVAEDecoder(
            latent_dim, context_dim, input_dim, hidden_dims[::-1], learn_variance
        )

        # For anomaly scoring
        self.sigma_train: Optional[torch.Tensor] = None
        self.isolation_forest: Optional[IsolationForest] = None

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            x_hat: Reconstruction mean
            mu: Latent mean
            log_var: Latent log variance
            z: Sampled latent
        """
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        x_hat, x_log_var = self.decoder(z, c)
        return x_hat, mu, log_var, z

    def compute_loss(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute ELBO loss.

        Returns:
            loss: Negative ELBO
            metrics: Dictionary of loss components
        """
        x_hat, mu, log_var, z = self(x, c)

        # Reconstruction loss
        if self.learn_variance:
            _, x_log_var = self.decoder(z, c)
            # Gaussian NLL
            recon_loss = 0.5 * (
                x_log_var + ((x - x_hat) ** 2) / (torch.exp(x_log_var) + 1e-8)
            ).mean()
        else:
            recon_loss = ((x - x_hat) ** 2).mean()

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - var)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        # Check for NaN
        nan_detected = torch.isnan(loss).item()

        metrics = {
            "loss": loss.item() if not nan_detected else float("nan"),
            "recon_loss": recon_loss.item() if not nan_detected else float("nan"),
            "kl_loss": kl_loss.item() if not nan_detected else float("nan"),
            "nan_detected": nan_detected,
        }

        return loss, metrics

    def fit_scorer(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        n_estimators: int = 100,
        contamination: float = 0.05,
    ):
        """Fit Isolation Forest on training residuals."""
        self.eval()
        with torch.no_grad():
            x_hat, _, _, _ = self(x, c)
            residual = x - x_hat

        self.sigma_train = residual.std(dim=0).clamp(min=1e-8)
        residual_norm = (residual / self.sigma_train).cpu().numpy()

        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(residual_norm)

    def score(self, x: torch.Tensor, c: torch.Tensor) -> np.ndarray:
        """Compute anomaly scores."""
        if self.isolation_forest is None:
            raise RuntimeError("Must call fit_scorer before scoring")

        self.eval()
        with torch.no_grad():
            x_hat, _, _, _ = self(x, c)
            residual = x - x_hat

        residual_norm = (residual / self.sigma_train).cpu().numpy()

        # Check for NaN
        if np.any(np.isnan(residual_norm)):
            return np.full(len(x), np.nan)

        scores = -self.isolation_forest.decision_function(residual_norm)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores


class IsolationForestBaseline:
    """Isolation Forest baseline (no context modeling)."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, c: np.ndarray = None):
        """Fit on concatenated features."""
        if c is not None:
            features = np.concatenate([x, c], axis=1)
        else:
            features = x
        self.model.fit(features)

    def score(self, x: np.ndarray, c: np.ndarray = None) -> np.ndarray:
        """Compute anomaly scores."""
        if c is not None:
            features = np.concatenate([x, c], axis=1)
        else:
            features = x

        scores = -self.model.decision_function(features)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores


class DeepIsolationForest:
    """Deep Isolation Forest with random MLP projections."""

    def __init__(
        self,
        input_dim: int,
        n_projections: int = 6,
        hidden_dim: int = 64,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_projections = n_projections
        self.random_state = random_state

        # Create random frozen MLPs
        torch.manual_seed(random_state)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )
            for _ in range(n_projections)
        ])

        # Freeze projections
        for proj in self.projections:
            for param in proj.parameters():
                param.requires_grad = False

        # Create IF for each projection
        self.forests = [
            IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state + i,
                n_jobs=-1,
            )
            for i in range(n_projections)
        ]

    def fit(self, x: np.ndarray, c: np.ndarray = None):
        """Fit on projected features."""
        if c is not None:
            features = np.concatenate([x, c], axis=1)
        else:
            features = x

        x_tensor = torch.tensor(features, dtype=torch.float32)

        for i, (proj, forest) in enumerate(zip(self.projections, self.forests)):
            with torch.no_grad():
                h = proj(x_tensor).numpy()
            forest.fit(h)

    def score(self, x: np.ndarray, c: np.ndarray = None) -> np.ndarray:
        """Compute aggregated anomaly scores."""
        if c is not None:
            features = np.concatenate([x, c], axis=1)
        else:
            features = x

        x_tensor = torch.tensor(features, dtype=torch.float32)

        all_scores = []
        for proj, forest in zip(self.projections, self.forests):
            with torch.no_grad():
                h = proj(x_tensor).numpy()
            scores = -forest.decision_function(h)
            all_scores.append(scores)

        # Average across projections
        scores = np.mean(all_scores, axis=0)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores
