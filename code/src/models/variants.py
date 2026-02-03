"""CWAE-MMD scoring and loss variants.

Scoring Variants:
- CWAE-MMD-IF: IF on residuals only (base)
- CWAE-MMD-DIF: DIF on residuals
- CWAE-MMD-IF-Latent: IF on latent only
- CWAE-MMD-IF-Dual: IF on latent + residuals (max aggregation)
- CWAE-MMD-DIF-Dual: DIF on latent + residuals

Loss Variants:
- CWAE-MMD-CRPS: CRPS reconstruction loss (robust to outliers)
- CWAE-MMD-Huber: Huber reconstruction loss (smooth L1/L2)

Kernel Variants:
- CWAE-MMD-IMQ-Extreme: Multi-scale IMQ with extra tail scales

Data Handling Variants:
- CWAE-MMD-TailCompress: With monotonic tail compression preprocessing
- CWAE-MMD-Oversample: With strategic tail oversampling
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List, Literal

from .cwae_mmd import CWAEMMD, Encoder, Decoder, compute_mmd


class DeepIsolationForestScorer:
    """Deep Isolation Forest scorer with random MLP projections."""

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
        self.input_dim = input_dim

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

    def fit(self, x: np.ndarray):
        """Fit on data."""
        x_tensor = torch.tensor(x, dtype=torch.float32)

        for proj, forest in zip(self.projections, self.forests):
            with torch.no_grad():
                h = proj(x_tensor).numpy()
            forest.fit(h)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        x_tensor = torch.tensor(x, dtype=torch.float32)

        all_scores = []
        for proj, forest in zip(self.projections, self.forests):
            with torch.no_grad():
                h = proj(x_tensor).numpy()
            scores = -forest.decision_function(h)
            all_scores.append(scores)

        # Average across projections
        return np.mean(all_scores, axis=0)


class CWAEMMDVariant(CWAEMMD):
    """CWAE-MMD with configurable scoring variants."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        mmd_weight: float = 1.0,
        kernel: str = "imq",
        kernel_scales: List[float] = None,
        scorer_type: Literal["if", "dif"] = "if",
        score_space: Literal["residual", "latent", "dual"] = "residual",
        n_dif_projections: int = 6,
        dif_hidden_dim: int = 64,
    ):
        """Initialize CWAE-MMD variant.

        Args:
            input_dim: Behavioral feature dimension
            context_dim: Context feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer sizes for encoder/decoder
            mmd_weight: Weight for MMD regularization
            kernel: MMD kernel type ("imq" or "rbf")
            kernel_scales: Bandwidth scales for multi-scale kernel
            scorer_type: "if" (Isolation Forest) or "dif" (Deep IF)
            score_space: "residual", "latent", or "dual" (both)
            n_dif_projections: Number of DIF random projections
            dif_hidden_dim: DIF projection hidden dimension
        """
        super().__init__(
            input_dim=input_dim,
            context_dim=context_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            mmd_weight=mmd_weight,
            kernel=kernel,
            kernel_scales=kernel_scales,
        )

        self.scorer_type = scorer_type
        self.score_space = score_space
        self.n_dif_projections = n_dif_projections
        self.dif_hidden_dim = dif_hidden_dim

        # Scorers (initialized in fit_scorer)
        self.residual_scorer = None
        self.latent_scorer = None

    def fit_scorer(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        n_estimators: int = 100,
        contamination: float = 0.05,
    ):
        """Fit scorer(s) based on configuration."""
        self.eval()
        with torch.no_grad():
            x_hat, z, residual = self(x, c)

        # Compute training residual statistics
        self.sigma_train = residual.std(dim=0).clamp(min=1e-8)
        residual_norm = (residual / self.sigma_train).cpu().numpy()
        z_np = z.cpu().numpy()

        # Fit residual scorer if needed
        if self.score_space in ["residual", "dual"]:
            if self.scorer_type == "if":
                self.residual_scorer = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                )
                self.residual_scorer.fit(residual_norm)
            else:  # dif
                self.residual_scorer = DeepIsolationForestScorer(
                    input_dim=self.input_dim,
                    n_projections=self.n_dif_projections,
                    hidden_dim=self.dif_hidden_dim,
                    n_estimators=n_estimators,
                    contamination=contamination,
                )
                self.residual_scorer.fit(residual_norm)

        # Fit latent scorer if needed
        if self.score_space in ["latent", "dual"]:
            if self.scorer_type == "if":
                self.latent_scorer = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                )
                self.latent_scorer.fit(z_np)
            else:  # dif
                self.latent_scorer = DeepIsolationForestScorer(
                    input_dim=self.latent_dim,
                    n_projections=self.n_dif_projections,
                    hidden_dim=self.dif_hidden_dim,
                    n_estimators=n_estimators,
                    contamination=contamination,
                )
                self.latent_scorer.fit(z_np)

    def score(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        aggregation: Literal["max", "mean", "sum"] = "max",
    ) -> np.ndarray:
        """Compute anomaly scores based on configuration.

        Args:
            x: Features to score
            c: Context
            aggregation: How to combine scores in dual mode

        Returns:
            Anomaly scores in [0, 1] (higher = more anomalous)
        """
        self.eval()
        with torch.no_grad():
            x_hat, z, residual = self(x, c)

        residual_norm = (residual / self.sigma_train).cpu().numpy()
        z_np = z.cpu().numpy()

        scores = None

        # Score based on configuration
        if self.score_space == "residual":
            if self.scorer_type == "if":
                scores = -self.residual_scorer.decision_function(residual_norm)
            else:
                scores = self.residual_scorer.score(residual_norm)

        elif self.score_space == "latent":
            if self.scorer_type == "if":
                scores = -self.latent_scorer.decision_function(z_np)
            else:
                scores = self.latent_scorer.score(z_np)

        elif self.score_space == "dual":
            # Get both scores
            if self.scorer_type == "if":
                scores_residual = -self.residual_scorer.decision_function(residual_norm)
                scores_latent = -self.latent_scorer.decision_function(z_np)
            else:
                scores_residual = self.residual_scorer.score(residual_norm)
                scores_latent = self.latent_scorer.score(z_np)

            # Normalize each before aggregation
            scores_residual = (scores_residual - scores_residual.min()) / (
                scores_residual.max() - scores_residual.min() + 1e-8
            )
            scores_latent = (scores_latent - scores_latent.min()) / (
                scores_latent.max() - scores_latent.min() + 1e-8
            )

            # Aggregate
            if aggregation == "max":
                scores = np.maximum(scores_residual, scores_latent)
            elif aggregation == "mean":
                scores = (scores_residual + scores_latent) / 2
            elif aggregation == "sum":
                scores = scores_residual + scores_latent

        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores

    @property
    def variant_name(self) -> str:
        """Get descriptive variant name."""
        scorer = "IF" if self.scorer_type == "if" else "DIF"
        space = {
            "residual": "Recon",
            "latent": "Latent",
            "dual": "Dual",
        }[self.score_space]
        return f"CWAE-MMD-{scorer}-{space}"


# Convenience factory functions
def create_cwae_mmd_if(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDVariant:
    """CWAE-MMD with IF on residuals (base variant)."""
    return CWAEMMDVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="if",
        score_space="residual",
        **kwargs,
    )


def create_cwae_mmd_dif(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDVariant:
    """CWAE-MMD with DIF on residuals."""
    return CWAEMMDVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="dif",
        score_space="residual",
        **kwargs,
    )


def create_cwae_mmd_if_latent(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDVariant:
    """CWAE-MMD with IF on latent space only."""
    return CWAEMMDVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="if",
        score_space="latent",
        **kwargs,
    )


def create_cwae_mmd_if_dual(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDVariant:
    """CWAE-MMD with IF on both latent and residuals."""
    return CWAEMMDVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="if",
        score_space="dual",
        **kwargs,
    )


def create_cwae_mmd_dif_dual(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDVariant:
    """CWAE-MMD with DIF on both latent and residuals."""
    return CWAEMMDVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="dif",
        score_space="dual",
        **kwargs,
    )


# ============================================================================
# Loss Variant Factories
# ============================================================================

def create_cwae_mmd_crps(input_dim: int, context_dim: int, **kwargs) -> CWAEMMD:
    """CWAE-MMD-CRPS: Uses CRPS loss instead of MSE.

    CRPS (Continuous Ranked Probability Score) is more robust to outliers
    than MSE because it doesn't square the errors.
    """
    return CWAEMMD(
        input_dim=input_dim,
        context_dim=context_dim,
        recon_loss="crps",
        **kwargs,
    )


def create_cwae_mmd_huber(input_dim: int, context_dim: int, delta: float = 1.0, **kwargs) -> CWAEMMD:
    """CWAE-MMD-Huber: Uses Huber loss instead of MSE.

    Huber loss is L2 for small errors and L1 for large errors,
    providing robustness to outliers while maintaining smoothness near zero.
    """
    return CWAEMMD(
        input_dim=input_dim,
        context_dim=context_dim,
        recon_loss="huber",
        huber_delta=delta,
        **kwargs,
    )


# ============================================================================
# Kernel Variant Factories
# ============================================================================

# Extended scales for extreme/heavy-tailed data
IMQ_EXTREME_SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def create_cwae_mmd_imq_extreme(input_dim: int, context_dim: int, **kwargs) -> CWAEMMD:
    """CWAE-MMD-IMQ-Extreme: Multi-scale IMQ with extended tail scales.

    Uses additional bandwidth scales (10.0, 20.0) to better capture
    similarity between extreme values in heavy-tailed distributions.
    """
    return CWAEMMD(
        input_dim=input_dim,
        context_dim=context_dim,
        kernel="imq",
        kernel_scales=IMQ_EXTREME_SCALES,
        **kwargs,
    )


# ============================================================================
# Combined Variants (Loss + Scoring)
# ============================================================================

class CWAEMMDCRPSVariant(CWAEMMDVariant):
    """CWAE-MMD with CRPS loss and configurable scoring."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        scorer_type: Literal["if", "dif"] = "if",
        score_space: Literal["residual", "latent", "dual"] = "residual",
        **kwargs,
    ):
        # Remove recon_loss if passed in kwargs to avoid conflict
        kwargs.pop("recon_loss", None)

        super().__init__(
            input_dim=input_dim,
            context_dim=context_dim,
            scorer_type=scorer_type,
            score_space=score_space,
            **kwargs,
        )
        # Override to use CRPS
        self.recon_loss_type = "crps"

    @property
    def variant_name(self) -> str:
        scorer = "IF" if self.scorer_type == "if" else "DIF"
        space = {
            "residual": "Recon",
            "latent": "Latent",
            "dual": "Dual",
        }[self.score_space]
        return f"CWAE-MMD-CRPS-{scorer}-{space}"


def create_cwae_mmd_crps_if(input_dim: int, context_dim: int, **kwargs) -> CWAEMMDCRPSVariant:
    """CWAE-MMD-CRPS-IF: CRPS loss with IF scoring on residuals."""
    return CWAEMMDCRPSVariant(
        input_dim=input_dim,
        context_dim=context_dim,
        scorer_type="if",
        score_space="residual",
        **kwargs,
    )


# ============================================================================
# Variant Registry
# ============================================================================

# All variants for experiments
ALL_VARIANTS = {
    # Scoring variants (MSE loss)
    "CWAE-MMD-IF": create_cwae_mmd_if,
    "CWAE-MMD-DIF": create_cwae_mmd_dif,
    "CWAE-MMD-IF-Latent": create_cwae_mmd_if_latent,
    "CWAE-MMD-IF-Dual": create_cwae_mmd_if_dual,
    "CWAE-MMD-DIF-Dual": create_cwae_mmd_dif_dual,
    # Loss variants
    "CWAE-MMD-CRPS": create_cwae_mmd_crps,
    "CWAE-MMD-CRPS-IF": create_cwae_mmd_crps_if,
    "CWAE-MMD-Huber": create_cwae_mmd_huber,
    # Kernel variants
    "CWAE-MMD-IMQ-Extreme": create_cwae_mmd_imq_extreme,
}


# Human-readable descriptions
VARIANT_DESCRIPTIONS = {
    "CWAE-MMD-IF": "Base: MSE loss, IF on residuals",
    "CWAE-MMD-DIF": "Deep IF on residuals",
    "CWAE-MMD-IF-Latent": "IF on latent space",
    "CWAE-MMD-IF-Dual": "IF on both latent and residuals",
    "CWAE-MMD-DIF-Dual": "Deep IF on both spaces",
    "CWAE-MMD-CRPS": "CRPS loss (outlier-robust)",
    "CWAE-MMD-CRPS-IF": "CRPS loss + IF scoring",
    "CWAE-MMD-Huber": "Huber loss (smooth L1/L2)",
    "CWAE-MMD-IMQ-Extreme": "Extended IMQ scales for heavy tails",
}
