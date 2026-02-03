"""Model implementations."""
from .cwae_mmd import CWAEMMD
from .baselines import CVAE, IsolationForestBaseline, DeepIsolationForest
from .variants import (
    CWAEMMDVariant,
    CWAEMMDCRPSVariant,
    create_cwae_mmd_if,
    create_cwae_mmd_dif,
    create_cwae_mmd_if_latent,
    create_cwae_mmd_if_dual,
    create_cwae_mmd_dif_dual,
    create_cwae_mmd_crps,
    create_cwae_mmd_crps_if,
    create_cwae_mmd_huber,
    create_cwae_mmd_imq_extreme,
    ALL_VARIANTS,
    VARIANT_DESCRIPTIONS,
    IMQ_EXTREME_SCALES,
)

__all__ = [
    # Base models
    "CWAEMMD",
    "CVAE",
    "IsolationForestBaseline",
    "DeepIsolationForest",
    # Variant classes
    "CWAEMMDVariant",
    "CWAEMMDCRPSVariant",
    # Factory functions - Scoring variants
    "create_cwae_mmd_if",
    "create_cwae_mmd_dif",
    "create_cwae_mmd_if_latent",
    "create_cwae_mmd_if_dual",
    "create_cwae_mmd_dif_dual",
    # Factory functions - Loss variants
    "create_cwae_mmd_crps",
    "create_cwae_mmd_crps_if",
    "create_cwae_mmd_huber",
    # Factory functions - Kernel variants
    "create_cwae_mmd_imq_extreme",
    # Registry
    "ALL_VARIANTS",
    "VARIANT_DESCRIPTIONS",
    "IMQ_EXTREME_SCALES",
]
