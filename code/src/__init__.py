"""CWAE-MMD: Conditional Wasserstein Autoencoder with MMD for Anomaly Detection.

Main modules:
- models: CWAE-MMD, CVAE, and variants
- training: Training utilities
- losses: Alternative loss functions (CRPS, Huber)
- evaluation: Metrics and evaluation utilities
- data: Synthetic dataset loaders
"""

from . import models
from . import training
from . import losses
from . import evaluation
from . import data
