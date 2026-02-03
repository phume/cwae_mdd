"""Data generation and loading utilities."""
from .synthetic import (
    generate_linear,
    generate_scale,
    generate_multimodal,
    generate_nonlinear,
    generate_skewed,
    SyntheticDataset,
)

__all__ = [
    "generate_linear",
    "generate_scale",
    "generate_multimodal",
    "generate_nonlinear",
    "generate_skewed",
    "SyntheticDataset",
]
