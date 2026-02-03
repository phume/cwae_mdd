"""Synthetic dataset generators for contextual anomaly detection."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SyntheticData:
    """Container for synthetic dataset."""
    x: np.ndarray  # Behavioral features [N, d]
    c: np.ndarray  # Context features [N, k]
    y: np.ndarray  # Labels (0 = normal, 1 = anomaly) [N]

    def to_tensors(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return (
            torch.tensor(self.x, dtype=torch.float32, device=device),
            torch.tensor(self.c, dtype=torch.float32, device=device),
            torch.tensor(self.y, dtype=torch.float32, device=device),
        )


class SyntheticDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic data."""

    def __init__(self, data: SyntheticData):
        self.x = torch.tensor(data.x, dtype=torch.float32)
        self.c = torch.tensor(data.c, dtype=torch.float32)
        self.y = torch.tensor(data.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.c[idx], self.y[idx]


def inject_anomalies(
    x: np.ndarray,
    c: np.ndarray,
    anomaly_rate: float,
    anomaly_type: str = "shift",
    shift_scale: float = 4.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into data.

    Args:
        x: Behavioral features
        c: Context features
        anomaly_rate: Fraction of samples to make anomalous
        anomaly_type: "shift" (additive), "scale", or "swap"
        shift_scale: Magnitude of shift in standard deviations
        rng: Random generator

    Returns:
        x_modified: Features with anomalies
        labels: Binary labels
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    n_anomalies = int(n * anomaly_rate)

    # Select anomaly indices
    anomaly_idx = rng.choice(n, size=n_anomalies, replace=False)
    labels = np.zeros(n, dtype=np.int32)
    labels[anomaly_idx] = 1

    x_modified = x.copy()

    if anomaly_type == "shift":
        # Add shift to random features
        for idx in anomaly_idx:
            feat_idx = rng.integers(0, x.shape[1])
            sigma = x[:, feat_idx].std()
            shift = rng.choice([-1, 1]) * shift_scale * sigma
            x_modified[idx, feat_idx] += shift

    elif anomaly_type == "scale":
        # Multiply random features by large factor
        for idx in anomaly_idx:
            feat_idx = rng.integers(0, x.shape[1])
            factor = rng.uniform(3.0, 5.0) * rng.choice([-1, 1])
            x_modified[idx, feat_idx] *= factor

    elif anomaly_type == "swap":
        # Swap with sample from different context region
        for idx in anomaly_idx:
            # Find sample with different context
            distances = np.linalg.norm(c - c[idx], axis=1)
            far_idx = np.argsort(distances)[-n//10:]  # Top 10% most different
            swap_idx = rng.choice(far_idx)
            x_modified[idx] = x[swap_idx]

    return x_modified, labels


def generate_linear(
    n_samples: int = 10000,
    context_dim: int = 2,
    behavior_dim: int = 10,
    anomaly_rate: float = 0.05,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> SyntheticData:
    """Generate linear relationship: y = Ac + noise.

    Tests basic context conditioning.
    """
    rng = np.random.default_rng(seed)

    # Context uniformly distributed
    c = rng.uniform(-2, 2, size=(n_samples, context_dim))

    # Linear mapping
    A = rng.standard_normal((context_dim, behavior_dim))
    x = c @ A + noise_scale * rng.standard_normal((n_samples, behavior_dim))

    # Inject anomalies
    x, y = inject_anomalies(x, c, anomaly_rate, "shift", rng=rng)

    return SyntheticData(x=x, c=c, y=y)


def generate_scale(
    n_samples: int = 10000,
    context_dim: int = 2,
    behavior_dim: int = 10,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> SyntheticData:
    """Generate scale relationship: y = diag(|c|) * noise.

    Tests context-dependent variance modeling.
    """
    rng = np.random.default_rng(seed)

    # Context uniformly distributed (positive for scale)
    c = rng.uniform(0.5, 3.0, size=(n_samples, context_dim))

    # Scale depends on context
    scales = np.abs(c).mean(axis=1, keepdims=True)
    x = scales * rng.standard_normal((n_samples, behavior_dim))

    # Inject anomalies (scale violations)
    x, y = inject_anomalies(x, c, anomaly_rate, "scale", rng=rng)

    return SyntheticData(x=x, c=c, y=y)


def generate_multimodal(
    n_samples: int = 10000,
    n_modes: int = 4,
    behavior_dim: int = 10,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> SyntheticData:
    """Generate multimodal data with categorical context.

    Tests handling of discrete context and mode-specific anomalies.
    """
    rng = np.random.default_rng(seed)

    # Categorical context (one-hot encoded)
    mode_idx = rng.integers(0, n_modes, size=n_samples)
    c = np.zeros((n_samples, n_modes))
    c[np.arange(n_samples), mode_idx] = 1.0

    # Mode-specific means
    mode_means = rng.standard_normal((n_modes, behavior_dim)) * 3

    # Generate data
    x = np.zeros((n_samples, behavior_dim))
    for i in range(n_modes):
        mask = mode_idx == i
        n_mode = mask.sum()
        x[mask] = mode_means[i] + rng.standard_normal((n_mode, behavior_dim)) * 0.5

    # Anomalies: samples from wrong mode
    x, y = inject_anomalies(x, c, anomaly_rate, "swap", rng=rng)

    return SyntheticData(x=x, c=c, y=y)


def generate_nonlinear(
    n_samples: int = 10000,
    context_dim: int = 2,
    behavior_dim: int = 10,
    hidden_dim: int = 32,
    anomaly_rate: float = 0.05,
    noise_scale: float = 0.3,
    seed: int = 42,
) -> SyntheticData:
    """Generate nonlinear relationship using random MLP.

    Tests nonlinear context-behavior modeling.
    """
    rng = np.random.default_rng(seed)

    # Context uniformly distributed
    c = rng.uniform(-2, 2, size=(n_samples, context_dim))

    # Random MLP for nonlinear mapping
    W1 = rng.standard_normal((context_dim, hidden_dim)) * np.sqrt(2.0 / context_dim)
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, behavior_dim)) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(behavior_dim)

    # Forward pass with ReLU
    h = np.maximum(0, c @ W1 + b1)
    x = h @ W2 + b2 + noise_scale * rng.standard_normal((n_samples, behavior_dim))

    # Inject anomalies
    x, y = inject_anomalies(x, c, anomaly_rate, "shift", rng=rng)

    return SyntheticData(x=x, c=c, y=y)


def generate_skewed(
    n_samples: int = 10000,
    context_dim: int = 2,
    behavior_dim: int = 10,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> SyntheticData:
    """Generate log-normal data with context-dependent parameters.

    Tests numerical stability with highly skewed distributions.
    """
    rng = np.random.default_rng(seed)

    # Context uniformly distributed
    c = rng.uniform(0, 2, size=(n_samples, context_dim))

    # Log-normal with context-dependent location
    mu = c.mean(axis=1, keepdims=True)  # [N, 1]
    x = np.exp(mu + rng.standard_normal((n_samples, behavior_dim)))

    # Inject anomalies (extreme values)
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    labels = np.zeros(n_samples, dtype=np.int32)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        feat_idx = rng.integers(0, behavior_dim)
        # Make extremely large or small
        if rng.random() > 0.5:
            x[idx, feat_idx] *= rng.uniform(10, 100)
        else:
            x[idx, feat_idx] /= rng.uniform(10, 100)

    return SyntheticData(x=x, c=c, y=labels)


def generate_aml_realistic(
    n_samples: int = 100000,
    behavior_dim: int = 8,
    anomaly_rate: float = 0.02,
    seed: int = 42,
) -> SyntheticData:
    """Generate realistic AML-like data with extreme characteristics.

    Mimics real financial transaction data:
    - Zero-inflation: Most clients have $0 for wire, international, etc.
    - Heavy tails: Non-zero values follow Pareto distribution (0 to 10M+)
    - Context-dependent sparsity: Retail clients rarely wire, corporates wire frequently
    - Context-dependent scale: Corporate amounts are much larger

    Context features (4):
    - client_type: 0=retail, 1=small_biz, 2=corporate (encoded as 3 one-hot)
    - account_age_years: 0-20

    Behavioral features (8):
    - wire_total: Sum of wire amounts (0 to 10M)
    - wire_count: Number of wire transactions (0 to 1000)
    - intl_total: International transfer total
    - intl_count: International transfer count
    - cash_total: Cash deposits total
    - cash_count: Cash deposit count
    - ach_total: ACH transfer total
    - ach_count: ACH transfer count

    Anomalies: Unusual activity patterns given client type
    """
    rng = np.random.default_rng(seed)

    # Client type distribution: 70% retail, 20% small_biz, 10% corporate
    client_type_probs = [0.70, 0.20, 0.10]
    client_type = rng.choice(3, size=n_samples, p=client_type_probs)

    # Account age (years) - uniform 0-20
    account_age = rng.uniform(0, 20, size=n_samples)

    # Context: one-hot client type + account age
    c = np.zeros((n_samples, 4))
    c[np.arange(n_samples), client_type] = 1.0
    c[:, 3] = account_age / 20.0  # Normalize to [0, 1]

    # Initialize behavioral features
    x = np.zeros((n_samples, behavior_dim))

    # Parameters by client type: [zero_rate, pareto_alpha, scale]
    # Higher alpha = lighter tail, lower alpha = heavier tail
    params = {
        # Wire transfers
        'wire': {
            0: {'zero_rate': 0.95, 'alpha': 1.5, 'scale': 1000},      # Retail: 95% zero, small amounts
            1: {'zero_rate': 0.70, 'alpha': 1.3, 'scale': 10000},     # Small biz: 70% zero, medium
            2: {'zero_rate': 0.30, 'alpha': 1.1, 'scale': 100000},    # Corporate: 30% zero, large
        },
        # International
        'intl': {
            0: {'zero_rate': 0.98, 'alpha': 1.5, 'scale': 500},
            1: {'zero_rate': 0.85, 'alpha': 1.3, 'scale': 5000},
            2: {'zero_rate': 0.50, 'alpha': 1.2, 'scale': 50000},
        },
        # Cash
        'cash': {
            0: {'zero_rate': 0.60, 'alpha': 2.0, 'scale': 500},
            1: {'zero_rate': 0.40, 'alpha': 1.8, 'scale': 2000},
            2: {'zero_rate': 0.80, 'alpha': 1.5, 'scale': 5000},      # Corporates use less cash
        },
        # ACH
        'ach': {
            0: {'zero_rate': 0.30, 'alpha': 2.0, 'scale': 1000},
            1: {'zero_rate': 0.20, 'alpha': 1.8, 'scale': 5000},
            2: {'zero_rate': 0.10, 'alpha': 1.5, 'scale': 50000},
        },
    }

    feature_map = [('wire', 0, 1), ('intl', 2, 3), ('cash', 4, 5), ('ach', 6, 7)]

    for feat_name, total_idx, count_idx in feature_map:
        for ctype in [0, 1, 2]:
            mask = client_type == ctype
            n_ctype = mask.sum()

            p = params[feat_name][ctype]

            # Zero-inflation: determine which clients have activity
            has_activity = rng.random(n_ctype) > p['zero_rate']
            n_active = has_activity.sum()

            if n_active > 0:
                # Generate counts (Poisson, context-dependent rate)
                base_rate = 5 * (ctype + 1) * (1 + account_age[mask][has_activity] / 20)
                counts = rng.poisson(base_rate)
                counts = np.maximum(counts, 1)  # At least 1 if active

                # Generate total amounts (Pareto distribution for heavy tails)
                # Pareto: x = scale * (1/U)^(1/alpha) where U ~ Uniform(0,1)
                u = rng.random(n_active)
                amounts = p['scale'] * (1.0 / (u + 1e-10)) ** (1.0 / p['alpha'])

                # Scale by account age (older = potentially larger)
                age_factor = 1 + account_age[mask][has_activity] / 10
                amounts = amounts * age_factor

                # Cap at realistic max (10M) but allow some extreme values
                amounts = np.minimum(amounts, 10_000_000)

                x[mask, total_idx][has_activity] = amounts
                x[mask, count_idx][has_activity] = counts

    # Labels: start with all normal
    labels = np.zeros(n_samples, dtype=np.int32)

    # Inject anomalies: unusual patterns for client type
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        ctype = client_type[idx]
        anomaly_type = rng.choice(['wrong_pattern', 'extreme_amount', 'suspicious_combo'])

        if anomaly_type == 'wrong_pattern':
            # Retail client with corporate-level wire activity
            if ctype == 0:  # Retail
                x[idx, 0] = rng.pareto(1.1) * 100000  # Large wire
                x[idx, 1] = rng.poisson(20)  # Many wires
            # Corporate with unusual cash activity
            elif ctype == 2:  # Corporate
                x[idx, 4] = rng.pareto(1.2) * 50000  # Large cash
                x[idx, 5] = rng.poisson(50)  # Many cash deposits

        elif anomaly_type == 'extreme_amount':
            # Extreme value for any feature
            feat_idx = rng.choice([0, 2, 4, 6])  # Amount features
            x[idx, feat_idx] = rng.pareto(1.05) * 500000  # Very heavy tail

        elif anomaly_type == 'suspicious_combo':
            # High international + high cash (structuring pattern)
            x[idx, 2] = rng.pareto(1.2) * 30000  # International
            x[idx, 3] = rng.poisson(15)
            x[idx, 4] = rng.pareto(1.3) * 9000   # Cash just under reporting threshold
            x[idx, 5] = rng.poisson(30)  # Many small cash deposits

    return SyntheticData(x=x.astype(np.float32), c=c.astype(np.float32), y=labels)


def create_train_test_split(
    data: SyntheticData,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[SyntheticData, SyntheticData]:
    """Split data into train and test sets."""
    rng = np.random.default_rng(seed)
    n = len(data.y)

    # Shuffle indices
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    train_data = SyntheticData(
        x=data.x[train_idx],
        c=data.c[train_idx],
        y=data.y[train_idx],
    )

    test_data = SyntheticData(
        x=data.x[test_idx],
        c=data.c[test_idx],
        y=data.y[test_idx],
    )

    return train_data, test_data
