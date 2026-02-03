#!/usr/bin/env python
"""Run CWAE-MMD on ALL available real datasets with comparison to baselines.

Datasets:
  - Thyroid (ODDS, auto-download)
  - Cardio (ODDS .mat)
  - Arrhythmia (ODDS .mat)
  - Ionosphere (ODDS .mat)
  - Pima (ODDS .mat)
  - WBC (ODDS .mat)
  - Vowels (ODDS .mat)
  - SAML-D (Kaggle CSV)
  - IEEE-CIS (Kaggle CSV)
  - PaySim (Kaggle CSV)
  - Credit Card Fraud (Kaggle CSV)

Methods:
  - IF (behavior only)
  - IF-concat (behavior + context)
  - DIF (Deep Isolation Forest on concat)
  - CVAE
  - CWAE-MMD-IF
  - CWAE-MMD-CRPS
  - CWAE-MMD-Huber
  - CWAE-MMD-DIF-Dual
  - CWAE-MMD-IF-Dual
  - CWAE-MMD-IF-Latent
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PUBLISH_DIR = os.path.dirname(PROJECT_DIR)  # CWAE_MDD_Publish/
DATA_DIR = os.path.join(PUBLISH_DIR, "data")
ODDS_DIR = os.path.join(DATA_DIR, "odds")
KAGGLE_DIR = os.path.join(DATA_DIR, "kaggle")
LOCAL_RAW = os.path.join(PROJECT_DIR, "data", "raw")

sys.path.insert(0, PROJECT_DIR)

from src.models import CWAEMMD
from src.models.variants import (
    create_cwae_mmd_if, create_cwae_mmd_dif,
    create_cwae_mmd_if_latent, create_cwae_mmd_if_dual, create_cwae_mmd_dif_dual,
    create_cwae_mmd_crps, create_cwae_mmd_huber,
)
from src.models.baselines import CVAE, IsolationForestBaseline, DeepIsolationForest
from src.training import train_model, set_seed
from src.evaluation.metrics import compute_metrics

# ============================================================================
# Configuration
# ============================================================================
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
RAW_FILE = os.path.join(RESULTS_DIR, "real_comparison_raw.csv")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "real_comparison_state.json")
ERROR_FILE = os.path.join(RESULTS_DIR, "real_comparison_errors.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CONFIG = {"epochs": 50, "patience": 10, "lr": 1e-3, "batch_size": 256}
SEEDS = [42, 123, 456]

MODEL_NAMES = [
    "IF", "IF-concat", "DIF", "CVAE",
    "CWAE-MMD-IF", "CWAE-MMD-CRPS", "CWAE-MMD-Huber",
    "CWAE-MMD-DIF-Dual", "CWAE-MMD-IF-Dual", "CWAE-MMD-IF-Latent",
]


# ============================================================================
# Data Loaders
# ============================================================================

def _standardize(arr):
    """Standardize array columns to zero mean, unit variance."""
    arr = np.nan_to_num(arr, nan=0.0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return (arr - arr.mean(axis=0)) / std


def load_odds_mat(name, mat_path, context_split="half"):
    """Load an ODDS .mat dataset.

    Args:
        name: Dataset name for printing
        mat_path: Path to .mat file
        context_split: How to split features into context/behavior.
            "half" = first half context, second half behavior
            int = first N features as context

    Returns:
        (context, behavior, labels) or None if file not found
    """
    if not os.path.exists(mat_path):
        print(f"  [SKIP] {name}: file not found at {mat_path}")
        return None

    data = loadmat(mat_path)
    X = data['X'].astype(np.float64)
    y = data['y'].ravel().astype(int)

    n_features = X.shape[1]
    if isinstance(context_split, int):
        n_ctx = context_split
    else:
        n_ctx = max(1, n_features // 2)

    context = _standardize(X[:, :n_ctx])
    behavior = _standardize(X[:, n_ctx:])

    if behavior.shape[1] == 0:
        context = _standardize(X[:, :1])
        behavior = _standardize(X[:, 1:])

    print(f"  {name}: N={len(y)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={y.mean():.4f}")
    return context, behavior, y


def load_thyroid():
    """Load thyroid from local raw or ODDS data."""
    for path in [
        os.path.join(LOCAL_RAW, "thyroid.mat"),
        os.path.join(ODDS_DIR, "thyroid.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("thyroid", path)

    # Auto-download
    import urllib.request
    os.makedirs(LOCAL_RAW, exist_ok=True)
    filepath = os.path.join(LOCAL_RAW, "thyroid.mat")
    url = "https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1"
    print("  Downloading thyroid dataset...")
    urllib.request.urlretrieve(url, filepath)
    return load_odds_mat("thyroid", filepath)


def load_cardio():
    """Load cardio from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "cardio.mat"),
        os.path.join(LOCAL_RAW, "cardio.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("cardio", path)
    print("  [SKIP] cardio: not found")
    return None


def load_arrhythmia():
    """Load arrhythmia from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "arrhythmia.mat"),
        os.path.join(LOCAL_RAW, "arrhythmia.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("arrhythmia", path)
    print("  [SKIP] arrhythmia: not found")
    return None


def load_ionosphere():
    """Load ionosphere from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "ionosphere.mat"),
        os.path.join(LOCAL_RAW, "ionosphere.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("ionosphere", path)
    print("  [SKIP] ionosphere: not found")
    return None


def load_pima():
    """Load pima from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "pima.mat"),
        os.path.join(LOCAL_RAW, "pima.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("pima", path)
    print("  [SKIP] pima: not found")
    return None


def load_wbc():
    """Load WBC from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "wbc.mat"),
        os.path.join(LOCAL_RAW, "wbc.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("wbc", path)
    print("  [SKIP] wbc: not found")
    return None


def load_vowels():
    """Load vowels from ODDS."""
    for path in [
        os.path.join(ODDS_DIR, "vowels.mat"),
        os.path.join(LOCAL_RAW, "vowels.mat"),
    ]:
        if os.path.exists(path):
            return load_odds_mat("vowels", path)
    print("  [SKIP] vowels: not found")
    return None


def load_saml(random_state=None, max_samples=100000):
    """Load SAML-D dataset."""
    data_path = None
    possible = [
        os.path.join(KAGGLE_DIR, "SAML-D.csv"),
        os.path.join(LOCAL_RAW, "SAML-D.csv"),
    ]
    for p in possible:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print("  [SKIP] SAML-D: not found")
        return None

    print(f"  Loading SAML-D from {data_path}...")
    df = pd.read_csv(data_path, nrows=max_samples * 2 if max_samples else None)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    # Find label column
    label_cols = ['Is_laundering', 'Is_Laundering', 'is_laundering', 'isFraud', 'label']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        print(f"  [SKIP] SAML-D: no label column found. Available: {df.columns.tolist()[:10]}")
        return None

    labels = df[label_col].values.astype(int)

    # Context features
    ctx_cols = ['Sender_bank_location', 'Receiver_bank_location',
                'Payment_currency', 'Received_currency', 'Payment_type']
    ctx_cols = [c for c in ctx_cols if c in df.columns]
    if ctx_cols:
        ctx_df = df[ctx_cols].copy()
        for col in ctx_df.columns:
            ctx_df[col] = pd.Categorical(ctx_df[col]).codes
        context = _standardize(ctx_df.values.astype(float))
    else:
        context = np.zeros((len(df), 1))

    # Behavior features
    if 'Amount' in df.columns:
        amt = df['Amount'].values.astype(float).reshape(-1, 1)
        behavior = _standardize(np.column_stack([amt, np.log1p(np.abs(amt))]))
    else:
        behavior = np.zeros((len(df), 1))

    print(f"  SAML-D: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.4f}")
    return context, behavior, labels


def load_ieee_cis(random_state=None, max_samples=100000):
    """Load IEEE-CIS Fraud Detection dataset."""
    data_path = None
    possible = [
        os.path.join(KAGGLE_DIR, "IEEE-CIS Fraud Detection", "train_transaction.csv"),
        os.path.join(KAGGLE_DIR, "train_transaction.csv"),
        os.path.join(LOCAL_RAW, "train_transaction.csv"),
    ]
    for p in possible:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print("  [SKIP] IEEE-CIS: not found")
        return None

    print(f"  Loading IEEE-CIS from {data_path}...")
    df = pd.read_csv(data_path, nrows=max_samples)
    labels = df['isFraud'].values

    # Context
    ctx_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    ctx_cols = [c for c in ctx_cols if c in df.columns]
    ctx_df = df[ctx_cols].copy()
    for col in ctx_df.columns:
        if ctx_df[col].dtype == 'object':
            ctx_df[col] = pd.Categorical(ctx_df[col]).codes
        ctx_df[col] = ctx_df[col].fillna(-1)
    context = _standardize(ctx_df.values.astype(float))

    # Behavior
    beh_cols = ['TransactionAmt', 'TransactionDT']
    beh_cols += [f'C{i}' for i in range(1, 15) if f'C{i}' in df.columns]
    beh_cols += [f'D{i}' for i in range(1, 16) if f'D{i}' in df.columns]
    beh_cols = [c for c in beh_cols if c in df.columns]
    behavior = _standardize(df[beh_cols].fillna(0).values.astype(float))

    print(f"  IEEE-CIS: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.4f}")
    return context, behavior, labels


def load_paysim(random_state=None, max_samples=100000):
    """Load PaySim dataset."""
    data_path = None
    possible = [
        os.path.join(KAGGLE_DIR, "Synthetic Financial Datasets For Fraud Detection.csv"),
        os.path.join(KAGGLE_DIR, "paysim.csv"),
        os.path.join(LOCAL_RAW, "paysim.csv"),
    ]
    for p in possible:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print("  [SKIP] PaySim: not found")
        return None

    print(f"  Loading PaySim from {data_path}...")
    df = pd.read_csv(data_path, nrows=max_samples)
    labels = df['isFraud'].values

    # Context
    ctx_df = pd.DataFrame()
    ctx_df['type'] = pd.Categorical(df['type']).codes
    ctx_df['step'] = df['step']
    ctx_df['nameOrig_hash'] = df['nameOrig'].apply(lambda x: hash(x) % 10000)
    context = _standardize(ctx_df.values.astype(float))

    # Behavior
    beh_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    behavior = df[beh_cols].values.astype(float)
    balance_change_orig = behavior[:, 2] - behavior[:, 1]
    balance_change_dest = behavior[:, 4] - behavior[:, 3]
    behavior = _standardize(np.column_stack([behavior, balance_change_orig, balance_change_dest]))

    print(f"  PaySim: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.4f}")
    return context, behavior, labels


def load_creditcard(random_state=None):
    """Load Credit Card Fraud Detection dataset."""
    data_path = None
    possible = [
        os.path.join(KAGGLE_DIR, "Credit Card Fraud Detection.csv"),
        os.path.join(KAGGLE_DIR, "creditcard.csv"),
        os.path.join(LOCAL_RAW, "creditcard.csv"),
    ]
    for p in possible:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print("  [SKIP] Credit Card: not found")
        return None

    print(f"  Loading Credit Card from {data_path}...")
    df = pd.read_csv(data_path)
    labels = df['Class'].values

    context = _standardize(df[['Time', 'Amount']].values.astype(float))
    v_cols = [f'V{i}' for i in range(1, 29)]
    behavior = _standardize(df[v_cols].values.astype(float))

    print(f"  Credit Card: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.4f}")
    return context, behavior, labels


# ============================================================================
# Dataset Registry
# ============================================================================

def get_all_datasets(seed):
    """Try loading all available datasets. Returns dict of name -> (ctx, beh, labels)."""
    datasets = {}

    print("\n=== Loading datasets ===")

    # ODDS datasets (always available or auto-download)
    for name, loader in [
        ("thyroid", load_thyroid),
        ("cardio", load_cardio),
        ("arrhythmia", load_arrhythmia),
        ("ionosphere", load_ionosphere),
        ("pima", load_pima),
        ("wbc", load_wbc),
        ("vowels", load_vowels),
    ]:
        try:
            result = loader()
            if result is not None:
                datasets[name] = result
        except Exception as e:
            print(f"  [ERROR] {name}: {str(e)[:80]}")

    # Fraud datasets (may not be available)
    for name, loader in [
        ("saml", lambda: load_saml(random_state=seed)),
        ("ieee_cis", lambda: load_ieee_cis(random_state=seed)),
        ("paysim", lambda: load_paysim(random_state=seed)),
        ("creditcard", lambda: load_creditcard(random_state=seed)),
    ]:
        try:
            result = loader()
            if result is not None:
                datasets[name] = result
        except Exception as e:
            print(f"  [ERROR] {name}: {str(e)[:80]}")

    print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets


# ============================================================================
# Train/test splitting
# ============================================================================

def train_test_split(context, behavior, labels, test_ratio=0.2, seed=42):
    """Split data into train (normal only) and test (normal + anomalies)."""
    rng = np.random.RandomState(seed)

    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    # Shuffle
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    # Split normal data
    n_test_normal = int(len(normal_idx) * test_ratio)
    test_normal_idx = normal_idx[:n_test_normal]
    train_idx = normal_idx[n_test_normal:]

    # Test set: some normal + all anomalies
    test_idx = np.concatenate([test_normal_idx, anomaly_idx])
    rng.shuffle(test_idx)

    return (
        (context[train_idx], behavior[train_idx], labels[train_idx]),
        (context[test_idx], behavior[test_idx], labels[test_idx]),
    )


# ============================================================================
# Model creation and running
# ============================================================================

def get_model(name, input_dim, context_dim):
    """Create model by name."""
    if name == "IF":
        return IsolationForestBaseline()
    if name == "IF-concat":
        return IsolationForestBaseline()
    if name == "DIF":
        return DeepIsolationForest(input_dim + context_dim)
    if name == "CVAE":
        return CVAE(input_dim, context_dim)
    if name == "CWAE-MMD-IF":
        return create_cwae_mmd_if(input_dim, context_dim)
    if name == "CWAE-MMD-CRPS":
        return create_cwae_mmd_crps(input_dim, context_dim)
    if name == "CWAE-MMD-Huber":
        return create_cwae_mmd_huber(input_dim, context_dim)
    if name == "CWAE-MMD-DIF-Dual":
        return create_cwae_mmd_dif_dual(input_dim, context_dim)
    if name == "CWAE-MMD-IF-Dual":
        return create_cwae_mmd_if_dual(input_dim, context_dim)
    if name == "CWAE-MMD-IF-Latent":
        return create_cwae_mmd_if_latent(input_dim, context_dim)
    raise ValueError(f"Unknown model: {name}")


def create_dataloader(behavior, context, labels, batch_size=256):
    """Create PyTorch DataLoader."""
    ds = TensorDataset(
        torch.tensor(behavior, dtype=torch.float32),
        torch.tensor(context, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def run_single_experiment(dataset_name, model_name, seed, train_data, test_data):
    """Run one experiment. Returns result dict."""
    ctx_train, beh_train, y_train = train_data
    ctx_test, beh_test, y_test = test_data

    input_dim = beh_train.shape[1]
    context_dim = ctx_train.shape[1]

    set_seed(seed)
    model = get_model(model_name, input_dim, context_dim)
    start = time.time()

    if model_name == "IF":
        model.fit(beh_train, None)
        scores = model.score(beh_test, None)
    elif model_name == "IF-concat":
        tf = np.concatenate([beh_train, ctx_train], axis=1)
        model.fit(tf, None)
        sf = np.concatenate([beh_test, ctx_test], axis=1)
        scores = model.score(sf, None)
    elif model_name == "DIF":
        tf = np.concatenate([beh_train, ctx_train], axis=1)
        model.fit(tf, None)
        sf = np.concatenate([beh_test, ctx_test], axis=1)
        scores = model.score(sf, None)
    else:
        # Neural network models: CVAE and all CWAE-MMD variants
        loader = create_dataloader(
            beh_train, ctx_train, y_train, TRAIN_CONFIG["batch_size"]
        )
        model, _ = train_model(
            model, loader,
            epochs=TRAIN_CONFIG["epochs"],
            patience=TRAIN_CONFIG["patience"],
            lr=TRAIN_CONFIG["lr"],
            verbose=False,
        )
        xt = torch.tensor(beh_train, dtype=torch.float32)
        ct = torch.tensor(ctx_train, dtype=torch.float32)
        model.fit_scorer(xt, ct)

        xte = torch.tensor(beh_test, dtype=torch.float32)
        cte = torch.tensor(ctx_test, dtype=torch.float32)
        scores = model.score(xte, cte)

    elapsed = time.time() - start
    metrics = compute_metrics(y_test, scores)

    return {
        "dataset": dataset_name,
        "method": model_name,
        "seed": seed,
        "n_samples": len(y_test) + len(y_train),
        "n_context": context_dim,
        "n_behavior": input_dim,
        "anomaly_rate": float(y_test.mean()),
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "nan_rate": metrics.nan_rate,
        "runtime_sec": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Checkpoint / incremental save
# ============================================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed": []}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f)


def append_result(result):
    df = pd.DataFrame([result])
    if os.path.exists(RAW_FILE):
        df.to_csv(RAW_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_FILE, index=False)


def append_error(error_info):
    df = pd.DataFrame([error_info])
    if os.path.exists(ERROR_FILE):
        df.to_csv(ERROR_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(ERROR_FILE, index=False)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("CWAE-MMD Real Dataset Comparison")
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Results: {RAW_FILE}")
    print("=" * 70)

    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])

    # Load datasets once (seed doesn't matter for loading, only for splitting)
    all_datasets = get_all_datasets(seed=42)

    if not all_datasets:
        print("\nERROR: No datasets found! Check data paths.")
        return

    total = len(all_datasets) * len(MODEL_NAMES) * len(SEEDS)
    remaining = total - sum(
        1 for ds in all_datasets for m in MODEL_NAMES for s in SEEDS
        if f"{ds}_{m}_{s}" in completed
    )
    print(f"\nTotal experiments: {total}")
    print(f"Already completed: {total - remaining}")
    print(f"Remaining: {remaining}")
    print()

    count = 0
    for dataset_name, (context, behavior, labels) in all_datasets.items():
        for seed in SEEDS:
            # Split data
            train_data, test_data = train_test_split(
                context, behavior, labels, test_ratio=0.2, seed=seed
            )

            for model_name in MODEL_NAMES:
                exp_key = f"{dataset_name}_{model_name}_{seed}"
                if exp_key in completed:
                    continue

                count += 1
                print(f"  [{count}/{remaining}] {dataset_name} | {model_name} | seed={seed}...",
                      end=" ", flush=True)

                try:
                    result = run_single_experiment(
                        dataset_name, model_name, seed, train_data, test_data
                    )
                    append_result(result)
                    checkpoint["completed"].append(exp_key)
                    save_checkpoint(checkpoint)
                    print(f"AUROC={result['auroc']:.4f} ({result['runtime_sec']:.1f}s)")

                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"ERROR: {str(e)[:60]}")

                    error_info = {
                        "dataset": dataset_name,
                        "method": model_name,
                        "seed": seed,
                        "error": str(e)[:200],
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_error(error_info)

                    # Still mark as completed to avoid retrying known failures
                    result = {
                        "dataset": dataset_name,
                        "method": model_name,
                        "seed": seed,
                        "n_samples": 0,
                        "n_context": 0,
                        "n_behavior": 0,
                        "anomaly_rate": 0,
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                        "nan_rate": 1.0,
                        "runtime_sec": 0,
                        "timestamp": datetime.now().isoformat(),
                    }
                    append_result(result)
                    checkpoint["completed"].append(exp_key)
                    save_checkpoint(checkpoint)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Results saved to: {RAW_FILE}")

    if os.path.exists(RAW_FILE):
        df = pd.read_csv(RAW_FILE)
        df_valid = df[df['auroc'].notna()]

        print(f"\nTotal experiments: {len(df)}")
        print(f"Valid results: {len(df_valid)}")

        # Summary table
        if len(df_valid) > 0:
            summary = df_valid.groupby(['dataset', 'method']).agg(
                auroc_mean=('auroc', 'mean'),
                auroc_std=('auroc', 'std'),
                auprc_mean=('auprc', 'mean'),
                n_runs=('auroc', 'count'),
            ).round(4)

            summary_file = os.path.join(RESULTS_DIR, "real_comparison_summary.csv")
            summary.to_csv(summary_file)
            print(f"Summary saved to: {summary_file}")

            # Print pivot table
            print("\n--- AUROC Mean by Dataset x Method ---")
            pivot = df_valid.pivot_table(
                values='auroc', index='dataset', columns='method', aggfunc='mean'
            ).round(4)
            print(pivot.to_string())


if __name__ == "__main__":
    main()
