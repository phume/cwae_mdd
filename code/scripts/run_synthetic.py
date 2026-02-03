#!/usr/bin/env python
"""Run full comparison batch - all datasets, all models, all seeds."""

import os
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import CWAEMMD
from src.models.variants import (
    create_cwae_mmd_if, create_cwae_mmd_dif,
    create_cwae_mmd_if_latent, create_cwae_mmd_if_dual, create_cwae_mmd_dif_dual,
    create_cwae_mmd_crps, create_cwae_mmd_huber, create_cwae_mmd_crps_if,
    create_cwae_mmd_imq_extreme
)
from src.models.baselines import CVAE, IsolationForestBaseline, DeepIsolationForest
from src.training import train_model, set_seed
from src.evaluation.metrics import compute_metrics
from src.data.synthetic import (
    generate_linear, generate_scale, generate_multimodal, generate_nonlinear,
    generate_skewed, create_train_test_split
)
from torch.utils.data import DataLoader, TensorDataset

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RAW_FILE = os.path.join(RESULTS_DIR, "full_comparison_raw.csv")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "full_comparison_checkpoint.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CONFIG = {"epochs": 50, "patience": 10, "lr": 1e-3, "batch_size": 256}
SEEDS = [42, 123, 456]

# Datasets
DATASETS = {
    "syn_linear": lambda s: generate_linear(n_samples=5000, seed=s),
    "syn_scale": lambda s: generate_scale(n_samples=5000, seed=s),
    "syn_multimodal": lambda s: generate_multimodal(n_samples=5000, seed=s),
    "syn_nonlinear": lambda s: generate_nonlinear(n_samples=5000, seed=s),
}

# Models
MODEL_NAMES = [
    "CWAE-MMD-IF", "CWAE-MMD-DIF", "CWAE-MMD-IF-Latent", "CWAE-MMD-IF-Dual",
    "CWAE-MMD-DIF-Dual", "CWAE-MMD-CRPS", "CWAE-MMD-Huber", "CWAE-MMD-CRPS-IF",
    "CWAE-MMD-IMQ-Extreme", "CWAEMMD-Base", "CVAE", "IF", "IF-concat", "DIF",
    "CWAE-MMD-Beta0.1", "CWAE-MMD-Beta10", "CWAE-MMD-Latent16"
]


def get_model(name, d, c):
    if name == "CWAE-MMD-IF": return create_cwae_mmd_if(d, c)
    if name == "CWAE-MMD-DIF": return create_cwae_mmd_dif(d, c)
    if name == "CWAE-MMD-IF-Latent": return create_cwae_mmd_if_latent(d, c)
    if name == "CWAE-MMD-IF-Dual": return create_cwae_mmd_if_dual(d, c)
    if name == "CWAE-MMD-DIF-Dual": return create_cwae_mmd_dif_dual(d, c)
    if name == "CWAE-MMD-CRPS": return create_cwae_mmd_crps(d, c)
    if name == "CWAE-MMD-Huber": return create_cwae_mmd_huber(d, c)
    if name == "CWAE-MMD-CRPS-IF": return create_cwae_mmd_crps_if(d, c)
    if name == "CWAE-MMD-IMQ-Extreme": return create_cwae_mmd_imq_extreme(d, c)
    if name == "CWAEMMD-Base": return CWAEMMD(d, c)
    if name == "CVAE": return CVAE(d, c)
    if name == "IF": return IsolationForestBaseline()
    if name == "IF-concat": return IsolationForestBaseline()
    if name == "DIF": return DeepIsolationForest(d + c)
    if name == "CWAE-MMD-Beta0.1": return CWAEMMD(d, c, mmd_weight=0.1)
    if name == "CWAE-MMD-Beta10": return CWAEMMD(d, c, mmd_weight=10.0)
    if name == "CWAE-MMD-Latent16": return CWAEMMD(d, c, latent_dim=16)
    raise ValueError(f"Unknown model: {name}")


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


def create_dl(data, bs=256):
    ds = TensorDataset(
        torch.tensor(data.x, dtype=torch.float32),
        torch.tensor(data.c, dtype=torch.float32),
        torch.tensor(data.y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=bs, shuffle=True)


def run_experiment(dataset_name, model_name, seed, train_data, test_data):
    input_dim = train_data.x.shape[1]
    context_dim = train_data.c.shape[1]

    set_seed(seed)
    model = get_model(model_name, input_dim, context_dim)
    start = time.time()

    if model_name in ["IF", "IF-concat"]:
        if model_name == "IF-concat":
            tf = np.concatenate([train_data.x, train_data.c], axis=1)
            model.fit(tf, None)
            sf = np.concatenate([test_data.x, test_data.c], axis=1)
            scores = model.score(sf, None)
        else:
            model.fit(train_data.x, train_data.c)
            scores = model.score(test_data.x, test_data.c)
    elif model_name == "DIF":
        tf = np.concatenate([train_data.x, train_data.c], axis=1)
        model.fit(tf, None)
        sf = np.concatenate([test_data.x, test_data.c], axis=1)
        scores = model.score(sf, None)
    else:
        loader = create_dl(train_data, TRAIN_CONFIG["batch_size"])
        model, _ = train_model(
            model, loader,
            epochs=TRAIN_CONFIG["epochs"],
            patience=TRAIN_CONFIG["patience"],
            lr=TRAIN_CONFIG["lr"],
            verbose=False
        )
        xt = torch.tensor(train_data.x, dtype=torch.float32)
        ct = torch.tensor(train_data.c, dtype=torch.float32)
        model.fit_scorer(xt, ct)
        xte = torch.tensor(test_data.x, dtype=torch.float32)
        cte = torch.tensor(test_data.c, dtype=torch.float32)
        scores = model.score(xte, cte)

    train_time = time.time() - start
    metrics = compute_metrics(test_data.y, scores)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "nan_rate": metrics.nan_rate,
        "train_time": train_time,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }


def main():
    print("=" * 60)
    print("CWAE-MMD Full Comparison")
    print("=" * 60)

    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed"])
    total = len(DATASETS) * len(MODEL_NAMES) * len(SEEDS)
    print(f"Total: {total}, Completed: {len(completed)}, Remaining: {total - len(completed)}")
    print()

    count = 0
    for dataset_name, gen_fn in DATASETS.items():
        for seed in SEEDS:
            print(f"\n--- {dataset_name} seed={seed} ---")

            # Generate data once per dataset/seed combo
            set_seed(seed)
            data = gen_fn(seed)
            train_data, test_data = create_train_test_split(data, test_ratio=0.2, seed=seed)

            for model_name in MODEL_NAMES:
                exp_key = f"{dataset_name}_{model_name}_{seed}"
                if exp_key in completed:
                    continue

                count += 1
                print(f"[{count}] {model_name}...", end=" ", flush=True)

                try:
                    result = run_experiment(dataset_name, model_name, seed, train_data, test_data)
                    append_result(result)
                    checkpoint["completed"].append(exp_key)
                    save_checkpoint(checkpoint)
                    print(f"AUROC={result['auroc']:.4f} ({result['train_time']:.1f}s)")
                except Exception as e:
                    print(f"ERROR: {str(e)[:50]}")
                    result = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                        "nan_rate": 1.0,
                        "train_time": 0,
                        "status": f"error: {str(e)[:80]}",
                        "timestamp": datetime.now().isoformat()
                    }
                    append_result(result)
                    checkpoint["completed"].append(exp_key)
                    save_checkpoint(checkpoint)

    print(f"\n\nCompleted {count} new experiments!")
    print(f"Total in checkpoint: {len(checkpoint['completed'])}")
    print(f"Results: {RAW_FILE}")


if __name__ == "__main__":
    main()
