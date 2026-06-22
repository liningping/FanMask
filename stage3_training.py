#!/usr/bin/env python3
"""
Stage 3: Training and Evaluation
=================================
Train sparse MLPs with fan-derived and baseline masks on MNIST/Fashion-MNIST.
Evaluate robustness under Gaussian input noise.

Acceptance criteria (see EXPERIMENT_ROADMAP.md Stage 3):
    - 25 mask types x 2 datasets x 5 seeds = 250 training runs (minimum)
    - Checkpoint per run: model weights, mask, config, seed, clean/noisy accuracy
    - Naming: {dataset}_{model}_{mask_id}_seed{s}
    - Output: main_results.csv with Acc(0), Acc(sigma), AvgRatio

Models:
    - fan_alpha, fan_gamma, fan_beta (connectome-derived)
    - rs_0..rs_4 (Random-Sparse, 5 masks averaged per seed)
    - dp_fan_alpha_0..4, dp_fan_gamma_0..4, dp_fan_beta_0..4 (DP rewires)
    - dense (dense baseline, upper bound)

Noise levels: sigma in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
AvgRatio: mean of Acc(sigma)/Acc(0) for sigma in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR   = Path("outputs/stage3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_DIR = Path("outputs/stage1")
STAGE2_DIR = Path("outputs/stage2")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS    = 20
LR        = 1e-3
BATCH_SIZE = 256
SEEDS     = [0, 1, 2, 3, 4]
NOISE_SIGMAS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
AVG_RATIO_SIGMAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
NOISE_REPS = 3        # repetitions per sigma for noisy evaluation
G = 8
GROUP_SIZE = 8        # 64 / G = 8
IN_DIM = 784
HIDDEN = 64
OUT_DIM = 10
EDGES = 22            # must match stage1/stage2

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SparseMLP(nn.Module):
    """
    2-layer masked MLP.
    mask: (G, G) binary group-level mask expanded to (HIDDEN, HIDDEN).
    Each mask[i,j] = 1 opens the full (GROUP_SIZE x GROUP_SIZE) block
    connecting group i to group j.
    """
    def __init__(self, mask):
        super().__init__()
        self.proj = nn.Linear(IN_DIM, HIDDEN)
        self.mask = mask.float()  # (G, G) broadcast to (HIDDEN, HIDDEN)
        self.cells = nn.ModuleList([
            MaskedLinear(HIDDEN, HIDDEN, mask),
            MaskedLinear(HIDDEN, HIDDEN, mask),
        ])
        self.head = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        for cell in self.cells:
            x = F.relu(cell(x))
        return self.head(x)


class MaskedLinear(nn.Module):
    """
    Linear layer with a group-level binary mask.
    weight shape: (out_features, in_features) = (G*out_G, G*in_G)
    mask shape:   (G, G)
    The mask is Kronecker-expanded and applied as a multiplicative gate
    on the reshaped weight tensor (G, G, out_G, in_G).
    """
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        G = mask.size(0)
        in_G  = in_features  // G
        out_G = out_features // G
        # mask_gate: (G, G, 1, 1) broadcast over (G, G, out_G, in_G)
        self.register_buffer("mask_gate", mask.float().view(G, G, 1, 1))

    def forward(self, x):
        G = self.mask_gate.size(0)
        out_G = self.out_features // G
        in_G  = self.in_features  // G
        w = self.linear.weight
        # view as (G, out_G, G, in_G), apply mask, permute to (G, in_G, out_G, G), reshape
        w_g = w.view(G, out_G, G, in_G) * self.mask_gate
        w_g = w_g.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return F.linear(x, w_g, self.linear.bias)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def loaders(dataset_name):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    if dataset_name == "mnist":
        train = datasets.MNIST("data", train=True, download=True, transform=transform)
        test  = datasets.MNIST("data", train=False, download=True, transform=transform)
    else:
        train = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
        test  = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test, batch_size=BATCH_SIZE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, sigma=0.0, noise_reps=1):
    """Return accuracy. If sigma>0, add Gaussian noise to inputs noise_reps times and average."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        # Add noise
        for _ in range(max(1, noise_reps)):
            x_noisy = x + sigma * torch.randn_like(x)
            logits = model(x_noisy)
            correct += (logits.argmax(1) == y.to(device)).sum().item()
            total   += y.size(0)
    return correct / total


def train_model(mask_id, seed, dataset_name, mask, ckpt_dir):
    """Train one model. Returns dict with all metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SparseMLP(mask).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = loaders(dataset_name)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
        acc = evaluate(model, test_loader, DEVICE)
        best_acc = max(best_acc, acc)

    # Evaluate at all sigma levels
    results = {"mask_id": mask_id, "seed": seed, "dataset": dataset_name,
               "best_clean_acc": best_acc}

    for sigma in NOISE_SIGMAS:
        reps = NOISE_REPS if sigma > 0 else 1
        acc  = evaluate(model, test_loader, DEVICE, sigma=sigma, noise_reps=reps)
        results[f"acc_{sigma:.2f}"] = acc
        results[f"ratio_{sigma:.2f}"] = acc / best_acc if best_acc > 0 else 0

    # AvgRatio: mean ratio for sigma >= 0.20
    ratios = [results[f"ratio_{s:.2f}"] for s in AVG_RATIO_SIGMAS]
    results["avg_ratio"] = np.mean(ratios)

    # Save checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save({
        "mask_id": mask_id, "seed": seed, "dataset": dataset_name,
        "best_clean_acc": best_acc, "avg_ratio": results["avg_ratio"],
        "state_dict": model.state_dict(),
        "mask": mask,
    }, ckpt_path)

    return results


# ---------------------------------------------------------------------------
# Mask registry
# ---------------------------------------------------------------------------
def build_mask_registry():
    """Collect all masks with their model IDs."""
    registry = []

    # Fan masks (Stage 1)
    for fan_name, key in [("fan_alpha", "fan_alpha"),
                         ("fan_gamma", "fan_gamma"),
                         ("fan_beta",  "fan_beta")]:
        M = np.load(STAGE1_DIR / f"{key}_mask.npy")
        registry.append((f"fan_{key}", torch.from_numpy(M)))

    # Random-Sparse (Stage 2)
    rs_dir = STAGE2_DIR / "random_sparse_masks"
    for i in range(5):
        M = np.load(rs_dir / f"rs_{i}.npy")
        registry.append((f"rs_{i}", torch.from_numpy(M)))

    # DP rewires (Stage 2)
    for fan_key in ["fan_alpha", "fan_gamma", "fan_beta"]:
        fan_dir = STAGE2_DIR / fan_key
        for i in range(5):
            M = np.load(fan_dir / f"dp_{i}.npy")
            registry.append((f"dp_{fan_key}_{i}", torch.from_numpy(M)))

    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    registry = build_mask_registry()
    print(f"Total mask types: {len(registry)}")

    # Collect existing results from checkpoints
    all_results = []
    for mask_id, mask in registry:
        for seed in SEEDS:
            for dataset in ["mnist", "fashion_mnist"]:
                ckpt_dir = OUT_DIR / dataset / mask_id / f"seed_{seed}"
                ckpt_path = ckpt_dir / "model.pt"
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, weights_only=False)
                        all_results.append({
                            "mask_id": ckpt["mask_id"],
                            "seed": ckpt["seed"],
                            "dataset": ckpt["dataset"],
                            "best_clean_acc": ckpt["best_clean_acc"],
                            "avg_ratio": ckpt["avg_ratio"],
                        })
                    except Exception as e:
                        print(f"  Warning: could not load {ckpt_path}: {e}")
                    continue
                print(f"  Training {dataset}/{mask_id}/seed_{seed}...", end=" ")
                result = train_model(mask_id, seed, dataset, mask, ckpt_dir)
                all_results.append(result)
                print(f"clean={result['best_clean_acc']:.4f}, avg_ratio={result['avg_ratio']:.4f}")

    # Save main_results.csv
    if all_results:
        keys = list(all_results[0].keys())
        with open(OUT_DIR / "main_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nDone. Results saved to {OUT_DIR}/")
    print(f"  main_results.csv ({len(all_results)} rows)")


if __name__ == "__main__":
    main()