#!/usr/bin/env python3
"""
Re-evaluate Fan-α and Fan-β checkpoints at all σ levels to obtain
robustness curves (which were not stored in stage3 main_results.csv).

Output: outputs/paper/fan_curves.csv  (mask_id, seed, dataset, sigma, ratio)
"""

import csv, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT = Path("outputs/paper")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G, HIDDEN = 8, 64
BATCH = 256
NOISE_SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
NOISE_REPS = 3
SEEDS = [0, 1, 2, 3, 4]
FAN_IDS = ["fan_fan_alpha", "fan_fan_beta",
           "dp_fan_alpha_0", "dp_fan_alpha_1", "dp_fan_alpha_2",
           "dp_fan_alpha_3", "dp_fan_alpha_4"]


class MaskedLinear(nn.Module):
    """Matches Stage 3 implementation: mask_gate shape (G, G, 1, 1) +
    permute(0, 2, 1, 3) inside forward."""
    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask_gate", mask.float().view(G, G, 1, 1))

    def forward(self, x):
        Gn = self.mask_gate.size(0)
        out_G = self.out_features // Gn
        in_G = self.in_features // Gn
        w = self.linear.weight
        w_g = w.view(Gn, out_G, Gn, in_G) * self.mask_gate
        w_g = w_g.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return F.linear(x, w_g, self.linear.bias)


class SparseMLP(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.proj = nn.Linear(784, HIDDEN)
        self.cells = nn.ModuleList([MaskedLinear(HIDDEN, HIDDEN, mask),
                                     MaskedLinear(HIDDEN, HIDDEN, mask)])
        self.head = nn.Linear(HIDDEN, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.proj(x))
        for cell in self.cells:
            x = F.relu(cell(x))
        return self.head(x)


def loaders(name):
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.view(-1))])
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    return DataLoader(cls("data", train=False, download=True, transform=t),
                       batch_size=BATCH)


@torch.no_grad()
def evaluate(model, loader, sigma=0.0, reps=1):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(DEVICE)
        for _ in range(max(1, reps)):
            xn = x + sigma * torch.randn_like(x)
            correct += (model(xn).argmax(1) == y.to(DEVICE)).sum().item()
            total += y.size(0)
    return correct / total


def main():
    rows = []
    for mid in FAN_IDS:
        for ds in ["mnist", "fashion_mnist"]:
            for seed in SEEDS:
                ckpt_path = Path(f"outputs/stage3/{ds}/{mid}/seed_{seed}/model.pt")
                if not ckpt_path.exists():
                    continue
                ck = torch.load(ckpt_path, weights_only=False)
                mask = ck["mask"].float()
                model = SparseMLP(mask).to(DEVICE)
                model.load_state_dict(ck["state_dict"])
                test_l = loaders(ds)
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                clean = evaluate(model, test_l, sigma=0.0, reps=1)
                row = {"mask_id": mid, "seed": seed, "dataset": ds,
                       "clean": clean}
                for s in NOISE_SIGMAS:
                    acc = evaluate(model, test_l, sigma=s,
                                   reps=NOISE_REPS if s > 0 else 1)
                    row[f"acc_{s:.2f}"] = acc
                rows.append(row)
                print(f"  {mid} {ds} seed={seed}: clean={clean:.4f}",
                      flush=True)

    keys = ["mask_id", "seed", "dataset", "clean"] + \
           [f"acc_{s:.2f}" for s in NOISE_SIGMAS]
    with open(OUT / "fan_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {OUT / 'fan_curves.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()