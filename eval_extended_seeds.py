#!/usr/bin/env python3
"""
P3: Extended seed runs (seeds 5..9) for v3 top-3 + RS family on MNIST + Fashion-MNIST.
Adds 80 runs total: (3 v3 + 5 RS) * 5 new seeds * 2 datasets.
"""

import csv, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT_DIR = Path("outputs/paper")
V3_DIR = Path("outputs/meta_search_v3")
STAGE2_DIR = Path("outputs/stage2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G, HIDDEN = 8, 64
EPOCHS, LR, BATCH_SIZE = 20, 1e-3, 256
NEW_SEEDS = [5, 6, 7, 8, 9]
NOISE_SIGMAS = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
AVG_RATIO_SIGMAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
NOISE_REPS = 3

V3_HUBS = [28, 45, 26]
RS_IDS = [0, 1, 2, 3, 4]


class MaskedLinear(nn.Module):
    def __init__(self, in_f, out_f, mask):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.linear = nn.Linear(in_f, out_f)
        self.register_buffer("mask_gate", mask.float().view(G, 1, G, 1))
    def forward(self, x):
        oG, iG = self.out_f // G, self.in_f // G
        w = self.linear.weight
        w_g = w.view(G, oG, G, iG) * self.mask_gate
        return F.linear(x, w_g.view(self.out_f, self.in_f), self.linear.bias)


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
    return (DataLoader(cls("data", train=True, download=True, transform=t),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(cls("data", train=False, download=True, transform=t),
                       batch_size=BATCH_SIZE))


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


def train_one(mask_t, seed, dataset):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SparseMLP(mask_t).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_l, test_l = loaders(dataset)
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        best = max(best, evaluate(model, test_l))
    r = {"clean": best}
    for s in NOISE_SIGMAS:
        acc = evaluate(model, test_l, sigma=s, reps=NOISE_REPS if s > 0 else 1)
        r[f"acc_{s:.2f}"] = acc
    r["avg_ratio"] = float(np.mean([r[f"acc_{s:.2f}"] / best for s in AVG_RATIO_SIGMAS]))
    return r


def main():
    log = OUT_DIR / "p3_extended_seeds.csv"
    rows = []
    if log.exists():
        with open(log) as f:
            rows = list(csv.DictReader(f))
    done = {(r["mask_id"], int(r["seed"]), r["dataset"]) for r in rows}

    plan = []
    for hub in V3_HUBS:
        for seed in NEW_SEEDS:
            for ds in ["mnist", "fashion_mnist"]:
                plan.append((f"v3_hub{hub}", hub, seed, ds, "v3"))
    for rsi in RS_IDS:
        for seed in NEW_SEEDS:
            for ds in ["mnist", "fashion_mnist"]:
                plan.append((f"rs_{rsi}", rsi, seed, ds, "rs"))

    todo = [p for p in plan if (p[0], p[2], p[3]) not in done]
    print(f"Total runs: {len(plan)}, to do: {len(todo)}", flush=True)

    for mask_id, key, seed, ds, kind in todo:
        if kind == "v3":
            M = np.load(V3_DIR / f"hub{key}_hard_mask.npy")
        else:
            M = np.load(STAGE2_DIR / "random_sparse_masks" / f"rs_{key}.npy")
        mask_t = torch.from_numpy(M)
        print(f"  {mask_id} seed={seed} {ds}...", end=" ", flush=True)
        r = train_one(mask_t, seed, ds)
        row = {"mask_id": mask_id, "seed": seed, "dataset": ds,
               "clean": r["clean"], "avg_ratio": r["avg_ratio"]}
        for s in NOISE_SIGMAS:
            row[f"acc_{s:.2f}"] = r[f"acc_{s:.2f}"]
        rows.append(row)
        print(f"clean={r['clean']:.4f} ar={r['avg_ratio']:.4f}", flush=True)
        with open(log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()