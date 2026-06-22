#!/usr/bin/env python3
"""Validate v2 hard masks: same protocol as meta_step4_validate.py but for v2 masks."""

import csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT_DIR = Path("outputs/meta_search_v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G = 8
HIDDEN = 64
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 256
SEEDS = [0, 1, 2, 3, 4]
NOISE_SIGMAS = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
AVG_RATIO_SIGMAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
NOISE_REPS = 3


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


def loaders():
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.view(-1))])
    return (DataLoader(datasets.MNIST("data", train=True, download=True, transform=t),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(datasets.MNIST("data", train=False, download=True, transform=t),
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


def train_one(mask_tensor, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SparseMLP(mask_tensor).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_l, test_l = loaders()
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        best = max(best, evaluate(model, test_l))
    r = {"clean": best}
    for s in NOISE_SIGMAS:
        acc = evaluate(model, test_l, sigma=s, reps=NOISE_REPS if s > 0 else 1)
        r[f"acc_{s:.2f}"] = acc
    avg_ratio = np.mean([r[f"acc_{s:.2f}"] / best
                         for s in AVG_RATIO_SIGMAS]) if best > 0 else 0
    r["avg_ratio"] = float(avg_ratio)
    return r


def main():
    mask_files = sorted(OUT_DIR.glob("hub*_hard_mask.npy"),
                        key=lambda p: int(p.stem.replace("hub", "").replace("_hard_mask", "")))
    print(f"Validating {len(mask_files)} v2 masks", flush=True)

    log_path = OUT_DIR / "validation_results_v3.csv"
    rows, done = [], set()
    if log_path.exists():
        with open(log_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((int(r["hub"]), int(r["seed"])))
        print(f"Resume: {len(rows)} runs already done", flush=True)

    for mp in mask_files:
        hub = int(mp.stem.replace("hub", "").replace("_hard_mask", ""))
        M = np.load(mp)
        edges = int(M.sum())
        if edges < 20:
            continue
        mask_t = torch.from_numpy(M)
        for seed in SEEDS:
            if (hub, seed) in done:
                continue
            print(f"  hub={hub:>2} seed={seed} ...", end=" ", flush=True)
            r = train_one(mask_t, seed)
            row = {"hub": hub, "seed": seed, "edges": edges,
                   "clean": r["clean"], "avg_ratio": r["avg_ratio"]}
            for s in NOISE_SIGMAS:
                row[f"acc_{s:.2f}"] = r[f"acc_{s:.2f}"]
            rows.append(row)
            print(f"clean={r['clean']:.4f}, avg_ratio={r['avg_ratio']:.4f}", flush=True)
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"Wrote {log_path} ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    main()