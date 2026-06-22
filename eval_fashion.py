#!/usr/bin/env python3
"""
P2.1 Fashion-MNIST validation of v3 top-3 motifs (hubs 28, 45, 26).
3 hubs x 5 seeds = 15 runs.
"""
import csv, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT_DIR = Path("outputs/paper")
V3_DIR = Path("outputs/meta_search_v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G, HIDDEN = 8, 64
EPOCHS, LR, BATCH_SIZE = 20, 1e-3, 256
SEEDS = [0, 1, 2, 3, 4]
NOISE_SIGMAS = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
AVG_RATIO_SIGMAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
NOISE_REPS = 3
V3_HUBS = [28, 45, 26]


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
    return (DataLoader(datasets.FashionMNIST("data", train=True, download=True, transform=t),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(datasets.FashionMNIST("data", train=False, download=True, transform=t),
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


def train_one(mask_t, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SparseMLP(mask_t).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_l, test_l = loaders()
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
    avg_ratio = np.mean([r[f"acc_{s:.2f}"] / best for s in AVG_RATIO_SIGMAS])
    r["avg_ratio"] = float(avg_ratio)
    return r


def main():
    log = OUT_DIR / "p2_fashion_v3.csv"
    rows = []
    if log.exists():
        with open(log) as f:
            for r in csv.DictReader(f): rows.append(r)
    done = {(int(r["hub"]), int(r["seed"])) for r in rows}
    for hub in V3_HUBS:
        M = torch.from_numpy(np.load(V3_DIR / f"hub{hub}_hard_mask.npy"))
        for seed in SEEDS:
            if (hub, seed) in done:
                continue
            print(f"  hub={hub} seed={seed}...", end=" ", flush=True)
            r = train_one(M, seed)
            row = {"hub": hub, "seed": seed, "clean": r["clean"],
                   "avg_ratio": r["avg_ratio"]}
            for s in NOISE_SIGMAS:
                row[f"acc_{s:.2f}"] = r[f"acc_{s:.2f}"]
            rows.append(row)
            print(f"clean={r['clean']:.4f}, ar={r['avg_ratio']:.4f}", flush=True)
            with open(log, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()