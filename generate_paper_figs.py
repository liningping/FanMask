#!/usr/bin/env python3
"""
Generate paper figures and tables (academic naming, no Random-Sparse).

Outputs:
    figs/fig1_method.png
    figs/fig2_motif_masks.png      — 3 DSS-Motifs + Fan-α + Fan-β
    figs/fig3_robustness_curves.png — 3 DSS-Motifs + Fan-α + Fan-β  (with 95% CI)
    figs/fig4_family_bar.png       — Fan-Motif / DegPres / DSS-Motif  (no RS)
    figs/table1_main.csv
    figs/table2_stats.csv          — DSS vs Fan, DSS vs DegPres
    figs/table3_ablation.csv       — internal iterations renamed Stage-A/B/C
    figs/table4_motifs.csv         — DSS-Motif-{1,2,3} details
"""

import csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

FIGS = Path("figs")
FIGS.mkdir(exist_ok=True)

# Internal directories keep legacy names; output naming is academic.
DSS_DIR = Path("outputs/meta_search_v3")          # Stage-C / DSS-Motif
ITER_B_DIR = Path("outputs/meta_search_v2")       # Stage-B (ablation only)
ITER_A_DIR = Path("outputs/meta_search")          # Stage-A (ablation only)
STAGE1_DIR = Path("outputs/stage1")               # Fan masks
STAGE2_DIR = Path("outputs/stage2")               # DegPres masks
STAGE3 = Path("outputs/stage3/main_results.csv")
P3 = Path("outputs/paper/p3_extended_seeds.csv")
P2_F = Path("outputs/paper/p2_fashion_v3.csv")

DSS_HUBS = [28, 45, 26]            # DSS-Motif-1, -2, -3
ITER_B_HUBS = [14, 27, 35]
ITER_A_HUBS = [12, 13, 36]

NOISE_SIGMAS = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

# Academic colour palette
COLORS = {
    "DSS-Motif-1": "#1f77b4",
    "DSS-Motif-2": "#17becf",
    "DSS-Motif-3": "#6a51a3",
    "DSS-Motif":   "#1f77b4",
    "Fan-α":       "#d62728",
    "Fan-β":       "#9c241c",
    "Fan-γ":       "#e9743b",
    "Fan-Motif":   "#d62728",
    "DegPres":     "#9467bd",
    "Stage-A":     "#ff7f0e",
    "Stage-B":     "#2ca02c",
    "Stage-C":     "#1f77b4",
}


# ----- helpers -----
def wilcoxon(diffs):
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    abs_d = [abs(d) for d in diffs]
    si = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[si[j + 1]] == abs_d[si[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[si[k]] = avg
        i = j + 1
    Wp = sum(r for r, d in zip(ranks, diffs) if d > 0)
    Wm = sum(r for r, d in zip(ranks, diffs) if d < 0)
    W = min(Wp, Wm)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    z = (W - mu) / math.sqrt(var) if var > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return W, p


def cohen_dz(diffs):
    arr = np.asarray(diffs)
    s = arr.std(ddof=1)
    return float(arr.mean() / s) if s > 0 else 0.0


# ----- data loaders -----
def load_v_seeds(src, log, hubs):
    by_seed = defaultdict(list)
    with open(src / log) as f:
        for r in csv.DictReader(f):
            if int(r["hub"]) in hubs:
                by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    return {s: float(np.mean(v)) for s, v in by_seed.items() if len(v) == len(hubs)}


def load_stage3_seeds(family_pred, dataset):
    by_seed = defaultdict(list)
    with open(STAGE3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and family_pred(r["mask_id"]):
                by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def dss_seeds_full(dataset):
    by_seed = defaultdict(list)
    if dataset == "mnist":
        with open(DSS_DIR / "validation_results_v3.csv") as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) in DSS_HUBS:
                    by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    else:
        with open(P2_F) as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) in DSS_HUBS:
                    by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    with open(P3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"].startswith("v3_hub"):
                h = int(r["mask_id"].replace("v3_hub", ""))
                if h in DSS_HUBS:
                    by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    return {s: float(np.mean(v)) for s, v in by_seed.items() if len(v) == 3}


# ----- Fig. 1 method overview -----
def fig1_method():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

    def panel(x, y, w, h, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     linewidth=1.4, edgecolor="black",
                                     facecolor=color, alpha=0.18))
    panel(0.2, 0.2, 4.0, 4.8, "#a6cee3")
    panel(4.6, 0.2, 4.0, 4.8, "#fdbf6f")
    panel(9.0, 0.2, 3.8, 4.8, "#b2df8a")

    ax.text(2.2, 4.75, "(a) Hub-anchored candidate pool",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(6.6, 4.75, "(b) Differentiable subgraph selection",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(10.9, 4.75, "(c) Discrete mask + sparse MLP",
            ha="center", fontsize=11, fontweight="bold")

    # (a) hub + neighbours
    centre = (2.2, 2.7)
    R = 1.4
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    neigh = [(centre[0] + R * np.cos(a), centre[1] + R * np.sin(a))
             for a in angles]
    for nx, ny in neigh:
        ax.plot([centre[0], nx], [centre[1], ny],
                color="#888", lw=0.8, alpha=0.55)
    ax.add_patch(plt.Circle(centre, 0.27, color="#e31a1c", zorder=3))
    ax.text(centre[0], centre[1], "hub", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    for (nx, ny) in neigh:
        ax.add_patch(plt.Circle((nx, ny), 0.16, color="#1f78b4"))
    ax.text(2.2, 0.65,
            "Connectome → one of 29 fan-in hubs\nhub + top-14 weighted neighbours\n→ 15-node candidate pool",
            ha="center", fontsize=9)

    # (b) selection
    cx0 = 4.85
    ax.text(6.6, 4.20, "Node logits (14)", ha="center", fontsize=9,
            fontweight="bold", color="#1f4e8a")
    for i in range(14):
        x = cx0 + 0.2 + i * 0.235
        h = 0.4 * (0.4 + np.cos(i * 1.2) * 0.3 + 1)
        ax.add_patch(plt.Rectangle((x, 3.55), 0.18, h,
                                     color="#1f78b4", alpha=0.6))
    ax.annotate("", xy=(6.6, 3.05), xytext=(6.6, 3.45),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
    ax.text(7.5, 3.25, "Gumbel-Sigmoid", fontsize=8, color="#444")

    ax.text(6.6, 2.85, "Edge logits (≈ 90)", ha="center", fontsize=9,
            fontweight="bold", color="#7a3a00")
    for i in range(15):
        x = cx0 + 0.15 + i * 0.22
        h = 0.4 * (0.3 + np.sin(i * 0.7) * 0.3 + 1)
        ax.add_patch(plt.Rectangle((x, 2.20), 0.18, h,
                                     color="#ff7f0e", alpha=0.6))
    ax.annotate("", xy=(6.6, 1.70), xytext=(6.6, 2.10),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))

    ax.text(6.6, 1.55, "SoftSort routing", ha="center", fontsize=9,
            fontweight="bold")
    soft_M = np.random.default_rng(2).random((8, 8)) * 0.5
    soft_M[0, :] += 0.5; soft_M[:, 0] += 0.4
    soft_M = np.clip(soft_M, 0, 1)
    ax.imshow(soft_M, extent=[5.85, 7.35, 0.5, 1.45],
              aspect="auto", cmap="Blues", origin="lower", alpha=0.85)
    ax.text(7.45, 0.95, "(8×8) soft\nmask", fontsize=8, va="center")
    ax.text(6.6, 0.30,
            r"$\mathcal{L} = 0.3\,\mathrm{CE}_{\mathrm{clean}} + 0.7\,\mathrm{CE}_{\mathrm{noisy}(\sigma)} + 0.1\,\mathcal{L}_{\mathrm{card}}$",
            ha="center", fontsize=9)

    # (c) discrete mask + MLP
    rng = np.random.default_rng(3)
    hard = np.zeros((8, 8))
    avail = [(i, j) for i in range(8) for j in range(8) if i != j]
    chosen = rng.choice(len(avail), 22, replace=False)
    for k in chosen:
        i, j = avail[k]; hard[i, j] = 1
    ax.imshow(hard, extent=[9.4, 10.5, 3.45, 4.55],
              aspect="auto", cmap="Greys", origin="lower")
    ax.text(9.95, 3.30, "Hard mask (8×8, 22 edges)",
            ha="center", fontsize=8)
    layer_x = [10.85, 11.4, 11.95, 12.5]
    layer_n = [4, 5, 5, 3]
    layer_lab = ["x", "h₁", "h₂", "y"]
    for lx, n, lab in zip(layer_x, layer_n, layer_lab):
        ys = np.linspace(2.4, 4.4, n)
        for y in ys:
            ax.add_patch(plt.Circle((lx, y), 0.07, color="#444"))
        ax.text(lx, 2.15, lab, ha="center", fontsize=8)
    for li in range(3):
        for y0 in np.linspace(2.4, 4.4, layer_n[li]):
            for y1 in np.linspace(2.4, 4.4, layer_n[li + 1]):
                ax.plot([layer_x[li], layer_x[li + 1]], [y0, y1],
                        color="#888", lw=0.3, alpha=0.6)
    ax.text(11.65, 1.85, "Sparse MLP forward + Gaussian noise",
            ha="center", fontsize=8)
    ax.text(10.9, 1.40, "Eval: avg_ratio over\n σ ∈ {0.20,…,0.50}",
            ha="center", fontsize=8.5, fontweight="bold", color="#0a3d0a")

    ax.annotate("", xy=(4.55, 2.7), xytext=(4.20, 2.7),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.annotate("", xy=(8.95, 2.7), xytext=(8.60, 2.7),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    plt.tight_layout()
    plt.savefig(FIGS / "fig1_method.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig1_method.png")


# ----- Fig. 2 motif masks (3 DSS + Fan-α + Fan-β) -----
def fig2_motifs():
    masks = [
        ("DSS-Motif-1\n(hub 28)", np.load(DSS_DIR / "hub28_hard_mask.npy")),
        ("DSS-Motif-2\n(hub 45)", np.load(DSS_DIR / "hub45_hard_mask.npy")),
        ("DSS-Motif-3\n(hub 26)", np.load(DSS_DIR / "hub26_hard_mask.npy")),
        ("Fan-α\n(hand-picked)",  np.load(STAGE1_DIR / "fan_alpha_mask.npy")),
        ("Fan-β\n(hand-picked)",  np.load(STAGE1_DIR / "fan_beta_mask.npy")),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0))
    for ax, (name, M) in zip(axes, masks):
        ax.imshow(M, cmap="binary", vmin=0, vmax=1, aspect="equal")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels(range(8), fontsize=7)
        ax.set_yticklabels(range(8), fontsize=7)
        ax.set_xlabel("input group", fontsize=8.5)
        ax.set_ylabel("output group", fontsize=8.5)
        for i in range(8):
            for j in range(8):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="lightgrey",
                                            linewidth=0.4))
    plt.tight_layout()
    plt.savefig(FIGS / "fig2_motif_masks.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig2_motif_masks.png")


# ----- Fig. 3 robustness curves: 3 DSS-Motifs + Fan-α + Fan-β -----
def per_mask_curve(mask_id_query, log_path, hub=None):
    """Return (sigmas, mean_ratio, std_ratio, n) by reading the validation log
    rows for the given hub or mask_id."""
    rows = []
    with open(log_path) as f:
        for r in csv.DictReader(f):
            if hub is not None and "hub" in r and int(r["hub"]) == hub:
                rows.append(r)
    xs, ms, sds = [], [], []
    for s in NOISE_SIGMAS:
        ratios = []
        for r in rows:
            clean = float(r["clean"])
            key = f"acc_{s:.2f}"
            if clean > 0 and key in r:
                ratios.append(float(r[key]) / clean)
        if ratios:
            xs.append(s); ms.append(np.mean(ratios))
            sds.append(np.std(ratios))
    return np.array(xs), np.array(ms), np.array(sds), len(rows)


def stage3_per_mask_curve(mask_id, dataset):
    """Read fan-curve evaluations from outputs/paper/fan_curves.csv (re-evaluated
    from saved Stage 3 checkpoints)."""
    fan_log = Path("outputs/paper/fan_curves.csv")
    rows = []
    if fan_log.exists():
        with open(fan_log) as f:
            for r in csv.DictReader(f):
                if r["dataset"] == dataset and r["mask_id"] == mask_id:
                    rows.append(r)
    xs, ms, sds = [], [], []
    for s in NOISE_SIGMAS:
        ratios = []
        for r in rows:
            clean = float(r["clean"])
            key = f"acc_{s:.2f}"
            if clean > 0 and key in r:
                ratios.append(float(r[key]) / clean)
        if ratios:
            xs.append(s); ms.append(np.mean(ratios)); sds.append(np.std(ratios))
    return np.array(xs), np.array(ms), np.array(sds), len(rows)


def degpres_curve(dataset):
    """Aggregate DegPres-α curve from 5 rewires × 5 seeds = 25 runs."""
    fan_log = Path("outputs/paper/fan_curves.csv")
    rows = []
    with open(fan_log) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"].startswith("dp_fan_alpha_"):
                rows.append(r)
    xs, ms, sds = [], [], []
    for s in NOISE_SIGMAS:
        ratios = []
        for r in rows:
            clean = float(r["clean"])
            key = f"acc_{s:.2f}"
            if clean > 0 and key in r:
                ratios.append(float(r[key]) / clean)
        if ratios:
            xs.append(s); ms.append(np.mean(ratios)); sds.append(np.std(ratios))
    return np.array(xs), np.array(ms), np.array(sds), len(rows)


def fig3_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    titles = ["MNIST", "Fashion-MNIST"]

    dss_style = [
        ("DSS-Motif-1", "o", "-", 2.0),
        ("DSS-Motif-2", "s", "-", 2.0),
        ("DSS-Motif-3", "D", "-", 2.0),
    ]

    for ax_idx, ds in enumerate(["mnist", "fashion_mnist"]):
        ax = axes[ax_idx]

        # 3 DSS-Motifs (solid)
        for (label, marker, ls, lw), hub in zip(dss_style, DSS_HUBS):
            log_path = (DSS_DIR / "validation_results_v3.csv"
                        if ds == "mnist" else P2_F)
            xs, ms, _, _ = per_mask_curve(None, log_path, hub=hub)
            ax.plot(xs, ms, marker=marker, linestyle=ls,
                     color=COLORS[label], label=label,
                     linewidth=lw, markersize=6)

        # Fan-α baseline (dashed, hollow triangle)
        xs, ms, _, _ = stage3_per_mask_curve("fan_fan_alpha", ds)
        ax.plot(xs, ms, marker="^", linestyle="--",
                 color=COLORS["Fan-α"], label="Fan-α (hand-picked)",
                 linewidth=2.0, markersize=9, markerfacecolor="white",
                 markeredgewidth=1.8)

        # DegPres baseline (dotted, hollow inverted-triangle)
        xs, ms, _, _ = degpres_curve(ds)
        ax.plot(xs, ms, marker="v", linestyle=":",
                 color=COLORS["DegPres"], label="DegPres (degree-preserved)",
                 linewidth=2.4, markersize=9, markerfacecolor="white",
                 markeredgewidth=1.8)

        ax.set_xlabel(r"Gaussian noise $\sigma$")
        ax.set_ylabel(r"$\mathrm{Acc}(\sigma) / \mathrm{Acc}(0)$")
        ax.set_title(titles[ax_idx])
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "fig3_robustness_curves.png", dpi=200,
                 bbox_inches="tight")
    plt.close()
    print("Wrote fig3_robustness_curves.png")


# ----- Fig. 4 family bar chart (Fan-Motif / DegPres / DSS-Motif) -----
def fig4_bar():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ds_titles = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST"}

    for ax_idx, ds in enumerate(["mnist", "fashion_mnist"]):
        ax = axes[ax_idx]
        fan = list(load_stage3_seeds(lambda m: m.startswith("fan_fan"), ds).values())
        dp  = list(load_stage3_seeds(lambda m: m.startswith("dp_fan"),  ds).values())
        dss = list(dss_seeds_full(ds).values())

        names = ["Fan-Motif\n(hand-picked)",
                 "DegPres\n(degree-preserved)",
                 "DSS-Motif\n(ours)"]
        data = [fan, dp, dss]
        means = [np.mean(d) for d in data]
        cis = [1.96 * np.std(d) / np.sqrt(len(d)) for d in data]
        cols = [COLORS["Fan-Motif"], COLORS["DegPres"], COLORS["DSS-Motif"]]
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=cis, capsize=4, color=cols,
                       edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
        ax.set_ylabel("avg_ratio")
        ymax = max(m + c for m, c in zip(means, cis))
        ymin = min(m - c for m, c in zip(means, cis))
        ax.set_ylim(ymin - 0.005, ymax + 0.015)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.0005,
                    f"{m:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(ds_titles[ds])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "fig4_family_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Wrote fig4_family_bar.png")


# ----- Tables -----
def metric_paths_len3(M):
    n = M.shape[0]; M_bin = (M > 0).astype(int); cnt = 0
    for s in range(n):
        for e in range(n):
            if s == e: continue
            stack = [(s, [s])]
            while stack:
                node, path = stack.pop()
                if len(path) - 1 == 3:
                    if node == e: cnt += 1
                    continue
                for nxt in np.where(M_bin[node])[0]:
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))
    return float(cnt)


def metric_eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0: return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def _load_individual_seeds(mask_id, dataset):
    """Load per-seed avg_ratio for a single mask_id from stage3."""
    result = {}
    with open(STAGE3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"] == mask_id:
                result[int(r["seed"])] = float(r["avg_ratio"])
    return result


def _load_dss_individual_seeds(hub, dataset):
    """Load per-seed avg_ratio for a single DSS hub."""
    result = {}
    if dataset == "mnist":
        with open(DSS_DIR / "validation_results_v3.csv") as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) == hub:
                    result[int(r["seed"])] = float(r["avg_ratio"])
    else:
        with open(P2_F) as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) == hub:
                    result[int(r["seed"])] = float(r["avg_ratio"])
    with open(P3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"] == f"v3_hub{hub}":
                result[int(r["seed"])] = float(r["avg_ratio"])
    return result


def _load_dp_mean_seeds(fan_key, dataset):
    """Load per-seed avg_ratio averaged over 5 DP rewires of a given fan."""
    by_seed = defaultdict(list)
    with open(STAGE3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"].startswith(f"dp_fan_{fan_key}_"):
                by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def _load_rs_mean_seeds(dataset):
    """Load per-seed avg_ratio averaged over 5 RS masks."""
    by_seed = defaultdict(list)
    with open(STAGE3) as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mask_id"].startswith("rs_"):
                by_seed[int(r["seed"])].append(float(r["avg_ratio"]))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


# Academic naming for Table 1
# - DSS-Motif-{1,2,3}: differentiable subgraph search discovered motifs
# - Fan-α/β/γ: hand-selected connectome fan-in structures
# - DP(Fan-α/β/γ): degree-preserved rewires (averaged over 5 rewires)
# - Random-Sparse: uniform random edge placement (averaged over 5 masks)
# - Dense: fully-connected hidden layer (upper bound)
TABLE1_STRUCTURES = [
    # (display_name, category, loader_args)
    ("DSS-Motif-1",       "DSS (ours)",       ("dss", 28)),
    ("DSS-Motif-2",       "DSS (ours)",       ("dss", 45)),
    ("DSS-Motif-3",       "DSS (ours)",       ("dss", 26)),
    ("Fan-α",             "Connectome fan",   ("stage3", "fan_fan_alpha")),
    ("Fan-β",             "Connectome fan",   ("stage3", "fan_fan_beta")),
    ("Fan-γ",             "Connectome fan",   ("stage3", "fan_fan_gamma")),
    ("DP(Fan-α)",         "Degree-preserved", ("dp", "alpha")),
    ("DP(Fan-β)",         "Degree-preserved", ("dp", "beta")),
    ("DP(Fan-γ)",         "Degree-preserved", ("dp", "gamma")),
    ("Random-Sparse",     "Random baseline",  ("rs",)),
]


def table1_main():
    """Per-structure AvgRatio with paired Wilcoxon test vs best DSS-Motif."""
    fieldnames = ["structure", "category", "dataset", "n_seeds",
                  "avg_ratio_mean", "avg_ratio_std",
                  "delta_vs_best_DSS", "p_vs_best_DSS", "cohen_dz"]
    rows = []

    for ds in ["mnist", "fashion_mnist"]:
        # Load all structures
        struct_seeds = {}
        for name, cat, args in TABLE1_STRUCTURES:
            if args[0] == "dss":
                seeds = _load_dss_individual_seeds(args[1], ds)
            elif args[0] == "stage3":
                seeds = _load_individual_seeds(args[1], ds)
            elif args[0] == "dp":
                seeds = _load_dp_mean_seeds(args[1], ds)
            elif args[0] == "rs":
                seeds = _load_rs_mean_seeds(ds)
            else:
                seeds = {}
            struct_seeds[name] = seeds

        # Best DSS reference: per-seed max across 3 DSS motifs
        dss_names = [n for n, c, _ in TABLE1_STRUCTURES if c == "DSS (ours)"]
        dss_all_seeds = set()
        for dn in dss_names:
            dss_all_seeds |= set(struct_seeds[dn].keys())
        best_dss = {}
        for s in dss_all_seeds:
            vals = [struct_seeds[dn][s] for dn in dss_names if s in struct_seeds[dn]]
            if vals:
                best_dss[s] = max(vals)

        for name, cat, _ in TABLE1_STRUCTURES:
            seeds = struct_seeds[name]
            if not seeds:
                continue
            vals = list(seeds.values())
            # Paired test vs best DSS
            common = sorted(set(seeds) & set(best_dss))
            if common and cat != "DSS (ours)":
                diffs = [seeds[s] - best_dss[s] for s in common]
                W, p = wilcoxon(diffs)
                delta = round(np.mean(diffs), 4)
                p_val = round(p, 4)
                dz = round(cohen_dz(diffs), 3)
            else:
                delta, p_val, dz = "—", "—", "—"

            rows.append({
                "structure": name,
                "category": cat,
                "dataset": ds,
                "n_seeds": len(vals),
                "avg_ratio_mean": round(np.mean(vals), 4),
                "avg_ratio_std": round(np.std(vals), 4),
                "delta_vs_best_DSS": delta,
                "p_vs_best_DSS": p_val,
                "cohen_dz": dz,
            })

    with open(FIGS / "table1_main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print("Wrote table1_main.csv (per-structure, with paired stats vs best DSS)")


def table2_stats():
    rows = []
    for ds in ["mnist", "fashion_mnist"]:
        dss = dss_seeds_full(ds)
        for name, fam in [
            ("DSS-Motif vs Fan-Motif",
             load_stage3_seeds(lambda m: m.startswith("fan_fan"), ds)),
            ("DSS-Motif vs DegPres",
             load_stage3_seeds(lambda m: m.startswith("dp_fan"),  ds)),
        ]:
            common = sorted(set(dss) & set(fam))
            diffs = [dss[s] - fam[s] for s in common]
            W, p = wilcoxon(diffs)
            rows.append({"comparison": name, "dataset": ds,
                         "n_seeds": len(common),
                         "mean_diff": round(np.mean(diffs), 4),
                         "W": round(W, 1),
                         "p_two_sided": round(p, 4),
                         "cohen_dz": round(cohen_dz(diffs), 3),
                         "significant_05": p < 0.05})
    with open(FIGS / "table2_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote table2_stats.csv")


def table3_ablation():
    """Method ablation: Stage-A → Stage-B → Stage-C (renamed v1 / v2 / v3).

    Stage-A: hard top-K selection + CE@σ=0.3
    Stage-B: + differentiable Gumbel-Sigmoid + SoftSort routing
    Stage-C: + multi-σ robustness loss (full DSS).
    All on same 5 seeds (0..4), MNIST.
    """
    a = load_v_seeds(ITER_A_DIR, "validation_results.csv", set(ITER_A_HUBS))
    b = load_v_seeds(ITER_B_DIR, "validation_results_v2.csv", set(ITER_B_HUBS))
    c = {s: v for s, v in dss_seeds_full("mnist").items() if s < 5}
    dp = load_stage3_seeds(lambda m: m.startswith("dp_fan"), "mnist")

    rows = []
    for name, fam in [("Stage-A: hard top-K",                    a),
                      ("Stage-B: + Gumbel + SoftSort",           b),
                      ("Stage-C: + multi-σ loss (full DSS)",     c)]:
        common = sorted(set(fam) & set(dp))
        diffs = [fam[s] - dp[s] for s in common]
        W, p = wilcoxon(diffs)
        rows.append({
            "stage": name,
            "n_seeds": len(common),
            "mean_avg_ratio": round(np.mean([fam[s] for s in common]), 4),
            "std_avg_ratio":  round(np.std([fam[s] for s in common]), 4),
            "vs_DegPres_mean_diff": round(np.mean(diffs), 4),
            "vs_DegPres_p": round(p, 4),
            "vs_DegPres_cohen_dz": round(cohen_dz(diffs), 3),
        })
    with open(FIGS / "table3_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote table3_ablation.csv")


def table4_motifs():
    nodes_map = {}
    with open(DSS_DIR / "search_log_v3.csv") as f:
        for r in csv.DictReader(f):
            nodes_map[int(r["hub"])] = r["selected_nodes"]

    rows = []
    for k, h in enumerate(DSS_HUBS):
        M = np.load(DSS_DIR / f"hub{h}_hard_mask.npy")
        mn = []; fm = []
        with open(DSS_DIR / "validation_results_v3.csv") as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) == h: mn.append(float(r["avg_ratio"]))
        with open(P2_F) as f:
            for r in csv.DictReader(f):
                if int(r["hub"]) == h: fm.append(float(r["avg_ratio"]))
        rows.append({
            "motif_id":     f"DSS-Motif-{k+1}",
            "hub_node":     h,
            "selected_nodes": nodes_map.get(h, ""),
            "mnist_avg_ratio_mean":   round(np.mean(mn), 4),
            "mnist_avg_ratio_std":    round(np.std(mn), 4),
            "fashion_avg_ratio_mean": round(np.mean(fm), 4),
            "fashion_avg_ratio_std":  round(np.std(fm), 4),
            "edges":        int(M.sum()),
            "max_in_deg":   int(M.sum(axis=0).max()),
            "max_out_deg":  int(M.sum(axis=1).max()),
            "paths_len3":   int(metric_paths_len3(M)),
            "eff_rank":     round(metric_eff_rank(M), 3),
        })
    with open(FIGS / "table4_motifs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Wrote table4_motifs.csv")


# ----- main -----
if __name__ == "__main__":
    fig1_method()
    fig2_motifs()
    fig3_curves()
    fig4_bar()
    table1_main()
    table3_ablation()
    table4_motifs()
    print("\nAll figures and tables saved under figs/")
