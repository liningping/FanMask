#!/usr/bin/env python3
"""
Stage 1: Fan-in Structure Discovery
==================================
Extract Fan-α, Fan-γ, Fan-β from zebrafish connectome using fixed structural rules.
Outputs masks and structural statistics before any ANN training.

Acceptance criteria (see EXPERIMENT_ROADMAP.md Stage 1):
    - Self-loops removed, weights normalized, binary graph via fixed threshold
    - Fan candidates enumerated with hub in≥4, out≥2, 8 nodes, 16 edges
    - Fan-α/γ/β selected by pre-registered structural rules (no performance filtering)
    - Outputs: all_fan_candidates.csv, fan_alpha/gamma/beta_mask.npy, fan_candidate_summary.csv
"""

import csv
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (must be fixed before ANN training - do not tune against results)
# ---------------------------------------------------------------------------
CONNECTOME_PATH = "data/conn_matrix_complete.npy"
OUT_DIR = Path("outputs/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

G = 8              # group count for group-level mask
EDGES = 22         # edges per mask (increased from 16 to preserve structural diversity)
TOP_IN = 4         # top incoming neighbors for hub
TOP_OUT = 3        # top outgoing neighbors for hub
HUB_MIN_IN = 4    # minimum incoming degree for hub candidate
HUB_MIN_OUT = 2   # minimum outgoing degree for hub candidate
Q = 0.25           # top-q fraction for threshold (top 25% edges)
MAX_JACCARD = 0.25       # max Jaccard overlap between fans
LAMBDA = 0.5             # fan purity penalty weight
MU = 0.2                 # fan purity connectome weight

# ---------------------------------------------------------------------------
# 1. Load and preprocess connectome
# ---------------------------------------------------------------------------
A = np.load(CONNECTOME_PATH)
print(f"Connectome loaded: shape {A.shape}")

# 1a. Remove self-loops
A_no_self = A.copy()
np.fill_diagonal(A_no_self, 0)

# 1b. Min-max normalize non-zero edge weights
mask_nonzero = A_no_self != 0
A_norm = np.zeros_like(A_no_self)
v_min = A_no_self[mask_nonzero].min()
v_max = A_no_self[mask_nonzero].max()
A_norm[mask_nonzero] = (A_no_self[mask_nonzero] - v_min) / (v_max - v_min)

# 1c. Fixed threshold -> binary graph (top-q fraction of all possible edges)
# Threshold is determined by connectome distribution only, not by ANN results.
n = A_norm.shape[0]
total_possible = n * (n - 1)  # directed, no self-loop
q = int(Q * total_possible)
non_zero_vals = A_norm[mask_nonzero]
# Find the q-th largest non-zero value as threshold (top-q edges)
kth = len(non_zero_vals) - q
threshold = np.partition(non_zero_vals, kth)[kth]
B = ((A_norm >= threshold) & (A_norm > 0)).astype(float)
np.fill_diagonal(B, 0)
print(f"Binary graph: threshold={threshold:.4f}, edge count={int(B.sum())}")

# ---------------------------------------------------------------------------
# 2. Fan candidate enumeration
# ---------------------------------------------------------------------------
def compute_structural_metrics(B, nodes, edge_weights):
    """Compute in/out degree, peripheral edges, clustering, fan purity."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    M = np.zeros((n, n))
    for (u, w), weight in zip(edge_weights, [A_norm[u, w] for (u, w) in edge_weights]):
        if u in idx and w in idx:
            M[idx[u], idx[w]] = 1

    hub_idx = 0
    d_in = M[:, hub_idx].sum()
    d_out = M[hub_idx, :].sum()

    # Peripheral edges: non-hub to non-hub
    E_periph = M[1:, 1:].sum()

    # Undirected average clustering coefficient
    M_und = (M + M.T) > 0
    C = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if M_und[i, j]:
                neighbors_i = set(np.where(M_und[i])[0]) - {i}
                neighbors_j = set(np.where(M_und[j])[0]) - {j}
                common = len(neighbors_i & neighbors_j)
                if common > 0:
                    C += common / (len(neighbors_i) * len(neighbors_j)) ** 0.5
                    count += 1
    C = C / count if count > 0 else 0

    # Fan purity score
    total_weight = sum(A_norm[u, w] for (u, w) in edge_weights if u in idx and w in idx)
    hub_weight = sum(A_norm[nodes[hub_idx], w] for w in nodes[1:] if A_norm[nodes[hub_idx], w] > 0)
    P_fan = (d_in + d_out) / EDGES - LAMBDA * E_periph / EDGES + MU * hub_weight / (total_weight + 1e-9)

    return {
        "d_in_hub": int(d_in),
        "d_out_hub": int(d_out),
        "E_periph": int(E_periph),
        "C": C,
        "P_fan": P_fan,
    }

candidates = []

for hub in range(n):
    # Skip hubs that don't meet minimum degree requirements
    in_neighbors = np.where(B[:, hub] > 0)[0]
    out_neighbors = np.where(B[hub, :] > 0)[0]
    if len(in_neighbors) < HUB_MIN_IN or len(out_neighbors) < HUB_MIN_OUT:
        continue

    # Select top incoming and outgoing by weight
    in_weights = [(u, A_norm[u, hub]) for u in in_neighbors]
    out_weights = [(hub, w, A_norm[hub, w]) for w in out_neighbors]
    in_weights.sort(key=lambda x: x[1], reverse=True)
    out_weights.sort(key=lambda x: x[2], reverse=True)

    top_in = [x[0] for x in in_weights[:TOP_IN]]
    top_out = [x[1] for x in out_weights[:TOP_OUT]]

    nodes = [hub] + top_in + top_out
    nodes = list(dict.fromkeys(nodes))  # deduplicate preserving order

    # Expand to G nodes if needed
    if len(nodes) < G:
        all_candidates = set(in_neighbors) | set(out_neighbors)
        all_candidates -= set(nodes)
        sorted_extra = sorted(all_candidates, key=lambda x: sum(A_norm[x, y] for y in nodes) + sum(A_norm[y, x] for y in nodes), reverse=True)
        for nd in sorted_extra:
            if len(nodes) >= G:
                break
            nodes.append(nd)

    # Prune to exactly G nodes
    nodes = nodes[:G]

    # Build edge list from induced subgraph
    edge_list = []
    for u in nodes:
        for w in nodes:
            if u != w and A_norm[u, w] > 0:
                edge_list.append((u, w))

    # Prune to EDGES edges (prioritize hub edges)
    def edge_priority(e):
        u, w = e
        if u == hub:
            return 2
        if w == hub:
            return 1
        return 0

    edge_list.sort(key=edge_priority, reverse=True)
    edge_list = edge_list[:EDGES]

    # Verify constraints
    hub_in = sum(1 for (u, w) in edge_list if w == hub)
    hub_out = sum(1 for (u, w) in edge_list if u == hub)
    if hub_in < HUB_MIN_IN or hub_out < HUB_MIN_OUT:
        continue

    # Build 8x8 mask
    node_idx = {v: i for i, v in enumerate(nodes)}
    M = np.zeros((G, G), dtype=np.float32)
    for (u, w) in edge_list:
        M[node_idx[u], node_idx[w]] = 1.0

    metrics = compute_structural_metrics(B, nodes, edge_list)
    candidates.append({
        "hub": hub,
        "nodes": nodes,
        "edge_list": edge_list,
        "mask": M,
        **metrics,
    })

print(f"Enumerated {len(candidates)} valid fan candidates")

# ---------------------------------------------------------------------------
# 3. Pre-register Fan-α, Fan-γ, Fan-β selection (multi-dimensional diversity)
#
# We enforce structural diversity across THREE pre-registered dimensions:
#   1. Clustering coefficient C    → low / mid / high bins
#   2. Hub in-degree              → low / mid / high bins
#   3. Peripheral edge count      → ensures peripheral structure differs
#
# All quantile boundaries are computed from the FULL candidate list before
# inspecting individual candidate values (pre-registered, no cherry-picking).
# ---------------------------------------------------------------------------
def jaccard_overlap(a, b):
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b)

# ---- 3a. Pre-register quantile thresholds ----
# Pre-registered percentiles for each dimension
PCTL = {"C": 33, "d_in": 33, "E_periph": 33}   # low boundary
PCTH = {"C": 67, "d_in": 67, "E_periph": 67}   # high boundary

def pct_thresholds(arr, dim):
    """Return (low_thr, high_thr) at pre-registered percentiles."""
    lo_idx = int(np.ceil(len(arr) * PCTL[dim] / 100)) - 1
    hi_idx = int(np.ceil(len(arr) * PCTH[dim] / 100)) - 1
    hi_idx = min(hi_idx, len(arr) - 1)
    lo_val = sorted(arr)[lo_idx]
    hi_val = sorted(arr)[hi_idx]
    return lo_val, hi_val

all_C     = [c["C"] for c in candidates]
all_din   = [c["d_in_hub"] for c in candidates]
all_periph = [c["E_periph"] for c in candidates]

C_lo, C_hi       = pct_thresholds(all_C, "C")
din_lo, din_hi   = pct_thresholds(all_din, "d_in")
periph_lo, periph_hi = pct_thresholds(all_periph, "E_periph")

print(f"\nQuantile thresholds (n={len(candidates)}):")
print(f"  C:       low<={C_lo:.4f},  high>={C_hi:.4f}")
print(f"  d_in:    low<={din_lo},    high>={din_hi}")
print(f"  E_periph: low<={periph_lo}, high>={periph_hi}")

# ---- 3b. Build selection pools ----
# Fan-α: low-C, low-in-degree (pure fan-in, minimal clustering / interference)
alpha_pool = [c for c in candidates
              if c["C"] <= C_lo and c["d_in_hub"] <= din_lo
              and c["E_periph"] >= periph_lo]

# Fan-β: high-C, mid/high-in-degree (clustered, retains fan-in strength)
beta_pool  = [c for c in candidates
              if c["C"] >= C_hi and c["d_in_hub"] >= din_lo
              and c["E_periph"] <= periph_hi]

# Fan-γ: mid-C, mid-in-degree (structural interpolation between α and β)
gamma_pool = [c for c in candidates
              if c["C"] > C_lo and c["C"] < C_hi
              and c["d_in_hub"] > din_lo and c["d_in_hub"] < din_hi
              and c["E_periph"] > periph_lo and c["E_periph"] < periph_hi]

print(f"\nPool sizes before Jaccard filter: α={len(alpha_pool)}, γ={len(gamma_pool)}, β={len(beta_pool)}")

# ---- 3c. Jaccard deduplication ----
def jaccard_filter(pool, exclude_list, max_j=MAX_JACCARD):
    return [c for c in pool
            if all(jaccard_overlap(c["nodes"], e["nodes"]) <= max_j for e in exclude_list)]

def pick_best(pool):
    if not pool: return None
    return max(pool, key=lambda x: x["P_fan"])

# Select α first (low-C, low-d_in is the most constrained dimension)
fan_alpha = pick_best(alpha_pool)
print(f"  Fan-α pool: {len(alpha_pool)}, selected hub={fan_alpha['hub'] if fan_alpha else 'None'}")

# Select β from high-C pool, excluding α
fan_beta  = pick_best(jaccard_filter(beta_pool, [fan_alpha] if fan_alpha else []))
print(f"  Fan-β pool: {len(beta_pool)}, selected hub={fan_beta['hub'] if fan_beta else 'None'}")

# Select γ from mid pool, excluding both α and β (with relaxed Jaccard)
exclude = [e for e in [fan_alpha, fan_beta] if e]
fan_gamma = pick_best(jaccard_filter(gamma_pool, exclude))
if not fan_gamma:
    fan_gamma = pick_best(jaccard_filter(gamma_pool, exclude, max_j=0.50))
if not fan_gamma:
    # Fallback: expand gamma pool to all candidates not yet selected
    remaining = [c for c in candidates if c not in [fan_alpha, fan_beta]]
    fan_gamma = pick_best(jaccard_filter(remaining, exclude, max_j=0.50))
print(f"  Fan-γ pool: {len(gamma_pool)}, selected hub={fan_gamma['hub'] if fan_gamma else 'None'}")

selected = {"Fan-α": fan_alpha, "Fan-γ": fan_gamma, "Fan-β": fan_beta}
selected = {k: v for k, v in selected.items() if v is not None}
print(f"Selected fans: {list(selected.keys())}")

# ---------------------------------------------------------------------------
# 4. Save outputs
# ---------------------------------------------------------------------------
# 4a. All candidates CSV
rows = []
for c in candidates:
    rows.append({
        "hub": c["hub"],
        "nodes": str(c["nodes"]),
        "d_in_hub": c["d_in_hub"],
        "d_out_hub": c["d_out_hub"],
        "E_periph": c["E_periph"],
        "C": c["C"],
        "P_fan": c["P_fan"],
    })
with open(OUT_DIR / "all_fan_candidates.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["hub","nodes","d_in_hub","d_out_hub","E_periph","C","P_fan"])
    writer.writeheader()
    writer.writerows(rows)

# 4b. Individual masks and summary
name_map = {"Fan-α": "alpha", "Fan-β": "beta", "Fan-γ": "gamma"}
summary_rows = []

for name, c in selected.items():
    key = name_map[name]
    np.save(OUT_DIR / f"fan_{key}_mask.npy", c["mask"])
    summary_rows.append({
        "fan": name,
        "hub_node": c["hub"],
        "nodes": str(c["nodes"]),
        "d_in_hub": c["d_in_hub"],
        "d_out_hub": c["d_out_hub"],
        "E_periph": c["E_periph"],
        "C": round(c["C"], 4),
        "P_fan": round(c["P_fan"], 4),
    })

with open(OUT_DIR / "fan_candidate_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fan","hub_node","nodes","d_in_hub","d_out_hub","E_periph","C","P_fan"])
    writer.writeheader()
    writer.writerows(summary_rows)

# ---------------------------------------------------------------------------
# 5. Verification checkpoint
# ---------------------------------------------------------------------------
print("\n=== Stage 1 Verification ===")
for name, c in selected.items():
    m = c["mask"]
    print(f"\n{name}: hub={c['hub']}, nodes={c['nodes']}")
    print(f"  mask shape={m.shape}, edge_count={int(m.sum())}, hub_in={c['d_in_hub']}, hub_out={c['d_out_hub']}")
    print(f"  E_periph={c['E_periph']}, C={c['C']:.4f}, P_fan={c['P_fan']:.4f}")
    assert m.shape == (G, G), f"Mask shape mismatch: {m.shape}"
    assert int(m.sum()) == EDGES, f"Edge count mismatch: {int(m.sum())}"
    assert c["d_in_hub"] >= HUB_MIN_IN, f"Hub in-degree too low: {c['d_in_hub']}"
    assert c["d_out_hub"] >= HUB_MIN_OUT, f"Hub out-degree too low: {c['d_out_hub']}"
    print(f"  PASS")

print(f"\nOutputs saved to {OUT_DIR}/")
print("  - all_fan_candidates.csv")
for name in selected:
    key = name_map[name]
    print(f"  - fan_{key}_mask.npy")
print("  - fan_candidate_summary.csv")