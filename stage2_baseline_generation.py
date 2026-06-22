#!/usr/bin/env python3
"""
Stage 2: Baseline Construction
==============================
Generate Random-Sparse and Degree-Preserved (DP) baseline masks.
- Random-Sparse: uniform random masks with same edge budget, no diagonal
- DP baselines: directed double-edge swap rewires that preserve in/out degree sequence

Acceptance criteria (see EXPERIMENT_ROADMAP.md Stage 2):
    - All sparse masks have edge_count=16, no diagonal edges
    - Random-Sparse: >= 5 masks, uniform over allowed edge space
    - DP: 5 rewires per Fan, in/out degree preserved exactly
    - Outputs: random_sparse_masks/*.npy, dp_fan_*/dp_*.npy, baseline_summary.csv
"""

import csv
import numpy as np
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = Path("outputs/stage2")
STAGE1_DIR = Path("outputs/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

G = 8
EDGES = 22         # must match stage1_fan_extraction.py
N_RS = 5          # number of random-sparse masks
N_DP = 5           # number of DP rewires per fan
SEED = 42

# Valid edge space: all directed pairs (i,j) with i != j (no self-loop)
ALL_EDGES = [(i, j) for i in range(G) for j in range(G) if i != j]
N_OMEGA = len(ALL_EDGES)  # 56

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load Fan masks from Stage 1
# ---------------------------------------------------------------------------
fans = {
    "fan_alpha": np.load(STAGE1_DIR / "fan_alpha_mask.npy"),
    "fan_gamma": np.load(STAGE1_DIR / "fan_gamma_mask.npy"),
    "fan_beta":  np.load(STAGE1_DIR / "fan_beta_mask.npy"),
}
print(f"Loaded {len(fans)} fan masks")

# ---------------------------------------------------------------------------
# 2. Random-Sparse masks
# ---------------------------------------------------------------------------
rs_dir = OUT_DIR / "random_sparse_masks"
rs_dir.mkdir(exist_ok=True)

rs_masks = []
for i in range(N_RS):
    edges = random.sample(ALL_EDGES, EDGES)
    M = np.zeros((G, G), dtype=np.float32)
    for (u, v) in edges:
        M[u, v] = 1.0
    rs_masks.append(M)
    np.save(rs_dir / f"rs_{i}.npy", M)

print(f"Generated {len(rs_masks)} Random-Sparse masks")

# ---------------------------------------------------------------------------
# 3. Degree-Preserved rewires via directed double-edge swap
# ---------------------------------------------------------------------------
def get_degree_seq(M):
    """Return (in_deg, out_deg) for each node."""
    in_deg = M.sum(axis=0)   # column sum = in-degree
    out_deg = M.sum(axis=1)  # row sum = out-degree
    return in_deg, out_deg

def dp_rewire(M, max_attempts=500):
    """
    Directed double-edge swap: replace (a->b, c->d) with (a->d, c->b)
    while preserving in-degree and out-degree sequences.
    Repeats until EDGES rewires applied or convergence.
    Returns a new mask with same degree sequence.
    """
    M_new = M.copy()
    edge_list = list(zip(*np.where(M_new == 1)))

    rewired = 0
    attempts = 0
    while rewired < EDGES and attempts < max_attempts:
        attempts += 1
        # Pick two distinct edges
        a, b = edge_list[random.randrange(len(edge_list))]
        c, d = edge_list[random.randrange(len(edge_list))]
        if a == c or b == d or a == d or c == b:
            continue
        # Proposed new edges: a->d, c->b
        if M_new[a, d] == 0 and M_new[c, b] == 0:
            M_new[a, b] = 0
            M_new[c, d] = 0
            M_new[a, d] = 1
            M_new[c, b] = 1
            # Update edge list
            edge_list = list(zip(*np.where(M_new == 1)))
            rewired += 1

    return M_new

def verify_dp(M_orig, M_new):
    """Check that in/out degree sequences are preserved."""
    in_o, out_o = get_degree_seq(M_orig)
    in_n, out_n = get_degree_seq(M_new)
    return np.array_equal(in_o, in_n) and np.array_equal(out_o, out_n)

for fan_name, M_fan in fans.items():
    fan_dir = OUT_DIR / fan_name
    fan_dir.mkdir(exist_ok=True)

    for i in range(N_DP):
        M_dp = dp_rewire(M_fan, max_attempts=2000)
        assert verify_dp(M_fan, M_dp), f"DP verification failed for {fan_name} rewire {i}"
        assert int(M_dp.sum()) == EDGES, f"Edge count changed: {int(M_dp.sum())}"
        np.save(fan_dir / f"dp_{i}.npy", M_dp)

print(f"Generated DP rewires for {len(fans)} fans ({N_DP} each)")

# ---------------------------------------------------------------------------
# 4. Baseline summary
# ---------------------------------------------------------------------------
rows = []

# Random-Sparse
for i, M in enumerate(rs_masks):
    in_d, out_d = get_degree_seq(M)
    rows.append({
        "model": f"rs_{i}",
        "type": "random_sparse",
        "edge_count": int(M.sum()),
        "hub_in": int(in_d[0]),
        "hub_out": int(out_d[0]),
        "fan_hub_in": "-",
        "fan_hub_out": "-",
    })

# DP baselines
for fan_name, M_fan in fans.items():
    fan_stem = fan_name.replace("fan_", "")
    for i in range(N_DP):
        M_dp = np.load(OUT_DIR / fan_name / f"dp_{i}.npy")
        in_d, out_d = get_degree_seq(M_dp)
        rows.append({
            "model": f"dp_{fan_stem}_{i}",
            "type": f"dp_{fan_stem}",
            "edge_count": int(M_dp.sum()),
            "hub_in": int(in_d[0]),
            "hub_out": int(out_d[0]),
            "fan_hub_in": int(M_fan[:, 0].sum()),
            "fan_hub_out": int(M_fan[0, :].sum()),
        })

with open(OUT_DIR / "baseline_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model","type","edge_count","hub_in","hub_out","fan_hub_in","fan_hub_out"])
    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------------------------
# 5. Verification
# ---------------------------------------------------------------------------
print("\n=== Stage 2 Verification ===")

# Check all sparse masks
all_ok = True
for fan_name, M_fan in fans.items():
    fan_stem = fan_name.replace("fan_", "")
    for i in range(N_DP):
        M = np.load(OUT_DIR / fan_name / f"dp_{i}.npy")
        ok = (M.shape == (G, G)) and (int(M.sum()) == EDGES) and verify_dp(M_fan, M)
        if not ok:
            print(f"  FAIL: {fan_name}/dp_{i}.npy")
            all_ok = False

for i, M in enumerate(rs_masks):
    ok = (M.shape == (G, G)) and (int(M.sum()) == EDGES)
    if not ok:
        print(f"  FAIL: rs_{i}.npy")
        all_ok = False

# Check diagonal policy
for fan_name, M_fan in fans.items():
    assert M_fan.diagonal().sum() == 0, f"Fan {fan_name} has diagonal edges!"
for i, M in enumerate(rs_masks):
    assert M.diagonal().sum() == 0, f"RS mask {i} has diagonal edges!"
for fan_name in fans:
    for i in range(N_DP):
        M = np.load(OUT_DIR / fan_name / f"dp_{i}.npy")
        assert M.diagonal().sum() == 0, f"DP {fan_name}/dp_{i} has diagonal edges!"

print(f"  Total models verified: {len(fans)} fans + {N_RS} RS + {len(fans)*N_DP} DP = {len(fans) + N_RS + len(fans)*N_DP}")
if all_ok:
    print("  PASS")

print(f"\nOutputs saved to {OUT_DIR}/")
print(f"  random_sparse_masks/rs_0.npy ... rs_{N_RS-1}.npy")
for fan_name in fans:
    fan_stem = fan_name.replace("fan_", "")
    print(f"  {fan_name}/dp_0.npy ... dp_{N_DP-1}.npy")
print("  baseline_summary.csv")