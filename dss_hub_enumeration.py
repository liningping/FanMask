#!/usr/bin/env python3
"""
Step 1: Hub Anchor Enumeration
==============================
Enumerate all fan-in hubs (in-degree >= 4) from connectome.
For each hub, build a 2-hop neighborhood as the node search space.

Output: outputs/meta_search/hub_anchors.csv
    hub | n_neighbors_2hop | n_edges_in_2hop | candidate_nodes (sorted by weight)
"""

import csv
from pathlib import Path
import numpy as np

CONNECTOME = "data/conn_matrix_complete.npy"
OUT_DIR = Path("outputs/meta_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q = 0.25            # top-q threshold for binarization (matches Stage 1)
HUB_MIN_IN = 4      # fan-in hub criterion
N_NODES = 8         # subgraph size
N_TOP_NEIGHBORS = 14  # candidate pool size: hub + top-14 weighted 1-hop neighbors = 15 candidates

A = np.load(CONNECTOME)
np.fill_diagonal(A, 0)

# Min-max normalize
mask_nz = A != 0
A_norm = np.zeros_like(A)
v_min, v_max = A[mask_nz].min(), A[mask_nz].max()
A_norm[mask_nz] = (A[mask_nz] - v_min) / (v_max - v_min)

# Top-q binarization
total = 52 * 51
q = int(Q * total)
nz = A_norm[mask_nz]
thr = np.partition(nz, len(nz) - q)[len(nz) - q]
B = ((A_norm >= thr) & (A_norm > 0)).astype(int)

# Find hubs
hubs = [v for v in range(52) if B[:, v].sum() >= HUB_MIN_IN]
print(f"Found {len(hubs)} fan-in hubs (in-degree >= {HUB_MIN_IN})")

# For each hub, take top-N_TOP weighted 1-hop neighbors as candidates
def hub_candidates(B, A_norm, hub, n_top):
    one_hop = list((set(np.where(B[:, hub])[0]) | set(np.where(B[hub])[0])) - {hub})
    # weighted by sum of A_norm[hub,v] + A_norm[v,hub]
    one_hop.sort(key=lambda v: A_norm[hub, v] + A_norm[v, hub], reverse=True)
    return [hub] + one_hop[:n_top]

rows = []
for hub in hubs:
    candidates = hub_candidates(B, A_norm, hub, N_TOP_NEIGHBORS)
    cand_arr = np.array(candidates)
    sub = B[np.ix_(cand_arr, cand_arr)]

    rows.append({
        "hub": hub,
        "hub_in": int(B[:, hub].sum()),
        "hub_out": int(B[hub].sum()),
        "n_candidates": len(candidates),
        "induced_edges": int(sub.sum()),
        "candidates": ",".join(str(v) for v in candidates),
    })

# Save
with open(OUT_DIR / "hub_anchors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Summary
print(f"\nNeighborhood sizes:")
print(f"  candidates:    min={min(r['n_candidates'] for r in rows)}, "
      f"max={max(r['n_candidates'] for r in rows)}, "
      f"mean={np.mean([r['n_candidates'] for r in rows]):.1f}")
print(f"  induced edges: min={min(r['induced_edges'] for r in rows)}, "
      f"max={max(r['induced_edges'] for r in rows)}, "
      f"mean={np.mean([r['induced_edges'] for r in rows]):.1f}")

# Print first 5
print(f"\nFirst 5 hubs:")
for r in rows[:5]:
    print(f"  hub={r['hub']:>2}: in={r['hub_in']}, out={r['hub_out']}, "
          f"n_cand={r['n_candidates']}, induced_edges={r['induced_edges']}")

print(f"\nWrote {OUT_DIR / 'hub_anchors.csv'} ({len(rows)} hubs)")