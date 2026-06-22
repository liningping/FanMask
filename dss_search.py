#!/usr/bin/env python3
"""
Plan E-Lite: Robustness-Guided Search (v3)
===========================================
v2 loss: CE(clean) + CE(noisy_σ=0.3) + 0.1 * card
v3 loss: 0.3 * CE(clean) + 0.7 * CE(noisy_σ_random) + 0.1 * card
   where σ_random is uniformly sampled from avg_ratio sigmas per mini-batch.

Output: outputs/meta_search_v3/hub*_hard_mask.npy + search_log_v3.csv
"""

import csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT_DIR = Path("outputs/meta_search_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G = 8
HIDDEN = 64
GROUP_SIZE = HIDDEN // G  # 8
EDGES = 22
N_NEIGHBORS_KEEP = 7      # plus hub = 8 nodes
SEARCH_EPOCHS = 12
SEARCH_SEEDS = 1
BATCH_SIZE = 256
LR = 5e-3
NOISE_SIGMA = 0.3
# v3: sample noise σ uniformly from avg_ratio evaluation set
SEARCH_SIGMAS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CLEAN_W = 0.3
NOISY_W = 0.7
TAU_INIT = 1.5
TAU_FINAL = 0.05
CARD_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Connectome preprocessing
# ---------------------------------------------------------------------------
def load_connectome():
    A = np.load("data/conn_matrix_complete.npy")
    np.fill_diagonal(A, 0)
    nz = A != 0
    A_norm = np.zeros_like(A)
    A_norm[nz] = (A[nz] - A[nz].min()) / (A[nz].max() - A[nz].min())
    total = 52 * 51
    q = int(0.25 * total)
    vals = A_norm[nz]
    thr = np.partition(vals, len(vals) - q)[len(vals) - q]
    B = ((A_norm >= thr) & (A_norm > 0)).astype(int)
    return A_norm, B


def gumbel_sigmoid(logits, tau, hard=False):
    """Differentiable [0,1] sample. hard=True uses straight-through."""
    g1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    g2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    y_soft = torch.sigmoid((logits + g1 - g2) / tau)
    if not hard:
        return y_soft
    y_hard = (y_soft > 0.5).float()
    return y_hard + (y_soft - y_soft.detach())


# ---------------------------------------------------------------------------
# Search model with fully-differentiable selection
# ---------------------------------------------------------------------------
class SearchModelV2(nn.Module):
    """
    candidates: 15 connectome nodes (candidates[0]=hub).
    edge_idx:   list of (i,j) connectome edges in local indices.

    Each forward:
      - Sample node_keep (15,) via Gumbel-Sigmoid; hub forced to 1.
      - Sample edge_keep (n_edges,) via Gumbel-Sigmoid.
      - Build (15,15) soft mask = node_keep_i * node_keep_j * edge_keep_k for each edge.
      - Routing to (G,G): assign each candidate node to a unique group via
        soft permutation derived from node_keep ranks (SoftSort).
    """
    def __init__(self, candidates, edge_idx):
        super().__init__()
        self.candidates = candidates
        self.edge_idx = edge_idx
        self.n_cand = len(candidates)        # = 15
        self.n_neigh_choice = self.n_cand - 1
        self.n_edges = len(edge_idx)

        # Logits
        self.node_logits = nn.Parameter(torch.zeros(self.n_neigh_choice))
        self.edge_logits = nn.Parameter(torch.zeros(self.n_edges))

        # ANN
        self.proj = nn.Linear(784, HIDDEN)
        self.cell0 = nn.Linear(HIDDEN, HIDDEN)
        self.cell1 = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, 10)

    def sample_selection(self, tau):
        """Returns (all_keep [15], edge_keep [n_edges])."""
        node_keep = gumbel_sigmoid(self.node_logits, tau)
        edge_keep = gumbel_sigmoid(self.edge_logits, tau)
        all_keep = torch.cat([torch.ones(1, device=node_keep.device), node_keep])
        return all_keep, edge_keep

    def cardinality_penalty(self, all_keep, edge_keep):
        """Encourage |kept neighbors| = N_NEIGHBORS_KEEP and |kept edges| = EDGES."""
        node_card = (all_keep[1:].sum() - N_NEIGHBORS_KEEP) ** 2 / (N_NEIGHBORS_KEEP ** 2)
        edge_card = (edge_keep.sum() - EDGES) ** 2 / (EDGES ** 2)
        return node_card + edge_card

    def build_local_mask(self, all_keep, edge_keep):
        """(n_cand, n_cand) soft mask in local indices."""
        M = torch.zeros(self.n_cand, self.n_cand, device=all_keep.device)
        for k, (i, j) in enumerate(self.edge_idx):
            M = M + torch.zeros_like(M).index_put_(
                (torch.tensor([i], device=M.device), torch.tensor([j], device=M.device)),
                all_keep[i] * all_keep[j] * edge_keep[k])
        return M

    def soft_route_to_8(self, local_M, all_keep):
        """
        Project (15,15) -> (8,8) by selecting hub + top-7 neighbors via SoftSort.
        SoftSort: build a soft permutation matrix P where P[g,c] is the prob
        that group g maps to candidate c.
        """
        # SoftSort approximation: sort all_keep[1:] descending; map top-7 to groups 1..7.
        # We use straight-through Gumbel argmax to keep this differentiable.
        # Simpler: use top-K soft selection: for each group g (g=1..7), the assignment
        # weights are softmax of (rank-encoding * scale).
        device = local_M.device
        n = self.n_cand  # 15

        # Rank-based soft assignment:
        # candidate weight for group g = softmax over candidates by (all_keep - g_threshold)
        # Approx: use sorted order of all_keep[1:] to make assignment.
        # For full differentiability, use SoftSort matrix.
        scores = all_keep[1:]  # 14 neighbors
        # SoftSort sort matrix (descending): P[i,j] = soft prob that pos-i is candidate-j
        N = scores.shape[0]
        scores_col = scores.unsqueeze(1)               # (14, 1)
        scores_row = scores.unsqueeze(0)               # (1, 14)
        # P_softsort[i, j] = softmax_j(-|sort_pos_i - score_j| / tau_sort)
        # Use SoftSort tau small for sharp
        tau_sort = 0.5
        # The standard SoftSort: sort_scores[i] - scores[j] -> argmax j
        sorted_scores, _ = torch.sort(scores, descending=True)  # (14,)
        diff = -(sorted_scores.unsqueeze(1) - scores_row).abs() / tau_sort  # (14, 14)
        P_sort = F.softmax(diff, dim=1)  # P_sort[i, j]: sorted-pos i ~ neighbor j

        # Group g (g=1..7) corresponds to sort position g-1 → assignment weights:
        # group_to_neighbor[g-1, j] = P_sort[g-1, j]
        group_assign = torch.zeros(G, n, device=device)
        # group 0 = hub (candidate 0)
        group_assign[0, 0] = 1.0
        # groups 1..7 = top-7 sorted neighbors
        for g in range(1, G):
            # P_sort row g-1, but indices in P_sort are over neighbors[0..13],
            # which correspond to candidates[1..14]
            group_assign[g, 1:] = P_sort[g - 1]

        # Build (G, G) mask via group_assign @ local_M @ group_assign.T
        # But local_M is (n_cand, n_cand), so:
        # routed_M[g_out, g_in] = sum_{i,j} group_assign[g_out, i] * local_M[i, j] * group_assign[g_in, j]
        routed_M = group_assign @ local_M @ group_assign.t()
        return routed_M, group_assign

    def forward(self, x, tau, sigma=0.0):
        all_keep, edge_keep = self.sample_selection(tau)
        local_M = self.build_local_mask(all_keep, edge_keep)
        soft_M, _ = self.soft_route_to_8(local_M, all_keep)

        x = x.view(-1, 784)
        if sigma > 0:
            x = x + sigma * torch.randn_like(x)
        x = F.relu(self.proj(x))

        # Cell 0
        w = self.cell0.weight
        w_g = w.view(G, GROUP_SIZE, G, GROUP_SIZE) * soft_M.view(G, 1, G, 1)
        x = F.relu(F.linear(x, w_g.reshape(HIDDEN, HIDDEN), self.cell0.bias))

        # Cell 1
        w = self.cell1.weight
        w_g = w.view(G, GROUP_SIZE, G, GROUP_SIZE) * soft_M.view(G, 1, G, 1)
        x = F.relu(F.linear(x, w_g.reshape(HIDDEN, HIDDEN), self.cell1.bias))

        cardinality = self.cardinality_penalty(all_keep, edge_keep)
        return self.head(x), cardinality

    @torch.no_grad()
    def discretize(self):
        """Take top-7 neighbors by node_logits and top-22 edges among them."""
        node_probs = torch.sigmoid(self.node_logits).cpu().numpy()
        edge_probs = torch.sigmoid(self.edge_logits).cpu().numpy()

        top_neigh = np.argsort(-node_probs)[:N_NEIGHBORS_KEEP]
        sel_local = [0] + (top_neigh + 1).tolist()
        sel_global = [self.candidates[i] for i in sel_local]

        # Among edges with both endpoints in sel_local, take top-EDGES
        valid = []
        for k, (i, j) in enumerate(self.edge_idx):
            if i in sel_local and j in sel_local:
                valid.append((k, i, j, edge_probs[k]))
        valid.sort(key=lambda x: -x[3])
        chosen = valid[:EDGES]

        local_to_slot = {v: idx for idx, v in enumerate(sel_local)}
        M = np.zeros((G, G), dtype=np.float32)
        for k, i, j, _ in chosen:
            M[local_to_slot[i], local_to_slot[j]] = 1.0
        return M, sel_global, len(chosen)


def loaders():
    t = transforms.Compose([transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.view(-1))])
    return (DataLoader(datasets.MNIST("data", train=True, download=True, transform=t),
                       batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(datasets.MNIST("data", train=False, download=True, transform=t),
                       batch_size=BATCH_SIZE))


def search_hub_one_seed(hub, candidates, A_norm, B, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cand_arr = np.array(candidates)
    sub = B[np.ix_(cand_arr, cand_arr)]
    edge_idx = [(int(i), int(j)) for i in range(len(candidates))
                for j in range(len(candidates))
                if i != j and sub[i, j] > 0]
    if len(edge_idx) < EDGES:
        return None, None, float("inf")

    model = SearchModelV2(candidates, edge_idx).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_l, _ = loaders()

    final_loss = 0
    for epoch in range(SEARCH_EPOCHS):
        # Cosine tau schedule
        progress = epoch / max(1, SEARCH_EPOCHS - 1)
        tau = TAU_FINAL + 0.5 * (TAU_INIT - TAU_FINAL) * (1 + np.cos(np.pi * progress))

        model.train()
        for x, y in train_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits_clean, card = model(x, tau, sigma=0.0)
            sigma = random.choice(SEARCH_SIGMAS)
            logits_noisy, _ = model(x, tau, sigma=sigma)
            loss = (CLEAN_W * crit(logits_clean, y)
                    + NOISY_W * crit(logits_noisy, y)
                    + CARD_WEIGHT * card)
            loss.backward()
            opt.step()
            final_loss = loss.item()

    hard_mask, sel_nodes, n_chosen = model.discretize()
    return hard_mask, sel_nodes, final_loss


def search_hub_multi_seed(hub, candidates, A_norm, B):
    """Run 3 search seeds, return result with lowest final loss."""
    best = (None, None, float("inf"))
    for seed in range(SEARCH_SEEDS):
        m, n, l = search_hub_one_seed(hub, candidates, A_norm, B, seed)
        if m is None:
            return None, None, float("inf")
        if l < best[2]:
            best = (m, n, l)
    return best


def main():
    A_norm, B = load_connectome()
    with open("outputs/meta_search/hub_anchors.csv") as f:
        hub_rows = list(csv.DictReader(f))
    print(f"Searching v2 for {len(hub_rows)} hubs", flush=True)

    log_path = OUT_DIR / "search_log_v3.csv"
    log = []
    done = set()
    if log_path.exists():
        with open(log_path) as f:
            for r in csv.DictReader(f):
                log.append(r); done.add(int(r["hub"]))
        print(f"Resume: {len(done)} hubs already done", flush=True)

    for row in hub_rows:
        hub = int(row["hub"])
        if hub in done:
            continue
        candidates = [int(v) for v in row["candidates"].split(",")]
        induced = int(row["induced_edges"])

        print(f"  hub={hub:>2}...", end=" ", flush=True)
        m, n, l = search_hub_multi_seed(hub, candidates, A_norm, B)
        if m is None:
            print("SKIP (insufficient edges)", flush=True)
            continue
        np.save(OUT_DIR / f"hub{hub}_hard_mask.npy", m)
        log.append({
            "hub": hub,
            "selected_nodes": ",".join(str(v) for v in n),
            "n_edges_in_mask": int(m.sum()),
            "best_search_loss": l,
        })
        print(f"loss={l:.4f}, edges={int(m.sum())}, nodes={n}", flush=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)

    print(f"Wrote {log_path}", flush=True)


if __name__ == "__main__":
    main()