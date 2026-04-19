# knn_mst_graph.py
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

# Optional: scikit-learn for fast kNN (recommended). If unavailable, we fall back to exact chunked search.
try:
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

# Optional: PyTorch Geometric for saving a ready-to-train Data object
try:
    import torch
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except Exception:
    _PYG_AVAILABLE = False


# ---------------------------
# Atchley utilities
# ---------------------------

def load_atchley_table(csv_path: str) -> Dict[str, List[float]]:
    """
    CSV must have column 'amino.acid' + 5 numeric columns (the Atchley factors).
    Returns dict: AA -> [f1..f5]
    """
    df = pd.read_csv(csv_path)
    if "amino.acid" not in df.columns:
        raise ValueError("Expected column 'amino.acid' in Atchley CSV.")
    aa_cols = [c for c in df.columns if c != "amino.acid"]
    if len(aa_cols) < 5:
        raise ValueError("Atchley CSV must contain 5 factor columns.")
    table = df.set_index("amino.acid")[aa_cols[:5]].astype(float).to_dict(orient="index")
    return {aa: list(vals.values()) for aa, vals in table.items()}


def sequences_to_atchley(
    sequences: List[str],
    atchley: Dict[str, List[float]],
    pad_to: Optional[int] = 8,
    unknown_policy: str = "zeros"
) -> np.ndarray:
    """
    Convert (already truncated) sequences to flattened Atchley vectors (L*5) with zero-padding.
    pad_to: residues to pad/truncate to (default 8).
    """
    valid = set(atchley.keys())
    clean = []
    for s in sequences:
        s = "" if s is None else str(s)
        s = s.strip().upper()
        if unknown_policy == "skip" and any(ch not in valid for ch in s):
            continue
        clean.append(s)

    if not clean:
        raise ValueError("No valid sequences after cleaning.")

    L = pad_to if pad_to is not None else max(len(s) for s in clean)
    X = np.zeros((len(clean), L * 5), dtype=np.float32)
    for i, s in enumerate(clean):
        for pos, ch in enumerate(s[:L]):
            if ch in atchley:
                X[i, pos*5:(pos+1)*5] = np.array(atchley[ch], dtype=np.float32)
    return X


# ---------------------------
# kNN (Euclidean)
# ---------------------------

def knn_search(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    use_sklearn: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (I, D) with shapes (N, k) — k neighbor indices and Euclidean distances for each row of X.
    Uses scikit-learn if available; otherwise a chunked exact search (still O(N^2) time).
    """
    N = X.shape[0]
    k_eff = min(k + 1, N)  # include self; drop later
    if use_sklearn is None:
        use_sklearn = _SKLEARN_AVAILABLE

    if use_sklearn:
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric, algorithm="auto")
        nbrs.fit(X)
        D, I = nbrs.kneighbors(X, return_distance=True)
        return I[:, 1:].astype(np.int64), D[:, 1:].astype(np.float32)  # drop self

    # Exact fallback (chunked; memory-friendly but time-heavy for large N)
    block = max(1, 8192 // max(1, X.shape[1]))
    all_I = np.empty((N, k), dtype=np.int64)
    all_D = np.empty((N, k), dtype=np.float32)
    for start in range(0, N, block):
        end = min(N, start + block)
        Xq = X[start:end]
        a2 = np.sum(Xq**2, axis=1, keepdims=True)        # (B,1)
        b2 = np.sum(X**2, axis=1, keepdims=True).T       # (1,N)
        dots = Xq @ X.T                                   # (B,N)
        dist2 = a2 + b2 - 2.0 * dots
        np.maximum(dist2, 0.0, out=dist2)
        for i in range(start, end):
            dist2[i - start, i] = np.inf
        part = np.argpartition(dist2, kth=k-1, axis=1)[:, :k]
        row_idx = np.arange(end - start)[:, None]
        part_d2 = dist2[row_idx, part]
        order = np.argsort(part_d2, axis=1)
        idx_sorted = part[row_idx, order]
        d2_sorted = part_d2[row_idx, order]
        all_I[start:end] = idx_sorted
        all_D[start:end] = np.sqrt(d2_sorted).astype(np.float32)
    return all_I, all_D


def build_undirected_from_knn(I: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn directed kNN (I, D) into *undirected* edges.
    Keep a single edge i<j with weight = min of the two directions if both exist.
    Returns 1D arrays (row, col, w) with row<col.
    """
    N, k = I.shape
    rows = np.repeat(np.arange(N), k)
    cols = I.reshape(-1)
    w = D.reshape(-1)

    # unordered pairs
    pairs = np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1)
    mask = pairs[:, 0] != pairs[:, 1]
    pairs, w = pairs[mask], w[mask]

    # sort, then consolidate duplicates via running minimum
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, w = pairs[order], w[order]

    min_pairs, min_w = [], []
    i = 0
    while i < len(pairs):
        j = i + 1
        cur_pair = (pairs[i, 0], pairs[i, 1])
        cur_min = w[i]
        while j < len(pairs) and pairs[j, 0] == cur_pair[0] and pairs[j, 1] == cur_pair[1]:
            if w[j] < cur_min:
                cur_min = w[j]
            j += 1
        min_pairs.append(cur_pair); min_w.append(cur_min); i = j

    min_pairs = np.array(min_pairs, dtype=np.int64)
    min_w = np.array(min_w, dtype=np.float32)
    row = min_pairs[:, 0]; col = min_pairs[:, 1]
    return row, col, min_w


# ---------------------------
# Connectivity + MST
# ---------------------------

def ensure_connected_with_bridges(
    N: int,
    row: np.ndarray,
    col: np.ndarray,
    w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    If the graph is disconnected, add the shortest available inter-component edges
    (from the current candidate edge set) until it becomes one component.
    Returns (row, col, w, was_connected_initially).
    """
    A = coo_matrix((np.ones_like(w, dtype=np.float32), (row, col)), shape=(N, N))
    A = (A + A.T).tocsr()
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return row, col, w, True

    # Union-Find on components
    parent = list(range(n_comp))
    rank = [0] * n_comp

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # shortest-first
    order = np.argsort(w)
    bridges_r, bridges_c, bridges_w = [], [], []
    comps_remaining = n_comp
    for idx in order:
        u, v = row[idx], col[idx]
        cu, cv = labels[u], labels[v]
        if cu == cv:
            continue
        if union(cu, cv):
            bridges_r.append(u); bridges_c.append(v); bridges_w.append(w[idx])
            comps_remaining -= 1
            if comps_remaining == 1:
                break

    if comps_remaining > 1:
        raise RuntimeError(
            "Still disconnected after scanning candidate edges. "
            "Increase k and rebuild kNN (or add a radius graph)."
        )

    # merge and deduplicate
    r2 = np.concatenate([row, np.array(bridges_r, dtype=np.int64)])
    c2 = np.concatenate([col, np.array(bridges_c, dtype=np.int64)])
    w2 = np.concatenate([w, np.array(bridges_w, dtype=np.float32)])
    return build_undirected_from_knn(
        np.stack([r2, c2], axis=1).reshape(-1, 1),
        np.stack([w2], axis=1).reshape(-1, 1)
    ) + (False,)


def mst_from_undirected_edges(
    N: int,
    row: np.ndarray,
    col: np.ndarray,
    w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run SciPy's MST on the symmetric weighted graph; returns undirected edges (row<col).
    """
    W = coo_matrix((w, (row, col)), shape=(N, N))
    W = (W + W.T).tocsr()
    mst = minimum_spanning_tree(W).tocoo()
    keep = mst.row < mst.col
    return mst.row[keep], mst.col[keep], mst.data.astype(np.float32)[keep]


# ---------------------------
# Normalization
# ---------------------------

def normalize_edge_weights(
    w: np.ndarray,
    ref_value: Optional[float] = None,
    percentile: float = 95.0
) -> Tuple[np.ndarray, float]:
    """
    Normalize distances to [0,1] using a robust scale.
    If ref_value is provided => global normalization (use same value across patients).
    Else => per-graph normalization using the given percentile (e.g., 95th).
    """
    if ref_value is None:
        ref_value = np.percentile(w, percentile) if len(w) > 0 else 1.0
        if ref_value <= 0:
            ref_value = 1.0
    w_norm = np.clip(w / ref_value, 0.0, 1.0).astype(np.float32)
    return w_norm, float(ref_value)


# ---------------------------
# Main entry points
# ---------------------------

def build_patient_graph(
    sequences: List[str],
    atchley_csv_path: str,
    k: int = 30,
    pad_to_len: Optional[int] = 8,
    normalize_percentile: float = 95.0,
    global_ref_value: Optional[float] = None,
    train_test_split: float = 0.8,
    out_dir: Optional[str] = None,
    patient_id: Optional[str] = None,
    ensure_connected: bool = True,
) -> dict:
    """
    Build a patient graph in 40-D Atchley space and save to disk.

    Saves:
      - {patient}_mst_edges.npz  (row, col, raw w, normalized w_norm, etc.)
      - {patient}_x.npy          (node features, shape [N, 40])
      - {patient}_meta.json      (masks, normalization info, etc.)
      - {patient}_graph.pt       (PyG Data, if torch_geometric is available)

    Returns dict with file paths.
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    pid = patient_id or "patient"

    # 1) Atchley 40-D features
    atchley = load_atchley_table(atchley_csv_path)
    X = sequences_to_atchley(sequences, atchley, pad_to=pad_to_len, unknown_policy="zeros")
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 sequences to build a graph.")

    # 2) k-NN in Euclidean
    I, D = knn_search(X, k=k, metric="euclidean")

    # 3) Undirected edges (row<col)
    row, col, w = build_undirected_from_knn(I, D)

    # 4) Ensure connectivity (if requested)
    was_connected = True
    if ensure_connected:
        try:
            row, col, w, was_connected = ensure_connected_with_bridges(N, row, col, w)
        except RuntimeError:
            # Fallback: try larger k once (up to 200)
            if _SKLEARN_AVAILABLE and k < min(200, N-1):
                I2, D2 = knn_search(X, k=min(200, N-1), metric="euclidean")
                row2, col2, w2 = build_undirected_from_knn(I2, D2)
                row, col, w, was_connected = ensure_connected_with_bridges(N, row2, col2, w2)
            else:
                raise

    # 5) MST
    mst_r, mst_c, mst_w = mst_from_undirected_edges(N, row, col, w)

    # 6) Normalize distances
    mst_w_norm, used_ref = normalize_edge_weights(mst_w, ref_value=global_ref_value, percentile=normalize_percentile)

    # 7) Simple node train/test split (80/20 by default)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_train = int(train_test_split * N)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train_mask = np.zeros(N, dtype=bool); train_mask[train_idx] = True
    test_mask = np.zeros(N, dtype=bool); test_mask[test_idx] = True

    saved = {}

    # Edges as .npz (undirected i<j)
    edges_npz = os.path.join(out_dir, f"{pid}_mst_edges.npz")
    np.savez_compressed(
        edges_npz,
        row=mst_r, col=mst_c, w=mst_w, w_norm=mst_w_norm,
        used_ref=used_ref, was_connected_initially=was_connected,
        k_used=k, pad_to_len=pad_to_len
    )
    saved["edges_npz"] = edges_npz

    # Node features
    x_npy = os.path.join(out_dir, f"{pid}_x.npy")
    np.save(x_npy, X)
    saved["x_npy"] = x_npy

    # Meta (masks, normalization, etc.)
    meta_json = os.path.join(out_dir, f"{pid}_meta.json")
    with open(meta_json, "w") as f:
        json.dump({
            "N": int(N),
            "train_ratio": float(train_test_split),
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "normalize_percentile": float(normalize_percentile),
            "used_ref_value": float(used_ref),
            "was_connected_initially": bool(was_connected),
        }, f, indent=2)
    saved["meta_json"] = meta_json

    # PyG Data (if available) — convenient for direct GNN training
    if _PYG_AVAILABLE:
        ei_src = np.concatenate([mst_r, mst_c], axis=0)
        ei_dst = np.concatenate([mst_c, mst_r], axis=0)
        edge_index = np.stack([ei_src, ei_dst], axis=0).astype(np.int64)
        edge_attr = np.concatenate([mst_w_norm, mst_w_norm], axis=0).astype(np.float32)

        data = Data(
            x=torch.from_numpy(X),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
        )
        data.train_mask = torch.from_numpy(train_mask)
        data.test_mask = torch.from_numpy(test_mask)

        pyg_path = os.path.join(out_dir, f"{pid}_graph.pt")
        torch.save(data, pyg_path)
        saved["pyg_pt"] = pyg_path

    return saved



def compute_global_ref_from_patients(
    patients_sequences: List[List[str]],
    atchley_csv_path: str,
    k: int = 30,
    pad_to_len: Optional[int] = 8,
    sample_per_patient: int = 50_000,
    percentile: float = 95.0,
) -> float:
    """
    Compute a global reference percentile (e.g., p95) of Euclidean k-NN distances
    pooled across many patient graphs (no all-pairs matrix).
    """
    atchley = load_atchley_table(atchley_csv_path)
    all_samples = []

    for seqs in patients_sequences:
        X = sequences_to_atchley(seqs, atchley, pad_to=pad_to_len, unknown_policy="zeros")
        if len(X) < 2:
            continue
        I, D = knn_search(X, k=k, metric="euclidean")
        flat = D.reshape(-1)
        if len(flat) > sample_per_patient:
            idx = np.random.choice(len(flat), size=sample_per_patient, replace=False)
            flat = flat[idx]
        all_samples.append(flat.astype(np.float32))

    if not all_samples:
        return 1.0

    pooled = np.concatenate(all_samples, axis=0)
    ref = np.percentile(pooled, percentile) if len(pooled) > 0 else 1.0
    return float(ref if ref > 0 else 1.0)

def truncate_sequences(df, cdr3_header, right=4, left=4):
    """
    Truncate sequences
    """
    
    def truncate_sequence(seq):
        if pd.isna(seq) or not isinstance(seq, str):
            return None  # Handle NaN or non-string entries gracefully
        mid = (len(seq) + 1) // 2
        return seq[max(0, mid - right):min(len(seq), mid + left)]
    
    df["cdr3_truncated"] = df[cdr3_header].apply(truncate_sequence)
    
    
    return df


# # from knn_mst_graph import build_patient_graph, compute_global_ref_from_patients

# ATCHLEY_CSV = "/mnt/data/atchley.csv"  # <-- change if needed

# # Suppose you have patients as: dict[id] -> list_of_truncated_sequences
# patients = {
#     "p001": ["CASSLGQETQ", "CASSLGQDTQ", "CASSQGQETQ", ...],  # truncated already (<=8 AA center ok)
#     "p002": [...],
#     # ...
# }

# # (Optional) compute a GLOBAL reference p95 on TRAINING patients only:
# train_ids = ["p001"]  # fill with your training patient IDs
# ref_p95 = compute_global_ref_from_patients(
#     [patients[pid] for pid in train_ids],
#     atchley_csv_path=ATCHLEY_CSV,
#     k=30,
#     pad_to_len=8,
#     percentile=95.0,
# )

# # Build and save each patient's graph (uses the global ref if provided)
# for pid, seqs in patients.items():
#     saved = build_patient_graph(
#         sequences=seqs,
#         atchley_csv_path=ATCHLEY_CSV,
#         k=30,
#         pad_to_len=8,
#         normalize_percentile=95.0,
#         global_ref_value=ref_p95,      # or None for per-patient scaling
#         train_test_split=0.8,
#         out_dir="./graphs_out",
#         patient_id=pid,
#         ensure_connected=True,
#     )
#     print(pid, "saved:", saved)
