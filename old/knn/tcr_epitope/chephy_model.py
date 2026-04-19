# tcr_knn_graph.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional, Dict
import warnings
from dataclasses import dataclass

# Optional deps; we’ll use them if present
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
except Exception:
    StandardScaler = None
    NearestNeighbors = None

# --------------------------
# 1) Feature matrix builder
# --------------------------

def load_atchley_map(csv_path: str = "/home/dsi/chenbis/repos/sol_lab/files/atchley.csv") -> Dict[str, np.ndarray]:
    """
    Loads Atchley factors from CSV and returns {AA: vec5}.
    Assumes the first text column holds the amino-acid letter,
    and the first 5 numeric columns are the factors.
    """
    df = pd.read_csv(csv_path)
    # Identify columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 5:
        raise ValueError(f"Expected at least 5 numeric columns in {csv_path}. Found {len(num_cols)}")
    vec_cols = num_cols[:5]
    # Amino-acid column (first non-numeric)
    non_num = [c for c in df.columns if c not in vec_cols]
    aa_col = non_num[0]
    mapping = {str(row[aa_col]).strip(): row[vec_cols].to_numpy(dtype=np.float32)
               for _, row in df.iterrows()}
    return mapping

def sequences_to_features(
    seqs: Iterable[str],
    atchley: Optional[Dict[str, np.ndarray]] = None,
    standardize: bool = False,
    scaler: Optional["StandardScaler"] = None,  # prefit scaler (train only)
) -> Tuple[np.ndarray, list, Optional["StandardScaler"]]:
    """
    Converts an iterable of 8-aa sequences into an (n, 40) feature matrix
    by concatenating 5D Atchley vectors per position. Optionally z-scores.

    Parameters
    ----------
    seqs : iterable of str (each length 8)
    atchley : dict {AA: vec5}; if None, loads from /mnt/data/atchley.csv
    standardize : if True, z-score columns. If scaler is provided, uses it.
    scaler : an optional prefit StandardScaler to apply (no refit).

    Returns
    -------
    X : (n, 40) float32
    seqs : list[str] in the order used
    fitted_scaler : StandardScaler or None
    """
    if atchley is None:
        atchley = load_atchley_map()

    # seqs = list(sequences_set)
    if any(len(s) != 8 for s in seqs):
        raise ValueError("All sequences must be exactly length 8.")

    # Encode; unknown residues get zeros
    def encode_seq(s: str) -> np.ndarray:
        vecs = []
        for aa in s:
            v = atchley.get(aa)
            if v is None:
                v = np.zeros(5, dtype=np.float32)
            vecs.append(v)
        return np.concatenate(vecs).astype(np.float32)  # 8*5=40

    X = np.vstack([encode_seq(s) for s in seqs]).astype(np.float32)

    fitted_scaler = None
    if standardize:
        if StandardScaler is None:
            raise ImportError("scikit-learn is required for standardization.")
        if scaler is None:
            fitted_scaler = StandardScaler(with_mean=True, with_std=True)
            X = fitted_scaler.fit_transform(X).astype(np.float32)
        else:
            fitted_scaler = scaler
            X = fitted_scaler.transform(X).astype(np.float32)

    return X, fitted_scaler


def sequences_to_features_any_length(
    seqs: Iterable[str],
    atchley: Optional[Dict[str, np.ndarray]] = None,
    standardize: bool = False,
    scaler: Optional["StandardScaler"] = None,
    agg: str = "pad",
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, Optional["StandardScaler"]]:
    """
    Convert variable-length sequences into fixed-size feature vectors.

    Two aggregation modes are supported:
      - "pad": pad sequences (on the right) with zeros up to `target_length` or the
               maximum sequence length in `seqs`. Returns shape (n, L*5).
      - "mean": average the Atchley vectors across positions, returning shape (n, 5).

    Parameters
    ----------
    seqs : iterable of str
    atchley : mapping AA -> 5-d vector
    standardize : if True, z-score columns. If `scaler` provided, uses it.
    scaler : optional prefit StandardScaler
    agg : 'pad' or 'mean'
    target_length : desired length when using 'pad' (defaults to max length in seqs)

    Returns
    -------
    X : ndarray (n, d)
    fitted_scaler : StandardScaler or None
    """
    if atchley is None:
        atchley = load_atchley_map()

    seqs_list = list(seqs)
    if len(seqs_list) == 0:
        return np.zeros((0, 0), dtype=np.float32), None

    lengths = [len(s) if isinstance(s, str) else 0 for s in seqs_list]

    if agg not in ("pad", "mean"):
        raise ValueError("agg must be 'pad' or 'mean'")

    def encode_seq_to_vectors(s: str):
        vecs = []
        for aa in s:
            v = atchley.get(aa)
            if v is None:
                v = np.zeros(5, dtype=np.float32)
            vecs.append(v)
        if not vecs:
            return np.zeros((0, 5), dtype=np.float32)
        return np.vstack(vecs).astype(np.float32)

    if agg == "mean":
        X = []
        for s in seqs_list:
            if not isinstance(s, str) or len(s) == 0:
                X.append(np.zeros(5, dtype=np.float32))
                continue
            vs = encode_seq_to_vectors(s)
            X.append(np.mean(vs, axis=0).astype(np.float32))
        X = np.vstack(X).astype(np.float32)
        fitted_scaler = None
        if standardize:
            if StandardScaler is None:
                raise ImportError("scikit-learn is required for standardization.")
            if scaler is None:
                fitted_scaler = StandardScaler(with_mean=True, with_std=True)
                X = fitted_scaler.fit_transform(X).astype(np.float32)
            else:
                # If provided scaler expects different feature size, fit a new one
                expects = getattr(scaler, "n_features_in_", None)
                if expects is not None and expects != X.shape[1]:
                    warnings.warn(
                        f"Provided scaler expects {expects} features but data has {X.shape[1]}; fitting a new scaler.",
                        UserWarning,
                    )
                    fitted_scaler = StandardScaler(with_mean=True, with_std=True)
                    X = fitted_scaler.fit_transform(X).astype(np.float32)
                else:
                    fitted_scaler = scaler
                    X = fitted_scaler.transform(X).astype(np.float32)
        return X, fitted_scaler

    # agg == 'pad'
    max_len = int(target_length) if target_length is not None else max(lengths)
    if max_len <= 0:
        # nothing to encode
        return np.zeros((len(seqs_list), 0), dtype=np.float32), None

    X = np.zeros((len(seqs_list), max_len * 5), dtype=np.float32)
    for i, s in enumerate(seqs_list):
        if not isinstance(s, str) or len(s) == 0:
            continue
        vs = encode_seq_to_vectors(s)
        # truncate if longer than max_len, else pad with zeros
        use_len = min(vs.shape[0], max_len)
        if use_len > 0:
            X[i, : use_len * 5] = vs[:use_len, :].reshape(use_len * 5)

    fitted_scaler = None
    if standardize:
        if StandardScaler is None:
            raise ImportError("scikit-learn is required for standardization.")
        if scaler is None:
            fitted_scaler = StandardScaler(with_mean=True, with_std=True)
            X = fitted_scaler.fit_transform(X).astype(np.float32)
        else:
            expects = getattr(scaler, "n_features_in_", None)
            if expects is not None and expects != X.shape[1]:
                warnings.warn(
                    f"Provided scaler expects {expects} features but data has {X.shape[1]}; fitting a new scaler.",
                    UserWarning,
                )
                fitted_scaler = StandardScaler(with_mean=True, with_std=True)
                X = fitted_scaler.fit_transform(X).astype(np.float32)
            else:
                fitted_scaler = scaler
                X = fitted_scaler.transform(X).astype(np.float32)

    return X, fitted_scaler

# ---------------------------------------
# 2) k-NN neighbor search (no full matrix)
# ---------------------------------------

@dataclass
class KNNConfig:
    k: int
    metric: str = "euclidean"     # for embeddings we use Euclidean
    use_ann_if_large: bool = True
    ann_threshold: int = 5000      # if n >= threshold, try ANN if available
    n_jobs: int = -1               # parallelism for sklearn

def get_k_neighbors(
    X: np.ndarray,
    cfg: KNNConfig,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, distances) of shape (n, k) for each row in X.
    Does NOT return self as a neighbor.

    Heuristic:
      - If n >= ann_threshold and pynndescent is available -> ANN
      - Else -> scikit-learn NearestNeighbors
    """
    n, d = X.shape

    # Try ANN (pynndescent) when large
    if cfg.use_ann_if_large and n >= cfg.ann_threshold:
        try:
            from pynndescent import NNDescent  # type: ignore
            # Build index
            index = NNDescent(
                X,
                n_neighbors=cfg.k + 1,   # +1 for self
                metric=cfg.metric,
                random_state=random_state,
                n_jobs=cfg.n_jobs if cfg.n_jobs != -1 else None,
            )
            inds, dists = index.neighbor_graph
            # Remove self (distance=0 at position 0 typically)
            inds = inds[:, 1:cfg.k + 1]
            dists = dists[:, 1:cfg.k + 1]
            return inds.astype(np.int64), dists.astype(np.float32)
        except Exception:
            pass  # fall back to sklearn

    # scikit-learn exact search
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for exact k-NN.")

    # Choose a sensible algorithm (sklearn's 'auto' is fine; brute for high-d or huge n)
    algorithm = "brute" if (d >= 30 or n > 30000) else "auto"
    nn = NearestNeighbors(
        n_neighbors=cfg.k + 1,  # include self
        algorithm=algorithm,
        metric=cfg.metric,
        n_jobs=cfg.n_jobs,
    ).fit(X)

    dists, inds = nn.kneighbors(X, return_distance=True)
    # drop self (first neighbor)
    inds = inds[:, 1:cfg.k + 1]
    dists = dists[:, 1:cfg.k + 1]
    return inds.astype(np.int64), dists.astype(np.float32)

# ----------------------------------------------------
# 3) Undirected graph builder with optional edge-weights
# ----------------------------------------------------

def build_undirected_graph_from_knn(
    indices: np.ndarray,          # (n, k) neighbor indices
    distances: np.ndarray,        # (n, k) neighbor distances (Euclidean)
    weighting: str = "none",      # "none" | "rbf" | "self_tuning" | "inverse"
    sigma: Optional[float] = None,
    mutual: bool = True,          # keep edge only if i in j's k-NN and j in i's
    return_edge_index: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Builds an undirected edge list (unique pairs) with optional weights.

    weighting:
      - "none": all weights = 1.0
      - "inverse": w = 1 / (1 + d)
      - "rbf": global Gaussian weight: w = exp(-d^2 / (2*sigma^2))
               If sigma is None, uses median of retained distances.
      - "self_tuning": local scale: w = exp(-d^2 / (sigma_i * sigma_j + eps)),
                       where sigma_i is the distance to i's k-th neighbor.

    Returns
    -------
    edge_index : (2, E) int64  (u, v), undirected (each pair once)
    edge_weight : (E,) float32 or None
    """
    n, k = indices.shape
    if distances.shape != indices.shape:
        raise ValueError("indices and distances must have the same shape")

    # Prepare symmetric pair counts to enforce mutual if needed
    from collections import defaultdict
    pair_count = defaultdict(int)
    pair_min_d = {}
    # record the d of i's k-th neighbor for self_tuning
    kth = distances[:, -1].astype(np.float32)

    for i in range(n):
        for r in range(k):
            j = int(indices[i, r])
            if j < 0 or j == i:
                continue
            key = (i, j) if i < j else (j, i)
            pair_count[key] += 1
            d = float(distances[i, r])
            if key in pair_min_d:
                pair_min_d[key] = min(pair_min_d[key], d)
            else:
                pair_min_d[key] = d

    # Keep pairs; if mutual=True, require count==2
    kept_pairs = [p for p, c in pair_count.items() if (c == 2 if mutual else c >= 1)]
    if not kept_pairs:
        # Return empty
        return np.zeros((2, 0), dtype=np.int64), None if weighting == "none" else np.zeros((0,), dtype=np.float32)

    d_vals = np.array([pair_min_d[p] for p in kept_pairs], dtype=np.float32)

    # Compute weights
    if weighting == "none":
        w = None
    elif weighting == "inverse":
        w = 1.0 / (1.0 + d_vals)
    elif weighting == "rbf":
        if sigma is None or sigma <= 0:
            # robust default: median distance
            sigma = float(np.median(d_vals))
            if sigma == 0.0:
                sigma = 1e-6
        w = np.exp(- (d_vals ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)
    elif weighting == "self_tuning":
        # σ_i = distance to the k-th neighbor of node i
        eps = 1e-12
        sigma_i = np.array([kth[u] for (u, v) in kept_pairs], dtype=np.float32)
        sigma_j = np.array([kth[v] for (u, v) in kept_pairs], dtype=np.float32)
        denom = (sigma_i * sigma_j) + eps
        w = np.exp(- (d_vals ** 2) / denom).astype(np.float32)
    else:
        raise ValueError("weighting must be one of: 'none', 'inverse', 'rbf', 'self_tuning'")

    # Build undirected edge_index with each pair once
    u = np.array([u for (u, v) in kept_pairs], dtype=np.int64)
    v = np.array([v for (u, v) in kept_pairs], dtype=np.int64)
    edge_index = np.vstack([u, v])  # shape (2, E)

    return edge_index, (None if w is None else w.astype(np.float32))

# --------------------------
# End-to-end convenience API
# --------------------------

def build_patient_knn_graph(
    seqs: Iterable[str],
    k: int,
    atchley_map: Optional[Dict[str, np.ndarray]] = None,
    standardize_features: bool = False,
    scaler: Optional["StandardScaler"] = None,
    weighting: str = "none",           # "none" | "inverse" | "rbf" | "self_tuning"
    sigma: Optional[float] = None,     # used only if weighting == "rbf"
    mutual: bool = True,
    knn_cfg: Optional[KNNConfig] = None,
):
    """
    One-shot helper: sequences -> features -> kNN -> undirected graph.
    Returns: (edge_index, edge_weight, X, seqs, fitted_scaler)
    """
    if knn_cfg is None:
        knn_cfg = KNNConfig(k=k)

    X, fitted_scaler = sequences_to_features(
        seqs,
        atchley=atchley_map,
        standardize=standardize_features,
        scaler=scaler,
    )
    inds, dists = get_k_neighbors(X, knn_cfg)
    edge_index, edge_weight = build_undirected_graph_from_knn(
        inds, dists, weighting=weighting, sigma=sigma, mutual=mutual
    )
    return edge_index, edge_weight, X, fitted_scaler


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
