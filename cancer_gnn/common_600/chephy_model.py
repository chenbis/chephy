import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


GREATEST_DIST = 654.925
atchley_path = "/home/dsi/chenbis/repos/sol_lab/files/atchley.csv"
atchley_df = pd.read_csv(atchley_path)
atchley_dict = atchley_df.set_index("amino.acid").to_dict(orient="index")



def compute_chephy_matrix_gpu(sequences):
    import cupy as cp
    from cupyx.scipy.spatial.distance import pdist

    """
    GPU-accelerated ChePhy matrix computation using CuPy.
    """
    # print(f"[GPU] Starting computation for {len(sequences)} sequences")

    aa_order = sorted(atchley_dict.keys())
    aa_vecs = cp.array([list(atchley_dict[aa].values()) for aa in aa_order])
    aa_to_index = {aa: i for i, aa in enumerate(aa_order)}
    # print(f"[GPU] Amino acid vectors prepared, shape: {aa_vecs.shape}")

    seq_indices = [[aa_to_index[aa] for aa in seq] for seq in sequences]
    max_len = max(len(seq) for seq in seq_indices)
    # print(f"[GPU] Max sequence length: {max_len}")
    padded = cp.zeros((len(sequences), max_len), dtype=cp.int32)

    for i, seq in enumerate(seq_indices):
        padded[i, :len(seq)] = cp.array(seq)
    # print(f"[GPU] Sequences padded to shape: {padded.shape}")

    # Convert to Atchley feature vectors
    features = aa_vecs[padded]  # (N, L, 5)
    vectors = features.reshape(len(sequences), -1)  # (N, L*5)
    # print(f"[GPU] Feature vectors shape: {vectors.shape}")
    # Compute distances
    # print("[GPU] Computing pairwise distances...")
    condensed = pdist(vectors, metric='sqeuclidean')
    # print(f"[GPU] Distance computation complete, condensed matrix size: {condensed.shape}")

    return cp.round(condensed, 3).astype(cp.float16)


def compute_chephy_matrix_cpu(sequences):
    """
    Compute distance matrices.
    """
    # print(f"[CPU] Starting computation for {len(sequences)} sequences")
    aa_order = sorted(atchley_dict.keys())
    aa_vecs = np.array([list(atchley_dict[aa].values()) for aa in aa_order])
    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}
    # print(f"[CPU] Amino acid vectors prepared, shape: {aa_vecs.shape}")

    # Convert sequences to index matrix
    max_len = max(len(seq) for seq in sequences)
    # print(f"[CPU] Max sequence length: {max_len}")
    index_matrix = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        index_matrix[i, :len(seq)] = [aa_to_index[aa] for aa in seq]
    # print(f"[CPU] Index matrix shape: {index_matrix.shape}")

    # Lookup Atchley vectors and flatten
    feature_matrix = aa_vecs[index_matrix]  # shape (N, L, 5)
    sequence_vectors = feature_matrix.reshape(len(sequences), -1)
    # print(f"[CPU] Feature vectors shape: {sequence_vectors.shape}")
    # Compute pairwise distances
    # print("[CPU] Computing pairwise distances...")
    condensed = pdist(sequence_vectors, metric='sqeuclidean')
    # print(f"[CPU] Distance computation complete, condensed matrix size: {condensed.shape}")
    return np.round(condensed, 3).astype(np.float16)


def find_che_phy_dist(sequences_set,greatest_distance=GREATEST_DIST):
    """
    Compute normalized condensed ChePhy and Hamming distances.
    """
    # print(f"[MAIN] Starting ChePhy distance computation for {len(sequences_set)} unique sequences")
    # print(f"[MAIN] Greatest distance normalization factor: {greatest_distance}")

    sequences_list = list(sequences_set)

    chephy_matrix = None
    try:
        import cupy as cp
        _ = cp.array([0.0])
        # print("[MAIN] CuPy available, using GPU computation")
        chephy_matrix = compute_chephy_matrix_gpu(sequences_list)
        # print("[MAIN] GPU computation completed")
        chephy_matrix = cp.nan_to_num(chephy_matrix)
        chephy_matrix = (chephy_matrix / greatest_distance).astype(cp.float16)
        chephy_matrix = cp.round(chephy_matrix, 3)
        # print("[MAIN] Normalization and rounding completed on GPU")
        # # print("normalized done on GPU")
        chephy_matrix = chephy_matrix.get()  # Transfer back to CPU
        # print("[MAIN] Data transferred back to CPU")



    except Exception as e:
        # print(f"[MAIN] GPU computation failed: {e}")
        # print("[MAIN] Falling back to CPU computation")
        chephy_matrix = compute_chephy_matrix_cpu(sequences_list)
        # print("[MAIN] CPU computation completed")
        np.nan_to_num(chephy_matrix, copy=False)
        chephy_matrix = (chephy_matrix / greatest_distance).astype(np.float16)
        chephy_matrix = np.round(chephy_matrix, 3).astype(np.float16)
        # print("[MAIN] Normalization and rounding completed on CPU")

    # print(f"[MAIN] Final matrix shape: {chephy_matrix.shape}, dtype: {chephy_matrix.dtype}")
    return chephy_matrix, sequences_list


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
