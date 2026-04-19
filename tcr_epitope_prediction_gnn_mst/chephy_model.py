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

    aa_order = sorted(atchley_dict.keys())
    aa_vecs = cp.array([list(atchley_dict[aa].values()) for aa in aa_order])
    aa_to_index = {aa: i for i, aa in enumerate(aa_order)}

    seq_indices = [[aa_to_index[aa] for aa in seq] for seq in sequences]
    max_len = max(len(seq) for seq in seq_indices)
    padded = cp.zeros((len(sequences), max_len), dtype=cp.int32)

    for i, seq in enumerate(seq_indices):
        padded[i, :len(seq)] = cp.array(seq)

    # Convert to Atchley feature vectors
    features = aa_vecs[padded]  # (N, L, 5)
    vectors = features.reshape(len(sequences), -1)  # (N, L*5)
    # Compute distances
    condensed = pdist(vectors, metric='sqeuclidean')
    # full_matrix = condensed_to_squareform(condensed, len(sequences))
    # print("squareform done on GPU")
    return cp.round(condensed, 3).astype(cp.float32)


def compute_chephy_matrix_cpu(sequences):
    """
    Compute distance matrices.
    """
    aa_order = sorted(atchley_dict.keys())
    aa_vecs = np.array([list(atchley_dict[aa].values()) for aa in aa_order])
    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}

    # Convert sequences to index matrix
    max_len = max(len(seq) for seq in sequences)
    index_matrix = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        index_matrix[i, :len(seq)] = [aa_to_index[aa] for aa in seq]

    # Lookup Atchley vectors and flatten
    feature_matrix = aa_vecs[index_matrix]  # shape (N, L, 5)
    sequence_vectors = feature_matrix.reshape(len(sequences), -1)
    # Compute pairwise distances
    condensed = pdist(sequence_vectors, metric='sqeuclidean')
    return np.round(condensed, 3).astype(np.float32)


def find_che_phy_dist(sequences_set,greatest_distance=GREATEST_DIST):
    """
    Compute normalized condensed ChePhy and Hamming distances.
    """

    sequences_list = list(sequences_set)

    chephy_matrix = None
    try:
        import cupy as cp
        _ = cp.array([0.0])
        chephy_matrix = compute_chephy_matrix_gpu(sequences_list)
        chephy_matrix = cp.nan_to_num(chephy_matrix)
        chephy_matrix = (chephy_matrix / greatest_distance).astype(cp.float32)
        chephy_matrix = cp.round(chephy_matrix, 3)
        chephy_matrix = chephy_matrix.get()  # Transfer back to CPU

        
        # return normalized.get(), sequences_list


    except Exception:
        chephy_matrix = compute_chephy_matrix_cpu(sequences_list)
        np.nan_to_num(chephy_matrix, copy=False)
        chephy_matrix = (chephy_matrix / greatest_distance).astype(np.float32)
        chephy_matrix = np.round(chephy_matrix, 3).astype(np.float32)

    return squareform(chephy_matrix), sequences_list


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
