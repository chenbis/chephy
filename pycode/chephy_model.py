import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


GREATEST_DIST = 654.925
atchley_path = "/work/sol_lab/files/atchley.csv"
atchley_df = pd.read_csv(atchley_path)
atchley_dict = atchley_df.set_index("amino.acid").to_dict(orient="index")

def compute_chephy_matrix(sequences, atchley_dict):
    """
    Compute distance matrices.
    """
    # Convert sequences into numerical representations
    sequence_vectors = []
    for seq in sequences:
        seq_matrix = np.array([list(atchley_dict[aa].values()) for aa in seq])  # Extract numerical values
        seq_vector = seq_matrix.flatten()  # Flatten to a single vector
        sequence_vectors.append(seq_vector)

    # Compute pairwise sqeuclidean (Euclidean^2) distances
    condensed_distances = pdist(sequence_vectors, metric='sqeuclidean')
    condensed_distances = np.round(condensed_distances, 3)  
    return np.round(condensed_distances, 3).astype(np.float32)

# def compute_ham_matrix(sequences):
#     """
#     Compute condensed Hamming distance matrix using integer encoding.
#     """
#     unique_chars = sorted(set("".join(sequences)))
#     char_to_int = {char: idx for idx, char in enumerate(unique_chars)}

#     sequence_array = np.array([[char_to_int[char] for char in seq] for seq in sequences])

#     # Condensed (upper triangle) Hamming distances
#     condensed_hamming = pdist(sequence_array, metric='hamming') * sequence_array.shape[1]
#     return squareform(np.round(condensed_hamming, 3).astype(np.float32))


def find_che_phy_dist(sequences_set,greatest_distance=GREATEST_DIST):
    """
    Compute normalized condensed ChePhy and Hamming distances.
    """
    sequences_list = list(sequences_set)

    chephy_condensed = compute_chephy_matrix(sequences_list)
    print("chephy done")
    # hamming_matrix = compute_ham_condensed(sequences_list)

    np.nan_to_num(chephy_condensed, copy=False)
    normalized_chephy = (chephy_condensed / greatest_distance).astype(np.float32)
    normalized_chephy = np.round(normalized_chephy, 3).astype(np.float32)
    print("normalized done")
    return squareform(normalized_chephy), sequences_list


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
