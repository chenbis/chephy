import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def compute_distance_matrices(sequences, substitution_matrix):
    """
    Compute actual, max distance, and Hamming distance matrices for a set of sequences.
    Optimized using NumPy array operations.
    """
    n = len(sequences)
    actual_matrix = np.zeros((n, n), dtype=np.float32)
    dist_matrix = np.zeros((n, n), dtype=np.int32)

    # Convert sequences to NumPy character arrays
    seq_array = np.array([list(seq) for seq in sequences], dtype="U1")

    # Convert substitution matrix to a NumPy-friendly lookup
    sub_keys = np.array(["".join(k) for k in substitution_matrix.keys()], dtype="U2")
    sub_values = np.array(list(substitution_matrix.values()), dtype=np.float32)
    sorted_indices = np.argsort(sub_keys)
    sub_keys, sub_values = sub_keys[sorted_indices], sub_values[sorted_indices]

    def lookup_substitution_fixed_v2(aa1, aa2):
        """Efficiently retrieve substitution matrix values."""
        key = np.char.add(aa1, aa2).astype(str)  # Ensure key is a 1D array of strings
        key_flat = key.flatten()
        indices = np.searchsorted(sub_keys, key_flat)
        return sub_values[indices].reshape(aa1.shape)  # Reshape to match sequence structure

    for i in tqdm(range(n), desc="Computing distance matrices"):
        seq1 = seq_array[i]
        for j in range(i + 1, n):
            seq2 = seq_array[j]

            # Vectorized lookup in the substitution matrix
            actual_dist = lookup_substitution_fixed_v2(seq1, seq2).sum()
            actual_matrix[i, j] = actual_matrix[j, i] = actual_dist

            # Compute Hamming distance using NumPy
            dist_matrix[i, j] = dist_matrix[j, i] = np.count_nonzero(seq1 != seq2)

    return actual_matrix, dist_matrix

def find_che_phy_dist(sequences_set, full_to_trunc_map, substitution_matrix, greatest_distance):
    """
    Compute distance matrices and store results in a Pandas DataFrame with original sequences as indices.
    Save only the upper triangle of the DataFrame to a file.
    """
    sequences_list = list(sequences_set)
    original_sequences = {trunc_seq: list(full_to_trunc_map[trunc_seq]) for trunc_seq in sequences_list}
    expanded_sequences = [seq for trunc_seq in sequences_list for seq in original_sequences[trunc_seq]]

    # Compute distance matrices
    actual_matrix, dist_matrix = compute_distance_matrices(sequences_list, substitution_matrix)

    # Normalize distances
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle division by zero
        normalized_matrix = np.nan_to_num(actual_matrix / float(greatest_distance), nan=0.0).astype(np.float32)

    # Create DataFrame to store results
    df = pd.DataFrame(index=expanded_sequences, columns=expanded_sequences, dtype=object)

    for i in range(len(sequences_list)):
        for j in range(i + 1, len(sequences_list)):
            seq1, seq2 = sequences_list[i], sequences_list[j]
            distance_info = [round(float(normalized_matrix[i, j]), 3), int(dist_matrix[i, j])]
            for orig_seq1 in original_sequences[seq1]:
                for orig_seq2 in original_sequences[seq2]:
                    df.at[orig_seq1, orig_seq2] = distance_info

    return df

def truncate_sequences(sequences, right=4, left=4):
    """
    Truncate sequences efficiently by precomputing start and end indices and map them back to the original sequences.
    """
    print("Truncating Sequences...")

    full_to_trunc_map = defaultdict(set)
    sequences_set = set()

    for sequence in sequences:
        if sequence:
            mid = len(sequence) // 2
            trunc_seq = sequence[max(0, mid - left):min(len(sequence), mid + right)]
            sequences_set.add(trunc_seq)
            full_to_trunc_map[trunc_seq].add(sequence)

    return sequences_set, full_to_trunc_map
