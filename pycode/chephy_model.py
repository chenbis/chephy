import numpy as np
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool

def compute_distance_pair(args):
    """Helper function for parallel computation of sequence distances."""
    i, j, seq1, seq2, substitution_matrix = args
    actual_distance = sum(substitution_matrix.get((aa1, aa2), 0) for aa1, aa2 in zip(seq1, seq2))
    hamming_distance = sum(aa1 != aa2 for aa1, aa2 in zip(seq1, seq2))
    return i, j, actual_distance, hamming_distance

def compute_distance_matrices_parallel(sequences, substitution_matrix):
    """Optimized distance matrix computation using multiprocessing."""
    n = len(sequences)
    indices = [(i, j, sequences[i], sequences[j], substitution_matrix)
               for i in range(n) for j in range(i+1, n)]  # Avoid duplicate computations

    with Pool() as pool:
        results = list(tqdm(pool.imap(compute_distance_pair, indices), total=len(indices), desc="Computing distances"))

    actual_matrix = np.zeros((n, n))
    dist_matrix = np.zeros((n, n), dtype=int)

    for i, j, actual_dist, hamming_dist in results:
        actual_matrix[i, j] = actual_dist
        dist_matrix[i, j] = hamming_dist
        actual_matrix[j, i] = actual_dist  # Symmetric matrix
        dist_matrix[j, i] = hamming_dist

    return actual_matrix, dist_matrix

def normalize_and_filter(actual_matrix, greatest_distance, dist_matrix, max_dist):
    """Normalize the distance matrix and filter results efficiently."""
    normalized_matrix = np.nan_to_num(actual_matrix / greatest_distance, nan=0.0)
    filtered_indices = np.where(normalized_matrix <= max_dist)
    results = [
        (i, j, normalized_matrix[i, j], dist_matrix[i, j])
        for i, j in zip(*filtered_indices)
        if i != j
    ]
    return results

def find_che_phy_dist(sequences_set, substitution_matrix, greatest_distance, max_dist=1):
    """Find sequence pairs within a normalized distance threshold."""
    sequences_list = list(sequences_set)
    actual_matrix, dist_matrix = compute_distance_matrices_parallel(sequences_list, substitution_matrix)
    filtered_results = normalize_and_filter(actual_matrix, greatest_distance, dist_matrix, max_dist)
    neighbors = defaultdict(dict)
    for i, j, normalized_distance, hamming_distance in tqdm(filtered_results, desc="Processing results"):
        seq1 = sequences_list[i]
        seq2 = sequences_list[j]
        neighbors[seq1][seq2] = {"weight": round(normalized_distance, 3), "dist": hamming_distance}
    return neighbors

def truncate_sequences(sequences, right=4, left=4):
    """Truncate sequences and map full to truncated versions efficiently."""
    full_to_trunc_map = defaultdict(set)
    sequences_set = set()
    for sequence in tqdm(sequences, desc="Truncating sequences"):
        if sequence:
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            trunc_seq = sequence[start_index:end_index]
            sequences_set.add(trunc_seq)
            full_to_trunc_map[trunc_seq].add(sequence)
    return sequences_set, full_to_trunc_map
