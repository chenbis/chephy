import numpy as np
from collections import defaultdict
from tqdm import tqdm

import numpy as np


def compute_distance_matrices(sequences, substitution_matrix):
    """
    Compute actual, max distance, and Hamming distance matrices for a set of sequences.
    """
    n = len(sequences)
    
    # Initialize the matrices
    actual_matrix = np.zeros((n, n))
    dist_matrix = np.zeros((n, n), dtype=int)  # For Hamming distances
    
    for i, seq1 in enumerate(tqdm(sequences, desc="Computing distance matrices")):
        for j, seq2 in enumerate(sequences):
            if i == j:
                continue  # Skip self-comparisons
            
            # Compute actual distance
            actual_matrix[i, j] = sum(
                substitution_matrix[(aa1, aa2)] for aa1, aa2 in zip(seq1, seq2)
            )
            
            # Compute Hamming distance
            dist_matrix[i, j] = sum(aa1 != aa2 for aa1, aa2 in zip(seq1, seq2))
    
    return actual_matrix, dist_matrix



def normalize_and_filter(actual_matrix, greatest_distance, dist_matrix, max_dist):
    """
    Normalize the distance matrix, include Hamming distances, and filter results based on the threshold.
    """

    print("Normalizing results...")
    # Element-wise division to normalize distances
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
        normalized_matrix = np.nan_to_num(actual_matrix / greatest_distance, nan=0.0)

    # Apply the max_dist threshold
    filtered_indices = np.where(normalized_matrix <= max_dist)
    
    # Collect pairs of indices, their normalized distances, and Hamming distances
    results = [
        (i, j, normalized_matrix[i, j], dist_matrix[i, j])
        for i, j in zip(*filtered_indices)
        if i != j  # Exclude self-comparisons
    ]
    return results


def find_che_phy_dist(sequences_set, substitution_matrix, greatest_distance, max_dist=1):
    """
    Find sequence pairs within a normalized distance threshold, including Hamming distance.
    """
    # Convert sequences set to a list for indexing
    sequences_list = list(sequences_set)
    
    # Compute distance matrices
    actual_matrix, dist_matrix = compute_distance_matrices(
        sequences_list, substitution_matrix
    )
    
    # Normalize and filter results
    filtered_results = normalize_and_filter(actual_matrix, greatest_distance, dist_matrix, max_dist)
    
    print("Creating distance dict")
    # Convert filtered results into the required format
    neighbors = defaultdict(dict)
    for i, j, normalized_distance, hamming_distance in filtered_results:
        seq1 = sequences_list[i]
        seq2 = sequences_list[j]
        neighbors[seq1][seq2] = {
            "weight": round(normalized_distance, 3),
            "dist": hamming_distance
        }
    
    return neighbors



def truncate_sequences(sequences, right=4, left=4):

    print("Truncating Sequences...")

    full_to_trunc_map = defaultdict(set)

    sequences_set = set()
    for sequence in sequences:
        if sequence == "":
            continue
        start_index = max(0, len(sequence)//2 - left)
        end_index = min(len(sequence), len(sequence)//2 + right)
        trunc_seq = sequence[start_index:end_index]
        sequences_set.add(trunc_seq)
        full_to_trunc_map[trunc_seq].add(sequence)

    return sequences_set, full_to_trunc_map
