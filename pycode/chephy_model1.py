import numpy as np
from collections import defaultdict
from tqdm import tqdm

def compute_distance_matrices(sequences, substitution_matrix):
    """
    Compute actual, max distance, and Hamming distance matrices for a set of sequences.
    Optimized using NumPy array operations.
    """
    n = len(sequences)
    actual_matrix = np.zeros((n, n))
    dist_matrix = np.zeros((n, n), dtype=int)

    # Convert sequences to NumPy arrays
    seq_array = np.array([list(seq) for seq in sequences], dtype="U1")

    for i in tqdm(range(n), desc="Computing distance matrices"):
        seq1 = seq_array[i]
        for j in range(i + 1, n):
            seq2 = seq_array[j]

            # Vectorized substitution matrix lookup
            actual_matrix[i, j] = actual_matrix[j, i] = sum(
                substitution_matrix.get((a1, a2), 0) for a1, a2 in zip(seq1, seq2)
            )

            # Compute Hamming distance using NumPy
            dist_matrix[i, j] = dist_matrix[j, i] = np.count_nonzero(seq1 != seq2)

    return actual_matrix, dist_matrix



def find_che_phy_dist(sequences_set, substitution_matrix, greatest_distance, max_dist=1):
    """
    Find sequence pairs within a normalized distance threshold, including Hamming distance.
    """
    sequences_list = list(sequences_set)

    # Compute distance matrices
    actual_matrix, dist_matrix = compute_distance_matrices(sequences_list, substitution_matrix)

    print("Creating distance dict")
    neighbors = defaultdict(dict)

    for i in range(len(sequences_list)):
        for j in range(i + 1, len(sequences_list)):
            seq1, seq2 = sequences_list[i], sequences_list[j]
            normalized_distance = actual_matrix[i, j] / greatest_distance if greatest_distance != 0 else 0

            neighbors[seq1][seq2] = {
                "weight": round(normalized_distance, 3),
                "dist": dist_matrix[i, j]
            }

    return neighbors
def truncate_sequences(sequences, right=4, left=4):
    """
    Truncate sequences efficiently by precomputing start and end indices.
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
