import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


GREATEST_DIST = 654.9255919999998


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
    distance_matrix = squareform(pdist(sequence_vectors, metric='sqeuclidean'))
    return distance_matrix

def compute_ham_matrix(sequences):
    """
    Compute Hamming distance matrix using integer encoding.
    """

    # Get unique characters and assign an integer to each
    unique_chars = sorted(set("".join(sequences)))  # Extract all unique characters
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}

    # Convert sequences into numerical arrays
    sequence_array = np.array([[char_to_int[char] for char in seq] for seq in sequences])

    # Compute pairwise Hamming distances
    distance_matrix = squareform(pdist(sequence_array, metric='hamming')) * sequence_array.shape[1]

    return distance_matrix

def find_che_phy_dist(sequences_set,greatest_distance=GREATEST_DIST):
    """
    Compute normalized ChePhy distances and Hamming distances.
    """
    atchley_df = pd.read_csv("../files/atchley.csv")
    atchley_dict = atchley_df.set_index('amino.acid').to_dict(orient='index')

    sequences_list = list(sequences_set)

    # Compute distance matrices
    chephy_matrix = compute_chephy_matrix(sequences_list, atchley_dict)
    ham_matrix = compute_ham_matrix(sequences_list)

    np.nan_to_num(chephy_matrix, copy=False)
    normalized_chephy_matrix = (chephy_matrix / greatest_distance).astype(np.float32)

    return normalized_chephy_matrix, ham_matrix, sequences_list


def truncate_sequences(sequences, right=4, left=4):
    """
    Truncate sequences efficiently by precomputing start and end indices and map them back to the original sequences.
    """

    full_to_trunc_map = defaultdict(set)
    sequences_set = set()

    for sequence in sequences:
        if sequence:
            mid = (len(sequence) + 1) // 2
            trunc_seq = sequence[max(0, mid - right):min(len(sequence), mid + left)]
            sequences_set.add(trunc_seq)
            full_to_trunc_map[trunc_seq].add(sequence)

    return sequences_set, full_to_trunc_map
