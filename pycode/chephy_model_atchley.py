import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform, hamming

GREATEST_DIST =  25.59151406228244


def compute_chephy_matrix(sequences, atchley_dict):
    """
    Compute distance matrices in parallel using multiprocessing.
    """
    print("Calculating chephy")
    # Convert sequences into numerical representations
    sequence_vectors = []
    for seq in sequences:
        seq_matrix = np.array([list(atchley_dict[aa].values()) for aa in seq])  # Extract numerical values
        seq_vector = seq_matrix.flatten()  # Flatten to a single vector
        sequence_vectors.append(seq_vector)

    # Compute pairwise Euclidean distances
    distance_matrix = squareform(pdist(sequence_vectors, metric='euclidean'))
    return distance_matrix

def compute_ham_matrix(sequences):
    """
    Compute Hamming distance matrix using integer encoding.
    """
    print("Calculating Hamming Matrix...")

    # Get unique characters and assign an integer to each
    unique_chars = sorted(set("".join(sequences)))  # Extract all unique characters
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}

    # Convert sequences into numerical arrays
    sequence_array = np.array([[char_to_int[char] for char in seq] for seq in sequences])

    # Compute pairwise Hamming distances
    distance_matrix = squareform(pdist(sequence_array, metric='hamming')) * sequence_array.shape[1]

    return distance_matrix



def find_che_phy_dist(sequences_set, full_to_trunc_map, a,greatest_distance=GREATEST_DIST):
    """
    Compute normalized ChePhy distances and Hamming distances.
    """
    print("Loading Atchley Factors...")
    atchley_df = pd.read_csv("../files/atchley.csv")
    atchley_dict = atchley_df.set_index('amino.acid').to_dict(orient='index')

    sequences_list = list(sequences_set)
    original_sequences = {trunc_seq: list(full_to_trunc_map[trunc_seq]) for trunc_seq in sequences_list}

    # Compute distance matrices
    chephy_matrix = compute_chephy_matrix(sequences_list, atchley_dict)
    ham_matrix = compute_ham_matrix(sequences_list)

    print("Normalizing ChePhy Matrix...")
    normalized_matrix = np.nan_to_num(chephy_matrix / greatest_distance, nan=0.0).astype(np.float32)
    print("Precomputing Full Sequence Mappings...")

    # **Step 1: Create a mapping dictionary for fast lookup**
    pairwise_mapping = {}
    for i, trunc_seq1 in enumerate(sequences_list):
        for j, trunc_seq2 in enumerate(sequences_list):
            if i < j:  # Avoid redundant calculations
                dist_info = (round(float(normalized_matrix[i, j]), 3), int(ham_matrix[i, j]))

                # Store mapping of full-sequence pairs
                for orig_seq1 in original_sequences[trunc_seq1]:
                    for orig_seq2 in original_sequences[trunc_seq2]:
                        pairwise_mapping[(orig_seq1, orig_seq2)] = dist_info
                        pairwise_mapping[(orig_seq2, orig_seq1)] = dist_info  # Ensure symmetry

    print("Building DataFrame...")

    # **Step 2: Convert dictionary to Pandas DataFrame efficiently**
    all_full_sequences = sorted(set(seq for seqs in original_sequences.values() for seq in seqs))
    df = pd.DataFrame(index=all_full_sequences, columns=all_full_sequences, dtype=object)

    # Convert mapping to a DataFrame format (avoids iterating over DataFrame)
    df = df.applymap(lambda _: None)  # Initialize with None to save memory
    df.update(pd.Series(pairwise_mapping).unstack())  # Fast update

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
