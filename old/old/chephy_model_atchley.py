import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform, hamming
from tqdm import tqdm

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
    atchley_df = pd.read_csv("files/atchley.csv")
    atchley_dict = atchley_df.set_index('amino.acid').to_dict(orient='index')

    sequences_list = list(sequences_set)
    original_sequences = {trunc_seq: list(full_to_trunc_map[trunc_seq]) for trunc_seq in sequences_list}

    # Compute distance matrices
    chephy_matrix = compute_chephy_matrix(sequences_list, atchley_dict)
    ham_matrix = compute_ham_matrix(sequences_list)

    print("Normalizing ChePhy Matrix...")
    np.nan_to_num(chephy_matrix, copy=False)
    normalized_matrix = (chephy_matrix / greatest_distance).astype(np.float32)

    print("Preparing Edge List with Vectorization...")
    
    # **Step 1: Precompute Full Sequences for Fast Mapping**
    full_seqs = sorted(set(seq for seqs in full_to_trunc_map.values() for seq in seqs))
    full_seq_indices = {seq: i for i, seq in enumerate(full_seqs)}

    # **Step 2: Convert Distance Matrix to Edge List in One Step**
    edge_data = []
    seq_map = {trunc_seq: list(full_to_trunc_map[trunc_seq]) for trunc_seq in sequences_list}

    for i in tqdm(range(len(sequences_list)), desc="Processing Sequences"):
        for j in range(i + 1, len(sequences_list)):  # Avoid duplicates
            weight = round(float(normalized_matrix[i, j]), 3)
            hamming_dist = int(ham_matrix[i, j])

            # Precomputed Full Sequence Mapping
            for orig_seq1 in seq_map[sequences_list[i]]:
                for orig_seq2 in seq_map[sequences_list[j]]:
                    edge_data.append((full_seq_indices[orig_seq1], full_seq_indices[orig_seq2], weight, hamming_dist))

    print("Converting to Edge DataFrame...")
    edge_df = pd.DataFrame(edge_data, columns=['source', 'target', 'weight', 'hamming'])

    return edge_df, full_seqs


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
