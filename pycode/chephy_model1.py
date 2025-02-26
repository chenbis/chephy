import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
from scipy.sparse import lil_matrix
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from multiprocessing import Pool, cpu_count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_substitution_matrix_tensor(substitution_matrix):
    sub_matrix_tensor = torch.zeros((128, 128), dtype=torch.float16, device=device)  # ASCII table size
    for (aa1, aa2), value in substitution_matrix.items():
        sub_matrix_tensor[ord(aa1), ord(aa2)] = value
    return sub_matrix_tensor

def compute_pairwise_distances(args):
    """
    Compute a chunk of the distance matrix in parallel.
    """
    sequences, substitution_matrix, i_range = args
    n = len(sequences)
    rows, cols, actual_values, hamming_values = [], [], [], []
    
    for i in tqdm(i_range):
        seq1 = sequences[i]
        for j in range(i + 1, n):
            seq2 = sequences[j]
            actual_dist = sum(substitution_matrix[(c1, c2)] for c1, c2 in zip(seq1, seq2))
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
            rows.append(i)
            cols.append(j)
            actual_values.append(np.float16(actual_dist))
            hamming_values.append(np.uint8(hamming_dist))
    
    return rows, cols, actual_values, hamming_values

def compute_distance_matrices_gpu(sequences, sub_matrix_tensor, batch_size=1024):
    n = len(sequences)
    seq_tensor = torch.tensor([[ord(c) for c in seq] for seq in sequences], dtype=torch.int, device=device)

    actual_matrix = torch.zeros((n, n), dtype=torch.float16, device=device)
    dist_matrix = torch.zeros((n, n), dtype=torch.int16, device=device)

    print("Computing distance matrix on GPU using batch processing...")

    with tqdm(total=n*n, desc="GPU Processing", unit=" comparisons") as pbar:
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            seq1_batch = seq_tensor[i:i_end]

            for j in range(i, n, batch_size):
                j_end = min(j + batch_size, n)
                seq2_batch = seq_tensor[j:j_end]

                # Compute pairwise substitution distances
                actual_dist = torch.sum(sub_matrix_tensor[seq1_batch.unsqueeze(1), seq2_batch.unsqueeze(0)], dim=-1)
                hamming_dist = torch.sum(seq1_batch.unsqueeze(1) != seq2_batch.unsqueeze(0), dim=-1)

                actual_matrix[i:i_end, j:j_end] = actual_dist
                dist_matrix[i:i_end, j:j_end] = hamming_dist

                pbar.update((i_end - i) * (j_end - j))  # Update progress bar

    return actual_matrix.cpu().numpy(), dist_matrix.cpu().numpy()


def compute_distance_matrices_cpu_parallel(sequences, substitution_matrix, num_workers=None):
    """
    Compute distance matrices in parallel using multiprocessing.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Use all but one CPU core
    
    num_workers = 50

    n = len(sequences)
    chunk_size = max(1, n // num_workers)
    index_ranges = [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    
    # Prepare multiprocessing pool
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(compute_pairwise_distances, 
                                      [(sequences, substitution_matrix, r) for r in index_ranges]),
                            total=len(index_ranges), desc="Computing Distance Matrix"))
    
    # Collect results
    all_rows, all_cols, all_actual_values, all_hamming_values = [], [], [], []
    for rows, cols, actual_values, hamming_values in tqdm(results):
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_actual_values.extend(actual_values)
        all_hamming_values.extend(hamming_values)
    
    # Convert to sparse matrices
    actual_matrix = csr_matrix((all_actual_values, (all_rows, all_cols)), shape=(n, n), dtype=np.float32)
    dist_matrix = csr_matrix((all_hamming_values, (all_rows, all_cols)), shape=(n, n), dtype=np.uint8)
    
    return actual_matrix, dist_matrix


def find_che_phy_dist(sequences_set, full_to_trunc_map, substitution_matrix, greatest_distance):
    """
    Compute distance matrices and store results in a Pandas DataFrame with original sequences as indices.
    Save only the upper triangle of the DataFrame to a file.
    """
    sequences_list = list(sequences_set)
    original_sequences = {trunc_seq: list(full_to_trunc_map[trunc_seq]) for trunc_seq in sequences_list}
    expanded_sequences = [seq for trunc_seq in sequences_list for seq in original_sequences[trunc_seq]]

    sub_matrix_tensor = create_substitution_matrix_tensor(substitution_matrix)
    # actual_matrix, dist_matrix = compute_distance_matrices_gpu(list(sequences_set), sub_matrix_tensor)
    actual_matrix, dist_matrix = compute_distance_matrices_cpu_parallel(list(sequences_set), substitution_matrix)
    # if len(sequences_set) < 1000:
    #     print("Using CPU for small dataset...")
    #     actual_matrix, dist_matrix = compute_distance_matrices_cpu(list(sequences_set), substitution_matrix)
    # else:
    #     print("Using GPU for large dataset...")
    #     actual_matrix, dist_matrix = compute_distance_matrices_gpu(list(sequences_set), sub_matrix_tensor)  


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
