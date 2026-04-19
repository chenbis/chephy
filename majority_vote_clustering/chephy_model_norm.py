import pandas as pd
from collections import defaultdict
import csv
from tqdm import tqdm
from pathlib import Path
import ujson
import numpy as np

def create_tree(sequences):
    tree = {}
    for sequence in sequences:
        current_dict = tree
        for letter in sequence:
            current_dict = current_dict.setdefault(letter, {})
    return tree

def traverse_tree(tree, sequence, max_diff, path="", current_diff=0, depth=0):
    if current_diff > max_diff:
        return []
    
    if not tree:
        return [path] if current_diff <= max_diff else []
    
    matching_sequences = []
    for char, subtree in tree.items():
        new_diff = current_diff + (1 if (depth < len(sequence) and char != sequence[depth]) else 0)
        matching_sequences.extend(traverse_tree(subtree, sequence, max_diff, path + char, new_diff, depth + 1))

    return matching_sequences

def find_sequences_within_distance(tree, sequence, max_diff):
    return traverse_tree(tree, sequence, max_diff)

def truncate_sequences(sequences, right=4, left=4):

    full_to_trunc_map = defaultdict(list)

    sequences_set = set()
    for sequence in sequences:
        if sequence == "":
            continue
        start_index = max(0, len(sequence)//2 - left)
        end_index = min(len(sequence), len(sequence)//2 + right)
        trunc_seq = sequence[start_index:end_index]
        sequences_set.add(trunc_seq)
        full_to_trunc_map[trunc_seq].append(sequence)

    return sequences_set, full_to_trunc_map

def invert_dict(d): 
    inverse = dict() 
    for key in d: 
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse: 
                # If not create a new list
                inverse[item] = {key} 
            else: 
                inverse[item].add(key) 
    return inverse
def load_distance_matrix(distances_csv="distance_matrix.csv"):
    distances_df = pd.read_csv(distances_csv, index_col=0)

    # Build fast lookup dict
    distances_dict = {
        (aa1, aa2): float(distances_df.loc[aa1, aa2])
        for aa1 in distances_df.index
        for aa2 in distances_df.columns
    }

    # Global scale: percentile of OFF-diagonal entries
    mat = distances_df.to_numpy(dtype=float)
    off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
    scale_C = float(np.percentile(off_diag, 95))  # you can try 90/95/99

    return distances_df, distances_dict, scale_C

def find_che_phy_dist(sequences_set, max_mutations, max_dist=1,
                      distances_csv="distance_matrix.csv",
                      percentile_scale=95):
    """
    finds chemical physical distance between all sequences in the set
    sequences set - set of sequences to compare
    max_mutations - maximum number of different amino acids between two close sequences
    """

    neighbors = {}
    tree = create_tree(sequences_set)
    for seq in tqdm(sequences_set):
        results = find_sequences_within_distance(tree, seq, max_mutations)
        results.remove(seq)
        if results:
            neighbors[seq] = set(results)
    
    
    inverted = invert_dict(neighbors)
    neighbors = defaultdict(set, {k: neighbors.get(k, set()) | inverted.get(k, set())\
                                   for k in set(neighbors) | set(inverted)})
    

    distances_csv = "/home/dsi/chenbis/repos/sol_lab/files/distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    
    couples = defaultdict(list)

    mat = distances_df.to_numpy(dtype=float)
    off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
    C = float(np.percentile(off_diag, percentile_scale))

    # 3) compute normalized distances for edges
    couples = defaultdict(list)

    for seq in tqdm(neighbors, desc="Distance scoring"):
        L = len(seq)
        denom = L * C if L > 0 and C > 0 else 1.0

        for var in neighbors[seq]:
            # sum per-position substitution costs
            actual_distance = sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, var))

            normalized_distance = actual_distance / denom
            # hard bound to [0,1] for ML
            if normalized_distance < 0:
                normalized_distance = 0.0
            elif normalized_distance > 1:
                normalized_distance = 1.0

            normalized_distance = float(f"{normalized_distance:.3f}")

            if normalized_distance <= max_dist:
                couples[seq].append([var, normalized_distance])

    couples = {k: sorted(v, key=lambda x: x[1]) for k, v in couples.items()}
    return couples

def write_couples_file(couples, directory, filename):
    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)
    output_file = "{}/{}.json".format(directory, filename)
    with open(output_file, "w") as f:
        ujson.dump(couples, f)