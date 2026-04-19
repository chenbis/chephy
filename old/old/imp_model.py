import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import ujson

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
        return [(path, current_diff)] if current_diff <= max_diff else []
    
    matching_sequences = []
    for char, subtree in tree.items():
        new_diff = current_diff + (1 if (depth < len(sequence) and char != sequence[depth]) else 0)
        matching_sequences.extend(traverse_tree(subtree, sequence, max_diff, path + char, new_diff, depth + 1))

    return matching_sequences

def find_sequences_within_distance(tree, sequence, max_diff):
    return traverse_tree(tree, sequence, max_diff)

def truncate_sequences(sequences, right=4, left=4):

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


def make_symmetric_neighbors(neighbors):
    combined = defaultdict(set, neighbors)
    for key, values in neighbors.items():
        for seq, value in values:
            combined[seq].add((key, value))  # Add the inverted edge
    return combined


def find_che_phy_dist(sequences_set, max_mutations, max_dist):
    neighbors = {}
    tree = create_tree(sequences_set)  # Create the tree only once

    # Precompute worst-case distances and substitution costs
    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    max_substitution_costs = {aa: distances_df.loc[aa].max() for aa in distances_df.index}
    worst_case_distances = {seq: sum(max_substitution_costs[aa] for aa in seq) for seq in sequences_set}

    # Use tqdm for progress tracking
    for seq in tqdm(sequences_set):
        results = find_sequences_within_distance(tree, seq, max_mutations)
        results.remove((seq, 0))  # Remove self-match
        if results:
            neighbors[seq] = set(results)

    neighbors = make_symmetric_neighbors(neighbors)  # Symmetrize neighbors

    # Process neighbors
    couples = defaultdict(dict)  # Changed from list to dict
    for seq in tqdm(neighbors):
        seq_neighbors = neighbors[seq]
        worst_case_distance = worst_case_distances[seq]

        for neighbor_seq, diff in seq_neighbors:
            # Calculate the actual normalized distance
            actual_distance = sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, neighbor_seq))
            normalized_distance = actual_distance / worst_case_distance if worst_case_distance > 0 else 0

            if normalized_distance <= max_dist:
                # Store as a dictionary for networkx compatibility
                couples[seq][neighbor_seq] = {
                    "weight": round(normalized_distance, 3),
                    "dist": diff
                }

    # Sort and return (sorting is not strictly necessary for dicts)
    return couples



def write_couples_file(couples, directory, filename):
    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)
    output_file = f"{directory}/{filename}.json"
    with open(output_file, "w") as f:
        ujson.dump(couples, f)



def get_worst_case_distance(seq, substitution_matrix):
    max_distance = 0
    for aa in seq:
        # For each amino acid, find the maximum possible substitution cost
        max_substitution_cost = substitution_matrix[aa].max()
        max_distance += max_substitution_cost
    return max_distance