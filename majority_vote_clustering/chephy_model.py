import pandas as pd
from collections import defaultdict
import csv
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

    for seq in neighbors:
        for var in neighbors[seq]:
            couples[seq].append([var, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, var))])

    couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}
    return couples

def write_couples_file(couples, directory, filename):
    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)
    output_file = "{}/{}.json".format(directory, filename)
    with open(output_file, "w") as f:
        ujson.dump(couples, f)