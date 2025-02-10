import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import ujson

GREATEST_DISTANCE = 72.38373253708322
def create_tree(sequences):
    tree = {}
    for sequence in sequences:
        current_dict = tree
        for letter in sequence:
            current_dict = current_dict.setdefault(letter, {})
    return tree

def traverse_tree(tree, sequence, substitution_matrix, max_diff=8):
    stack = [(tree, "", 0, 0, 0)]  # (current_node, path, current_diff, chephy_dist, depth)
    results = []

    while stack:
        node, path, current_diff, chephy_dist, depth = stack.pop()

        if current_diff > max_diff:
            continue

        if not node:
            normalized_chephy_dist = round(chephy_dist / GREATEST_DISTANCE, 3)
            results.append((path, current_diff, normalized_chephy_dist))
            continue

        if depth < len(sequence):
            seq_char = sequence[depth]
            for char, subtree in node.items():
                substitution_cost = substitution_matrix.get((seq_char, char), float('inf'))
                new_diff = current_diff + (1 if char != seq_char else 0)
                new_chephy_dist = chephy_dist + (substitution_cost if char != seq_char else 0)
                stack.append((subtree, path + char, new_diff, new_chephy_dist, depth + 1))

    return results



def find_che_phy_dist(sequences_set, full_to_trunc_map, substitution_matrix):
    tree = create_tree(sequences_set)  # Create the tree once
    matrix = {}

    # Compute distances for truncated sequences
    for trunc_seq in tqdm(sequences_set):
        matrix[trunc_seq] = {}
        results = traverse_tree(tree, trunc_seq, substitution_matrix)
        
        for result in results:
            matrix[trunc_seq][result[0]] = [result[2], result[1]]

    # Expand matrix to full sequences
    expanded_matrix = {}

    for trunc_seq, full_seqs in full_to_trunc_map.items():
        if trunc_seq in matrix:
            for full_seq in full_seqs:
                expanded_matrix[full_seq] = {}
                for trunc_neighbor, values in matrix[trunc_seq].items():
                    if trunc_neighbor in full_to_trunc_map:
                        for full_neighbor in full_to_trunc_map[trunc_neighbor]:
                            expanded_matrix[full_seq][full_neighbor] = values

    lower_triangle_expanded_matrix = {}
    for seq1 in expanded_matrix:
        for seq2 in expanded_matrix[seq1]:
            if seq1 > seq2:
                lower_triangle_expanded_matrix.setdefault(seq1, {})[seq2] = expanded_matrix[seq1][seq2]
    
    df = pd.DataFrame.from_dict(lower_triangle_expanded_matrix, orient='index')
    return df

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