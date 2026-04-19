import pandas as pd
from collections import defaultdict
import json, csv, os
from tqdm import tqdm
from Levenshtein import distance
from multiprocessing import Pool
from pathlib import Path
import ujson

def find_strings_within_distance(sequences_set, seq, limit):

    """
    returns hamming distance between two sequences, stops when reaches limit of subtitutions

    todo: I need to think of a way to improve this (maybe with mojo?)

    """

    def hamming_distance(seq1, seq2):
        distance = 0
        for c1, c2 in zip(seq1, seq2):
            if c1 != c2:
                distance += 1
                if distance > limit:
                    return False
        return distance
    close = set()
    for var in sequences_set:
        if hamming_distance(var, seq):
            close.add(var)
    return close
    
# def find_strings_within_distance(strings_set, new_string, max_distance):
#     return {string for string in strings_set if string != new_string and distance(string, new_string) <= max_distance}

def truncate_sequences(csv_file, column_name, right, left):

    sequences_set = set()
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[column_name].strip()
            if sequence == "":
                continue
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            sequences_set.add(sequence[start_index:end_index])
            
    return sequences_set

# def process_sequence(args):
#     seq, sequences_set = args
#     return [(var, seq) for var in find_strings_within_distance(sequences_set, seq, max_mutations)]

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

def find_che_phy_dist(sequences_set, max_mutations):
    """
    finds chemical physical distance between all sequences in the set
    sequences set - set of sequences to compare
    max_mutations - maximum number of different amino acids between two close sequences
    """

    # with Pool() as pool:
    #     results = list(tqdm(pool.imap(process_sequence, [(seq, sequences_set) for seq in sequences_set]), total=len(sequences_set)))

    neighbors = {}
    sequences_set_copy = sequences_set.copy()
    for seq in tqdm(sequences_set):
        sequences_set_copy.remove(seq)
        results = find_strings_within_distance(sequences_set_copy, seq, max_mutations)
        if results:
            neighbors[seq] = results
    
    
    inverted = invert_dict(neighbors)
    neighbors = defaultdict(set, {k: neighbors.get(k, set()) | inverted.get(k, set())\
                                   for k in set(neighbors) | set(inverted)})
    

    distances_csv = "distance_matrix.csv"
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

def main():

    max_mutations = 4
    out_folder = "output_files/tests"
    out_name = "che_phy"

    # 1000 sequences
    sequences_file = 'files/forchen_F_26L.csv'
    sequences_headers = ["cdr3_amino_acid"]

    # 40K sequences
    sequences_file = 'files/for_chen_B2.csv'
    sequences_headers = ["CDR3.aa"]

    # 1M sequences
    sequences_file = "random_sequences.csv"
    sequences_headers = ["sequences"]

    # sequences_file = 'files/articles/neo.csv'
    # sequences_headers = ["Melan.A", "MAGE.A3", "GP100","MAGE.A10", "PHLPP2", "HHAT", "ZCCHC11",\
    #                      "SCL25A48", "MUC1", "MMP9", "UTP20", "Influenza.A.PB1",\
    #                         "EBV.BMLF1","hCMV.pp65","hCMV.pp65.2"]


    for header in sequences_headers:
        sequences_set = truncate_sequences(sequences_file, header, 4, 4)
        couples = find_che_phy_dist(sequences_set, max_mutations)
        write_couples_file(couples, "{}/{}".format(out_folder, header), out_name)

if __name__ == "__main__":
    main()