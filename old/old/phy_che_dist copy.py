import pandas as pd
import json
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

max_mutations = 8
distances_csv = "distance_matrix.csv"
distances_df = pd.read_csv(distances_csv, index_col=0)
distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
# min_distance = min(distances_dict.values())
# max_distance = max(distances_dict.values())
# normalized_distances_dict = {(aa1, aa2): (distance - min_distance) / (max_distance - min_distance) for (aa1, aa2), distance in distances_dict.items()}

def find_strings_within_distance(strings_set, new_string, max_distance):
    strings_array = np.array(list(strings_set))
    distances = np.array([sum(s1 != s2 for s1, s2 in zip(s, new_string)) for s in strings_array])
    indices = np.where(distances <= max_distance)[0]
    return set(strings_array[indices]) - {new_string}

# def find_strings_within_distance(strings_set, new_string, max_distance):
#     strings_array = np.array(list(strings_set))
#     distances = np.array([normalized_distances_dict.get((s, new_string), 0) for s in strings_array])
#     indices = np.where(distances <= 0.1)[0]
#     return set(strings_array[indices]) - {new_string}


def truncate_sequences(args):
    csv_file, column_name, right, left = args
    with open(csv_file, newline='') as csvfile:
        reader = pd.read_csv(csvfile)
        for sequence in reader[column_name]:
            start_index = max(0, len(sequence) // 2 - left)
            end_index = min(len(sequence), len(sequence) // 2 + right)
            yield sequence[start_index:end_index]

def process_sequence(args):
    seq, sequences_set = args
    return [(var, seq) for var in find_strings_within_distance(sequences_set, seq, max_mutations)]

def main():
    # 1000 sequences
    sequences_file = 'files/forchen_F_26L.csv'
    sequences_header = "cdr3_amino_acid"

    # # 40K sequences
    # sequences_file = 'files/for_chen_B.csv'
    # sequences_header = "CDR3.aa"

    # # 1M sequences
    # sequences_file = "random_sequences.csv"
    # sequences_header = "sequences"
    
    print("truncating sequences")
    sequences_set = set(truncate_sequences((sequences_file, sequences_header, 4, 4)))

    print("creating mutations")
    neighbors = set()
    with Pool() as pool:
        for result in tqdm(pool.imap(process_sequence, [(seq, sequences_set) for seq in sequences_set]), total=len(sequences_set)):
            neighbors.update(result)

    couples = defaultdict(list)
    for mut, seq in neighbors:
        couples[seq].append([mut, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, mut))])

    couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}

    output_file = "couples_{}.json".format("test")
    with open(output_file, "w") as f:
        json.dump(couples, f)

if __name__ == "__main__":
    main()