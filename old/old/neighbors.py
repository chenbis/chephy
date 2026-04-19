import json
import pandas as pd
from tqdm import tqdm

# sorts amino acids couples according to the distance df

def get_sorted_substitutions(distances_df):
    substitutions = []
    amino_acids = distances_df.index

    for source_acid in amino_acids:
        for target_acid in amino_acids:
            if source_acid != target_acid:
                distance = distances_df.loc[source_acid, target_acid]
                substitutions.append((source_acid, target_acid, distance))

    return sorted(substitutions, key=lambda x: x[2])

def seq_to_dict(seq):
    result_dict = {}
    for idx, char in enumerate(seq):
        if char in result_dict:
            result_dict[char].append(idx)
        else:
            result_dict[char] = [idx]
    return result_dict

def find_couples(sequences_set, sorted_amino, max_neighbors=1):
    couples = {}
    for seq in tqdm(sequences_set, desc="Progress", unit=" sequence"):
        dict_seq = seq_to_dict(seq)
        couples[seq] = []
        flag = 0
        for sub in sorted_amino:
            if sub[0] not in dict_seq:
                continue

            if flag < max_neighbors:
                for occ in dict_seq[sub[0]]:
                    seq_to_search = seq[:occ] + sub[1] + seq[occ + 1:]
                    distance = sub[2]
                    if seq_to_search in sequences_set:
                        couples[seq].append([seq_to_search, distance])
                        flag += 1
                        if flag < max_neighbors:
                            break
            else:
                break

    return couples



# read distance matrix
distances_csv = "distance_matrix.csv"

sequences_csv = 'files/forchen_F_26L.csv'
sequences_header = "cdr3_amino_acid"

# sequences_csv = 'files/for_chen_B.csv'
# sequences_header = "CDR3.aa"

# sequences_csv = "../random_sequences.csv"
# sequences_header = "sequences"


distances_df = pd.read_csv(distances_csv, index_col=0)

sorted_amino = get_sorted_substitutions(distances_df)

sequences_df = pd.read_csv(sequences_csv)

sequences = sequences_df[sequences_header].tolist()
sequences_set = set(sequences)

# sequences_set.add("CASSLALAGGTDTQYI")
# sequences_set.add("CASSLALAGGTDTQYC")

sorted_amino = get_sorted_substitutions(distances_df)
couples = find_couples(sequences_set, sorted_amino, max_neighbors=3)

couples_list = {key: list(value) for key, value in couples.items()}

# Serialize data into file without escape characters:
with open("couples.json", 'w') as outfile:
    json.dump(couples_list, outfile, indent=4)