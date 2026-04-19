import pandas as pd
from collections import defaultdict
import json, csv, time, os
from tqdm import tqdm
from Levenshtein import distance
from multiprocessing import Pool

MUTATIONS_DIR = "mutations"
max_mutations = 3
aa = 'ACDEFGHIKLMNPQRSTVWY'

def find_strings_within_distance(strings_set, new_string, max_distance):
    return {string for string in strings_set if string != new_string and distance(string, new_string) <= max_distance}

def truncate_sequences(args):
    csv_file, column_name, right, left = args
    sequences_set = set()
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[column_name].strip()
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            sequences_set.add(sequence[start_index:end_index])
    return sequences_set

# def delete_mutation_folder():
#     timestamp = str(int(time.time()))
#     sub_folder = os.path.join(MUTATIONS_DIR, timestamp)
#     os.makedirs(sub_folder, exist_ok=True)
#     return sub_folder

def process_sequence(args):
    seq, sequences_set = args
    return [(var, seq) for var in find_strings_within_distance(sequences_set, seq, max_mutations)]


def main():
    total_start_time = time.time()
    # sub_folder = delete_mutation_folder()

    # 1000 sequences
    sequences_file = 'files/forchen_F_26L.csv'
    sequences_header = "cdr3_amino_acid"

    # # 40K sequences
    # sequences_file = 'files/for_chen_B.csv'
    # sequences_header = "CDR3.aa"

    # 1M sequences
    sequences_file = "random_sequences.csv"
    sequences_header = "sequences"

    sequences_file = 'files/articles/mels.csv'
    sequences_header = "mel7.cdr3a"

    print("truncating sequences")
    start_time = time.time()

    sequences_set = truncate_sequences((sequences_file, sequences_header, 4, 4))

    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    print("creating mutations")
    start_time = time.time()

    with Pool() as pool:
        results = list(tqdm(pool.imap(process_sequence, [(seq, sequences_set) for seq in sequences_set]), total=len(sequences_set)))

    neighbors = {item for sublist in results for item in sublist}


    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    
    couples = defaultdict(list)

    for mut, seq in neighbors:
        couples[seq].append([mut, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, mut))])

    couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}

    output_file = "couples_{}.json".format("test")
    with open(output_file, "w") as f:
        json.dump(couples, f)

    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    total_end_time = time.time()
    print("total time:", (total_end_time - total_start_time)/60, "minutes")

if __name__ == "__main__":
    main()