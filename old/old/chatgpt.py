import pandas as pd
import itertools
from collections import defaultdict
import json, csv, time
from tqdm import tqdm

def generate_mutations(sequence, max_mutations, sequences_set):
    yield sequence, ""
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    # processed = set()
    for i in range(1, max_mutations + 1):
        for positions in itertools.combinations(range(len(sequence)), i):
            for new_aas in itertools.product(aa, repeat=i):
                mutated_sequence = list(sequence)
                for pos, new_aa in zip(positions, new_aas):
                    mutated_sequence[pos] = new_aa
                mutated_sequence = "".join(mutated_sequence)
                
                if mutated_sequence != sequence and mutated_sequence in sequences_set:
                    # processed.add(mutated_sequence)
                    yield mutated_sequence, sequence

def truncate_sequences(csv_file, column_name, right, left):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[column_name].strip()
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            yield sequence[start_index:end_index]

def main():
    total_start_time = time.time()
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
    start_time = time.time()
    sequences_set = set(truncate_sequences(sequences_file, sequences_header, 4, 4))
    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    max_mutations = 2
    mutations_file = "mutations.csv"

    print("creating mutations")
    start_time = time.time()
    with open(mutations_file, "w") as f:
        for seq in tqdm(sequences_set):
            for mut, org in generate_mutations(seq, max_mutations, sequences_set):
                f.write(f"{mut},{org}\n")
                # f.flush()

    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    print("conveting to pandas")
    start_time = time.time()
    mutations_df = pd.read_csv(mutations_file, names=['mut', 'org'], na_filter=False)
    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    
    print("finding duplicates")
    start_time = time.time()

    couples = defaultdict(list)
    mask_org_empty = mutations_df['org'] == ""
    duplicates_with_empty_org = mutations_df[mutations_df.duplicated(['mut'], keep=False) & mask_org_empty]
    filtered_df = mutations_df[mutations_df['mut'].isin(duplicates_with_empty_org['mut'])]

    for _, row in filtered_df.iterrows():
        if row["org"] != "":
            mut = row["mut"]
            seq = row["org"]
            couples[seq].append([mut, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, mut))])
    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}

    output_file = "couples.json"
    with open(output_file, "w") as f:
        json.dump(couples, f)

    total_end_time = time.time()
    print("total time:", (total_end_time - total_start_time)/60, "minutes")

if __name__ == "__main__":
    main()
