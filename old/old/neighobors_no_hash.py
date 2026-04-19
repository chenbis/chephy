import pandas as pd
import itertools
from tqdm import tqdm
from collections import defaultdict
import json

def generate_mutations(sequence, max_mutations):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(1, max_mutations + 1):
        for positions in itertools.combinations(range(len(sequence)), i):
            for new_aas in itertools.product(aa, repeat=i):
                mutated_sequence = list(sequence)
                for pos, new_aa in zip(positions, new_aas):
                    mutated_sequence[pos] = new_aa
                mutated_sequence = "".join(mutated_sequence)
                if mutated_sequence != sequence:
                    yield mutated_sequence, sequence



def truncate_sequences(sequences, right, left):
    middle_strings = []
    for sequence in tqdm(sequences, "truncating sequences"):
        start_index = max(0, len(sequence)//2 - left)
        end_index = min(len(sequence)//2 + right, len(sequence))
        middle_strings.append(sequence[start_index:end_index])
    return middle_strings


def main():

    # 1000 sequences
    sequences_csv = 'files/forchen_F_26L.csv'
    sequences_header = "cdr3_amino_acid"

    # # 40K sequences
    # sequences_csv = '../files/for_chen_B.csv'
    # sequences_header = "CDR3.aa"

    # # 1M sequences
    # sequences_csv = "random_sequences.csv"
    # sequences_header = "sequences"

    sequences_df = pd.read_csv(sequences_csv)
    sequences_set = set(sequences_df[sequences_header].unique())
    sequences_set = set(truncate_sequences(sequences_set, 4, 4))


    max_mutations = 3

    mutations_lst = []

    for seq in tqdm(sequences_set, desc="generating mutations"):
        mutations_lst.extend(generate_mutations(seq, max_mutations))

    mutations_df = pd.DataFrame(mutations_lst, columns=['mut', 'org'])


    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}

    couples = defaultdict(list)
    mask_org_empty = mutations_df['org'] == ""
    duplicates_with_empty_org = mutations_df[mutations_df.duplicated(['mut'], keep=False) & mask_org_empty]
    filtered_df = mutations_df[mutations_df['mut'].isin(duplicates_with_empty_org['mut'])]

    for index, row in tqdm(list(filtered_df.iterrows()), "finding duplicates"):
        if row["org"] != "":
            mut = row["mut"]
            seq = row["org"]
            couples[seq].append([mut, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, mut))])

    couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}

    output_file = "couples.json"
    with open(output_file, "w") as f:
        json.dump(couples, f)


if __name__ == "__main__":
    main()