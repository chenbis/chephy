import pandas as pd
import itertools
import hashlib
from tqdm import tqdm
import json
from collections import defaultdict
from multiprocessing import Pool


df = pd.read_csv('files/atchley.csv')
df.set_index('amino.acid', inplace=True)
atchley_features = df.to_dict(orient='index')
for key, value in atchley_features.items():
    atchley_features[key] = list(value.values())


def hash_atchley_sha_int(sequence):

    hash_input = ''
    for aa in sequence:
        hash_input += ''.join(map(str, atchley_features.get(aa)))
    hash_object = hashlib.md5(hash_input.encode())
    hash_digest = hash_object.digest()
    
    # Convert the hexadecimal digest to a decimal number
    hash_number = int.from_bytes(hash_digest, byteorder='big')
    return hash_number

def hash_sequences(mutations_lst, hash_method):
    new_lst = []
    for row in tqdm(mutations_lst, desc="hashing sequences"):
        hash_value = hash_method(row[0])
        new_lst.append([row[0], hash_value, row[2]])
    return new_lst


def generate_mutations(sequence, max_mutations):
    yield sequence, 0, ""
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(1, max_mutations + 1):
        for positions in itertools.combinations(range(len(sequence)), i):
            for new_aas in itertools.product(aa, repeat=i):
                mutated_sequence = list(sequence)
                for pos, new_aa in zip(positions, new_aas):
                    mutated_sequence[pos] = new_aa
                mutated_sequence = "".join(mutated_sequence)
                if mutated_sequence != sequence:
                    yield mutated_sequence, 0, sequence

def process_sequence(seq):
    max_mutations = 2
    return list(generate_mutations(seq, max_mutations))

# def get_distance(seq1, seq2, distances_df):
#     distance = sum([distances_df.loc[aa1, aa2] for aa1, aa2 in zip(seq1, seq2) if aa1 != aa2])
#     return distance


sequence_map = {}

sequences_csv = 'files/forchen_F_26L.csv'
sequences_header = "cdr3_amino_acid"

# sequences_csv = 'files/for_chen_B.csv'
# sequences_header = "CDR3.aa"

# sequences_csv = "../random_sequences.csv"
# sequences_header = "sequences"

mutations_df = pd.DataFrame(columns=['mutation','mutation_hash' 'original'])

sequences_df = pd.read_csv(sequences_csv)
sequences_set = set(sequences_df[sequences_header].unique())
# sequences_hashed = hash_sequences(sequences_set, hash_atchley_sha_int,sequence_map)


# max_mutations = 2
batch_size = 100
mutations_lst = []
sequences_list = list(sequences_set)
for i in tqdm(range(0, len(sequences_list), batch_size), desc="generating mutations"):
    batch_sequences = sequences_list[i:i+batch_size]
    with Pool() as pool:
        mutations_lst.extend(pool.map(process_sequence, batch_sequences))

mutations_lst = list(itertools.chain.from_iterable(mutations_lst))

mutations_lst = hash_sequences(mutations_lst, hash_atchley_sha_int)

mutations_df = pd.DataFrame(mutations_lst, columns=['mut', 'mut_hash', 'org'])
mutations_sorted = mutations_df.sort_values(by='mut_hash')


distances_csv = "distance_matrix.csv"
distances_df = pd.read_csv(distances_csv, index_col=0)
distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}# distances_dict = distances_df.to_dict()

couples = defaultdict(list)
mask_org_empty = mutations_sorted['org'] == ""
duplicates_with_empty_org = mutations_sorted[mutations_sorted.duplicated(['mut_hash'], keep=False) & mask_org_empty]
filtered_df = mutations_sorted[mutations_sorted['mut_hash'].isin(duplicates_with_empty_org['mut_hash'])]

for index, row in tqdm(filtered_df.iterrows(), "finding duplicates"):
    if row["org"] != "":
        mut = row["mut"]
        seq = row["org"]
        couples[seq].append([mut, sum(distances_dict[(aa1, aa2)] for aa1, aa2 in zip(seq, mut))])

couples = {key: sorted(value, key=lambda x: x[1]) for key, value in couples.items()}

output_file = "couples.json"
with open(output_file, "w") as f:
    json.dump(couples, f)
