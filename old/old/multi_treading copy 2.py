import pandas as pd
import itertools
from collections import defaultdict
import json, csv, time, os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
 
MUTATIONS_DIR = "mutations"
max_mutations = 3

# def generate_mutations(sequence, max_mutations, sequences_set: set, sub_folder):
#     aa = 'ACDEFGHIKLMNPQRSTVWY'
#     mutations = set()
#     for i in range(1, max_mutations + 1):
#         for positions in itertools.combinations(range(len(sequence)), i):
#             for new_aas in itertools.product(aa, repeat=i):
#                 mutated_sequence = list(sequence)
#                 for pos, new_aa in zip(positions, new_aas):
#                     mutated_sequence[pos] = new_aa
#                 mutated_sequence = "".join(mutated_sequence)
#                 mutations.add(mutated_sequence)

#     matches = list(mutations.intersection(sequences_set))
#     matches.remove(sequence)

#     if matches:
#         output_file = os.path.join(sub_folder, f"{sequence}.csv")
#         with open(output_file, "a") as f:
#             for mutated_sequence in matches:
#                 f.write(f"{mutated_sequence},{sequence}\n")

def generate_mutations(sequence, max_mutations, sequences_set: set, sub_folder):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    mutations = set()
    for i in range(1, max_mutations + 1):
        for positions in itertools.combinations(range(len(sequence)), i):
            for new_aas in itertools.product(aa, repeat=i):
                mutated_sequence = list(sequence)
                for pos, new_aa in zip(positions, new_aas):
                    mutated_sequence[pos] = new_aa
                mutated_sequence = "".join(mutated_sequence)
                if mutated_sequence in sequences_set and mutated_sequence != sequence:
                    mutations.add(mutated_sequence)
    if mutations:
        output_file = os.path.join(sub_folder, f"{sequence}.csv")
        for mutated_sequence in list(mutations):
            with open(output_file, "a") as f:
                f.write(f"{mutated_sequence},{sequence}\n")

def truncate_sequences(csv_file, column_name, right, left):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[column_name].strip()
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            yield sequence[start_index:end_index]

def process_sequence(seq, max_mutations, sequences_set, sub_folder):
    generate_mutations(seq, max_mutations, sequences_set, sub_folder)
 
def delete_mutation_folder():
    timestamp = str(int(time.time()))
    sub_folder = os.path.join(MUTATIONS_DIR, timestamp)
    os.makedirs(sub_folder, exist_ok=True)
# # Now create the directory if it doesn't exist
#     os.makedirs(MUTATIONS_DIR, exist_ok=True)
    return sub_folder

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    total_start_time = time.time()
    sub_folder = delete_mutation_folder()

    # 1000 sequences
    sequences_file = 'files/forchen_F_26L.csv'
    sequences_header = "cdr3_amino_acid"

    # # 40K sequences
    # sequences_file = 'files/for_chen_B.csv'
    # sequences_header = "CDR3.aa"

    # # 1M sequences
    # sequences_file = "random_sequences.csv"
    # sequences_header = "sequences"

    # sequences_file = 'files/articles/mels.csv'
    # sequences_header = "mel7.cdr3a"

    print("truncating sequences")
    start_time = time.time()

    sequences_set = set(truncate_sequences(sequences_file, sequences_header, 4, 4))



    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    print("creating mutations")
    start_time = time.time()

    # # Process sequences in batches
    # max_workers = 10
    # batch_size = len(sequences_set) // max_workers + (1 if len(sequences_set) % max_workers != 0 else 0)
    # with tqdm(total=len(sequences_set)) as pbar:
    #     for batch_seqs in chunks(list(sequences_set), batch_size):
    #         with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #             futures = [executor.submit(process_sequence, seq, max_mutations, sequences_set, sub_folder) for seq in tqdm(batch_seqs)]
    #             for future in futures:
    #                 future.result()  # Wait for each process to finish before proceeding
    #                 pbar.update(1)

    # Process sequences in batches
    batch_size = 5000
    # num_batches = len(sequences_set) // batch_size + (1 if len(sequences_set) % batch_size != 0 else 0)
    with tqdm(total=len(sequences_set)) as pbar:
        for batch_seqs in chunks(list(sequences_set), batch_size):
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_sequence, seq, max_mutations, sequences_set, sub_folder) for seq in batch_seqs]
                for future in futures:
                    future.result()  # Wait for each process to finish before proceeding
                    pbar.update(1)

    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    
    couples = defaultdict(list)

    print("creating output file")
    start_time = time.time()
    for filename in os.listdir(sub_folder):
        with open(os.path.join(sub_folder, filename)) as f:
            csvFile = csv.reader(f)
            for line in csvFile:
                mut = line[0]
                seq = line[1]
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
