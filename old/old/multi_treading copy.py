import pandas as pd
import itertools
from collections import defaultdict
import json, csv, time, os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MUTATIONS_DIR = "mutations"
max_mutations = 1

def generate_mutations(sequence, max_mutations, sequences_set: set, output_file):
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    found_neighbor = False
    with open(output_file, "a") as f:
        for i in range(1, max_mutations + 1):
            for positions in itertools.combinations(range(len(sequence)), i):
                for new_aas in itertools.product(aa, repeat=i):
                    mutated_sequence = list(sequence)
                    for pos, new_aa in zip(positions, new_aas):
                        mutated_sequence[pos] = new_aa
                    mutated_sequence = "".join(mutated_sequence)

                    if mutated_sequence != sequence and mutated_sequence in sequences_set:
                        f.write(f"{mutated_sequence},{sequence}\n")
                        found_neighbor = True
    return found_neighbor

def truncate_sequences(csv_file, column_name, right, left):
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[column_name].strip()
            start_index = max(0, len(sequence)//2 - left)
            end_index = min(len(sequence), len(sequence)//2 + right)
            yield sequence[start_index:end_index]

def process_sequence(seq, max_mutations, sequences_set, sub_folder):
    output_file = os.path.join(sub_folder, f"{seq}.csv")
    if not (generate_mutations(seq, max_mutations, sequences_set, output_file)):
        os.remove(output_file)

 
def delete_mutation_folder():
    timestamp = str(int(time.time()))
    sub_folder = os.path.join(MUTATIONS_DIR, timestamp)
    os.makedirs(sub_folder, exist_ok=True)
# # Now create the directory if it doesn't exist
#     os.makedirs(MUTATIONS_DIR, exist_ok=True)
    return sub_folder

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

  # Load sequences into a set directly
    print("truncating sequences")
    start_time = time.time()
    sequences_set = set()
    with open(sequences_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = row[sequences_header].strip()
            start_index = max(0, len(sequence) // 2 - 4)
            end_index = min(len(sequence), len(sequence) // 2 + 4)
            sequences_set.add(sequence[start_index:end_index])
    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    # Process sequences in batches
    print("creating mutations")
    start_time = time.time()
    batch_size = 5000
    sequence_batches = []
    with tqdm(total=len(sequences_set), desc="Creating sequence batches") as pbar:
        for i in range(0, len(sequences_set), batch_size):
            sequence_batches.append(list(sequences_set)[i:i + batch_size])
            pbar.update(batch_size)

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        with tqdm(total=len(sequence_batches), desc="Processing sequences") as pbar:
            for batch in sequence_batches:
                futures.extend([executor.submit(process_sequence, seq, max_mutations, sequences_set, sub_folder) for seq in batch])
                pbar.update(1)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sequences"):
            future.result()
    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")


    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    distances_dict = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    
    # Process mutation files in parallel
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

    output_file = "couples.json"
    with open(output_file, "w") as f:
        json.dump(couples, f)

    end_time = time.time()
    print("time:", (end_time - start_time)/60, "minutes")

    total_end_time = time.time()
    print("total time:", (total_end_time - total_start_time)/60, "minutes")

if __name__ == "__main__":
    main()