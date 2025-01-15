import chephy_model as cpm
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse, datetime, csv
import ujson
from pathlib import Path
from numpy import integer, floating, ndarray
import networkx as nx
import clustering as clustering
from itertools import product


def map_trunc_to_full(couples, full_to_trunc_map):
    # Initialize an empty dictionary for the full couples
    couples_full = defaultdict(dict)

    # Iterate over each truncated sequence and its neighbors
    for short_seq, neighbors in couples.items():
        # Get the original sequences for the current short sequence
        original_seqs = full_to_trunc_map.get(short_seq, [])

        # Iterate over each neighbor and its details
        for neighbor, details in neighbors.items():
            # Get the original sequences for the neighbor
            neighbor_original_seqs = full_to_trunc_map.get(neighbor, [])
            
            # Add each pair of original sequences with their weight and dist
            for original_seq in original_seqs:
                for neighbor_original_seq in neighbor_original_seqs:
                    # Ensure the nested structure is maintained
                    if original_seq not in couples_full:
                        couples_full[original_seq] = {}
                    couples_full[original_seq][neighbor_original_seq] = {
                        "weight": details["weight"],
                        "dist": details["dist"]
                    }

    return dict(couples_full)

def find_close_sequences(cdr3, max_dist=0.5, max_mutations=3, right=4, left=4):
    sequences_set, full_to_trunc_map = cpm.truncate_sequences(cdr3, right, left)
    couples_trunc = cpm.find_che_phy_dist(sequences_set, max_mutations, max_dist)
    couples_full = map_trunc_to_full(couples_trunc, full_to_trunc_map)
    return couples_full

# def hamming_distance(seq1, seq2):
#     return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

# def find_sequences_within_distance(cdr3_list, max_dist, right=4, left=4):
#     """Find all sequences that are within a Hamming distance of 1 for each sequence."""

#     sequences_set, full_to_trunc_map = cpm.truncate_sequences(cdr3_list, right, left)
#     sequences_set = list(sequences_set)
    

#     couples_trunc = defaultdict(list)
    
#     for i, seq1 in enumerate(sequences_set):
#         # Initialize an empty list for each sequence
        
#         for seq2 in sequences_set:
#             # Skip comparing the sequence with itself
#             if seq1 == seq2:
#                 continue
            
#             # If the Hamming distance is 1, add to the list
#             distance = hamming_distance(seq1, seq2)
#             if distance <= max_dist:
#                 couples_trunc[seq1].append([seq2, distance/len(seq1)])
    
#     couples_full = map_trunc_to_full(couples_trunc, full_to_trunc_map)
#     return couples_full

# def prepare_data(data, cdr3_header, epitope_header):
    
#     ## remove cases where cdr3 is associated with multiple epitopes


#     # Step 1: Group by 'cdr3' and 'antigen.epitope', and count occurrences
#     epitope_counts = data.groupby([cdr3_header, epitope_header]).size().reset_index(name='count')

#     # Step 2: Get the most frequent epitope for each 'cdr3'
#     most_frequent_epitopes = epitope_counts.loc[epitope_counts.groupby(cdr3_header)['count'].idxmax()]

#     # Step 3: Merge with the original dataframe to retain only the most frequent epitopes
#     df_filtered = data.merge(most_frequent_epitopes[[cdr3_header, epitope_header]], on=[cdr3_header, epitope_header])

#     return df_filtered

def prepare_data(data, cdr3_header):

    # # remove nan values
    # data_filtered = data.dropna(subset=[cdr3_header])

    # remove cdr3 sequences that contain non-aa letters (this also removes nan)
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]

    # move truncate to here

    # merge with other prepare_data method after clustering is ready

    # drop duplicates
    data.drop_duplicates(subset=cdr3_header)

    return data_filtered


def get_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def save_params(parmas_file, args):

    args_dict = vars(args)
    # Write to a CSV file
    with open(parmas_file, "w", newline="") as file:
        writer = csv.writer(file)
        # Write header (keys)
        writer.writerow(args_dict.keys())
        # Write values
        writer.writerow(args_dict.values())


def write_file(couples, directory, filename):

    # Convert NumPy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, (integer, floating)):
            return obj.item()
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)
    output_file = f"{directory}/{filename}.json"
    processed_data = convert_numpy_types(couples)

    with open(output_file, "w") as f:
        ujson.dump(processed_data, f)

def calculate_greatest_dist(distance_matrix, seq_len = 8):
    max_distance = distance_matrix.max().max()
    greatest_distance = max_distance * seq_len
    return greatest_distance

# Function to save graph to CSV
def graph_to_csv(G, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['source', 'target', 'weight', "distance"])
        
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get('weight', 1), data.get('dist', 1)])

def create_mst(couples, output_folder):
    G = nx.from_dict_of_dicts(couples)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    graph_to_csv(mst, output_folder)
    return mst

def cluster(mst, output_folder, labels_path):

    labels_df = pd.read_csv(labels_path)

    methods = ['spectral', "kmeans"]
    param_grid = {
        'spectral': {'n_clusters': [199, 190, 180, 170, 165, 160, 150, 100, 200, 50]},
        'kmeans': {'n_clusters': [199, 190, 180, 170, 165, 160, 150, 100, 200, 50]},
    }

    all_results = []  # List to store all results

    for method in methods:
        method_params = param_grid[method]
        param_combinations = list(product(*method_params.values()))
        param_names = list(method_params.keys())  # Extract parameter names

        results = clustering.grid_search_clustering_parallel(
            mst, labels_df, method, param_combinations, param_names
        )

        for result in results:
            accuracy, recall, f1, params, clusters = result

            all_results.append({
                'method': method,
                'params': str(params),
                'accuracy': accuracy,
                'recall': recall,
                'f1_score': f1, 
                'clusters': str(clusters)
            })

    # Convert the results to a DataFrame and save it as CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{output_folder}/clustering_stats.csv', index=False)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="files/vdjdb_score3.csv", help="Input csv for the model to train on, must be a csv file")
    parser.add_argument("-m", "--mutations", default=8, type=int, help="Maximum number of mutations, default is 8")
    parser.add_argument("-of", "--out_folder", default=get_time(), help="Name of output sub folder, default is current time")
    parser.add_argument("-r", "--right", default=4, help="Trim from the right side, default is 4")
    parser.add_argument("-l", "--left", default=4, help="Trim from the left side, default is 4")

    args = parser.parse_args()


    max_mutations = args.mutations
    out_folder = f"output/{args.out_folder}"
    out_name = "neighbors"
    params_file = "params.csv"
    right = args.right
    left = args.left
    input_file = args.input
    max_dist=1


    distances_csv = "distance_matrix.csv"
    distances_df = pd.read_csv(distances_csv, index_col=0)
    max_distance = calculate_greatest_dist(distances_df)
    print(max_distance)
    substitution_matrix = {(aa1, aa2): distances_df.loc[aa1, aa2] for aa1 in distances_df.index for aa2 in distances_df.columns}
    # max_substitution_costs = {aa: distances_df.loc[aa].max() for aa in distances_df.index}
    
 
    data = pd.read_csv(input_file)
    cdr3_header = "cdr3"

    # data = prepare_data(data, cdr3_header, epitope_header)
    data = prepare_data(data, cdr3_header)    
    cdr3 = list(data[cdr3_header])
    
    sequences_set, full_to_trunc_map = cpm.truncate_sequences(cdr3, right, left)
    neighbors = cpm.find_che_phy_dist(sequences_set, substitution_matrix, max_distance)
    couples_full = map_trunc_to_full(neighbors, full_to_trunc_map)  


    write_file(couples_full, f"{out_folder}", f"{out_name}")
    save_params(f"{out_folder}/{params_file}", args)
    mst = create_mst(couples_full, f"{out_folder}/mst.csv")
    cluster(mst, out_folder, labels_path="files/vdjdb_score3.csv")

    # think of a general method to write files
    # add mst calculation (in cosmo_create.ipynb)
    # save mst to cosmo file (edge data) (in cosmo_create.ipynb)
    # add clustering calculation (in clustering.py)
    # save clustering to cosmo file (node data) (in clustering.py)


if __name__ == "__main__":
    main()