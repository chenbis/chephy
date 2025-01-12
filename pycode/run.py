import chephy_model as cpm
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse, datetime, csv
import ujson
from pathlib import Path



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


# def map_clusters(couples, max_neig):
#     # Create a graph
#     G = nx.Graph()

#     # Add edges to the graph based on the dictionary
#     for node, neighbors in tqdm(couples.items()):
#         count = 0
#         for neighbor, weight in neighbors:
#             if count < max_neig:
#                 G.add_edge(node, neighbor, weight=weight)
#                 count += 1


#     # Find connected components
#     clusters = list(nx.connected_components(G))

#     # Create a dictionary to map sequences to cluster numbers
#     cluster_dict = {}
#     # for cluster_number, cluster in enumerate(clusters):
#     #     for sequence in cluster:
#     #         cluster_dict[sequence] = cluster_number    

#     for cluster_number, cluster in enumerate(clusters):
#         cluster_dict[cluster_number] = cluster
   
#     return cluster_dict



def get_true_clusters(data, cdr3_header, eptiope_header):
    
    clusters_dict = {}

    # Assign clusters based on common antigen
    for cluster_id, (antigen, group) in enumerate(data.groupby(eptiope_header)):
        for cdr3 in group[cdr3_header]:
            clusters_dict[cdr3] = cluster_id

    return clusters_dict

def classify(clusters, data, cdr3_header):
    cluster_classification = {}
    for cluster, tcrs in tqdm(clusters.items()):
        epitope_strengths = defaultdict(int)
        for tcr in list(tcrs):
            if tcr == "CASSEGWHSYEQYF":
                print()
            row = data[data[cdr3_header] == tcr]
            epitope = row['antigen.epitope'].values[0]
            strength = row['vdjdb.score'].values[0]
            epitope_strengths[epitope] += strength

        best_epitope = max(epitope_strengths, key=epitope_strengths.get)
        cluster_classification[cluster] = best_epitope

    return cluster_classification




def hamming_distance(seq1, seq2):
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def find_sequences_within_distance(cdr3_list, max_dist, right=4, left=4):
    """Find all sequences that are within a Hamming distance of 1 for each sequence."""

    sequences_set, full_to_trunc_map = cpm.truncate_sequences(cdr3_list, right, left)
    sequences_set = list(sequences_set)
    

    couples_trunc = defaultdict(list)
    
    for i, seq1 in enumerate(sequences_set):
        # Initialize an empty list for each sequence
        
        for seq2 in sequences_set:
            # Skip comparing the sequence with itself
            if seq1 == seq2:
                continue
            
            # If the Hamming distance is 1, add to the list
            distance = hamming_distance(seq1, seq2)
            if distance <= max_dist:
                couples_trunc[seq1].append([seq2, distance/len(seq1)])
    
    couples_full = map_trunc_to_full(couples_trunc, full_to_trunc_map)
    return couples_full

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

    data_filtered = data[data['CDR3b'].str.match(amino_acid_pattern, case=False, na=False)]

    # move truncate to here

    # merge with other prepare_data method after clustering is ready

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
    path = Path(directory)

    path.mkdir(parents=True, exist_ok=True)
    output_file = f"{directory}/{filename}.json"
    with open(output_file, "w") as f:
        ujson.dump(couples, f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="files/vdjdb_cdr3.csv", help="Input csv for the model to train on, must be a csv file")
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
    
    data = pd.read_csv(input_file)

    # # 1000 sequences
    # data = pd.read_csv('files/forchen_F_26L.csv')
    # cdr3_header = "cdr3_amino_acid"

    # # # 10000 sequences
    # data = pd.read_csv("files/cdrs_list.csv")
    # cdr3_header = "Sequences"


    # vdjdb beta chain
    # data = pd.read_csv("files/vdjdb_cdr3.csv")
    # data = data[(data["vdjdb.score"] >= 3)]
    
    cdr3_header = "CDR3b"

    # data = prepare_data(data, cdr3_header, epitope_header)
    data = prepare_data(data, cdr3_header)


    data.drop_duplicates(subset=cdr3_header)
    cdr3 = list(data[cdr3_header])
    
    couples_full = find_close_sequences(cdr3, max_dist=max_dist, max_mutations=max_mutations, right=right, left=left)
    
    write_file(couples_full, f"{out_folder}", f"{out_name}")
    save_params(f"{out_folder}/{params_file}", args)



if __name__ == "__main__":
    main()