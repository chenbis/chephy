import chephy_model as cpm
from collections import defaultdict
import time
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


def map_trunc_to_full(couples, full_to_trunc_map):
    # Initialize an empty dictionary for the full couples
    couples_full = defaultdict(list)

    # Iterate over each key, value pair in the couples dictionary
    for short_seq, neighbors in couples.items():
        # Get the original sequences for the current short sequence
        original_seqs = set(full_to_trunc_map.get(short_seq, []))

        # Iterate over each original sequence
        for original_seq in original_seqs:
            
            # Iterate over each neighbor in the neighbors list
            for neighbor, distance in neighbors:
                # Get the original sequences for the neighbor
                neighbor_original_seqs = full_to_trunc_map.get(neighbor, [])
                
                # Add each original neighbor sequence with the same distance
                for neighbor_original_seq in neighbor_original_seqs:
                    couples_full[original_seq].append([neighbor_original_seq, distance])

    return couples_full


def find_close_sequences(cdr3, max_mutations=3, right=4, left=4):
    sequences_set, full_to_trunc_map = cpm.truncate_sequences(cdr3, 4, 4)
    couples_trunc = cpm.find_che_phy_dist(sequences_set, max_mutations)
    couples_full = map_trunc_to_full(couples_trunc, full_to_trunc_map)
    return couples_full


def map_clusters(couples, max_neig):
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph based on the dictionary
    for node, neighbors in couples.items():
        count = 0
        for neighbor, weight in neighbors:
            if count < max_neig:
                G.add_edge(node, neighbor, weight=weight)
                count += 1


    # Find connected components
    clusters = list(nx.connected_components(G))

    # Create a dictionary to map sequences to cluster numbers
    cluster_dict = {}
    # for cluster_number, cluster in enumerate(clusters):
    #     for sequence in cluster:
    #         cluster_dict[sequence] = cluster_number    

    for cluster_number, cluster in enumerate(clusters):
        cluster_dict[cluster_number] = cluster
   
    return cluster_dict



def get_true_clusters(data, cdr3_header, eptiope_header):
    clusters_dict = {}

    # Assign clusters based on common antigen
    for cluster_id, (antigen, group) in enumerate(data.groupby(eptiope_header)):
        for cdr3 in group[cdr3_header]:
            clusters_dict[cdr3] = cluster_id

    return clusters_dict

def classify(clusters, data):
    cluster_classification = {}
    for cluster, tcrs in clusters.items():
        epitope_strengths = defaultdict(int)
        for tcr in list(tcrs):
            if tcr == "CASSEGWHSYEQYF":
                print()
            row = data[data['cdr3'] == tcr]
            epitope = row['antigen.epitope'].values[0]
            strength = row['vdjdb.score'].values[0]
            epitope_strengths[epitope] += strength

        best_epitope = max(epitope_strengths, key=epitope_strengths.get)
        cluster_classification[cluster] = best_epitope

    return cluster_classification


def main():

    max_mutations = 3
    out_folder = "output_files/tests"
    out_name = "test"
    max_neig = 1

    # # 1000 sequences
    # sequences_file = 'files/forchen_F_26L.csv'
    # sequences_headers = ["cdr3_amino_acid"]

    # # 40K sequences
    # sequences_file = 'files/for_chen_B2.csv'
    # sequences_headers = ["CDR3.aa"]

    # # 1M sequences
    # sequences_file = "random_sequences.csv"
    # sequences_headers = ["sequences"]


    # vdjdb beta chain
    data = pd.read_csv("files/vdjdb_cdr3.csv")
    data = data[data["vdjdb.score"] == 3]
    # sequences_file = "files/vdjdb_cdr3.csv"
    cdr3_header = "cdr3"
    epitope_header = "antigen.epitope"
    data.drop_duplicates(subset=cdr3_header)
    cdr3 = list(data[cdr3_header])
    

    couples_full = find_close_sequences(cdr3, max_mutations)
    
    start_time = time.time()
    cpm.write_couples_file(couples_full, "{}/{}".format(out_folder, cdr3_header), "{}_full".format(out_name))
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken to write full couples: {elapsed_time:.2f} minutes")



    start_time = time.time()
    cluster_dict_pred = map_clusters(couples_full, max_neig)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"{elapsed_time:.2f} minutes")
    # cluster_dict_true = get_true_clusters(data, cdr3_header, epitope_header)

    start_time = time.time()
    cluster_classification = classify(cluster_dict_pred, data)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"{elapsed_time:.2f} minutes")


    data['epitope.pred'] = None

    
    for cluster, tcrs in tqdm(cluster_dict_pred.items()):
        predicted_epitope = cluster_classification[cluster]
        data.loc[data['cdr3'].isin(tcrs), 'epitope.pred'] = predicted_epitope


    data.to_csv("{}/{}/predicted clusters.csv".format(out_folder, cdr3_header), encoding='utf-8', index=False)


    # ari = adjusted_rand_score(cluster_dict_true, cluster_dict_pred)
    


    # print(ari)
    # cpm.write_couples_file(couples_full, "{}/{}".format(out_folder, header), "{}_full".format(out_name))





if __name__ == "__main__":
    main()