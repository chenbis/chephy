import chephy_model as cpm
import pandas as pd
import argparse, datetime, csv
from pathlib import Path
# import clustering
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from itertools import product


def prepare_data(data, cdr3_header, epitope_header, score_header='vdjdb.score'):

    # remove cdr3 sequences that contain non-aa letters (this also removes nan)
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]

    data_filtered = data_filtered[data_filtered[cdr3_header].str.len() >= 8]

    return data_filtered


def get_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_stats(stats_file, args, execution_time):
    
    # todo add stats to every method

    """Save parameters and execution time to stats file."""
    args_dict = vars(args)
    args_dict["execution_time_seconds"] = execution_time  # Add execution time

    with open(stats_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(args_dict.keys())  # Write headers
        writer.writerow(args_dict.values())  # Write values


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV or TSV file and loads it into a Pandas DataFrame.
    Raises an error if the file is not a CSV or TSV.
    
    :param file_path: Path to the file.
    :return: Pandas DataFrame containing the file data.
    """
    # Check file extension
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_extension.lower() == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")    

def read_csv_files_from_folder(folder_path="/dsi/scratch/home/dsi/solefroni/orforchen/downsamples_clonotype_21355"):
    """
    Reads all CSV files in a given folder and concatenates them into a single DataFrame.
    
    :param folder_path: Path to the folder containing CSV files.
    :return: Concatenated Pandas DataFrame.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        raise ValueError("No CSV files found in the specified folder.")



def cluster(mst, output_folder, data, labels_header, train_indices):

    methods = ["spectral", "louvain", "dbscan"]
    param_grid = {
        'spectral': {'n_clusters': [199, 190, 180]},
        'kmeans': {'n_clusters': [199, 190, 180]},
        "louvain":{},
        "dbscan":{"eps":[0.9], "min_samples":[2]}
    }

    all_results = []  # List to store all results

    for method in methods:
        method_params = param_grid[method]
        param_combinations = list(product(*method_params.values()))
        param_names = list(method_params.keys())  # Extract parameter names

        results = clustering.grid_search_clustering_parallel(
            mst, data, labels_header, method, param_combinations, param_names, train_indices
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
    results_df.to_csv(f'{output_folder}/training_stats.csv', index=False)
    return results_df  

def mixmax_normalization(G, epsilon=1e-6):
    # Extract edge weights
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    
    if weights:  # Ensure there are edges
        max_weight = max(weights)
        min_weight = min(weights)

        # Normalize edge weights
        for u, v, d in G.edges(data=True):
            d['weight'] = (d['weight'] - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 0
            d['weight'] += epsilon  # Avoid zero weights

def train(train_distance_matrix, train_indices, data, out_folder, hamming_matrix):

    mst_graph = create_graph(hamming_matrix, train_distance_matrix, threshold=2)
    # mst_graph = construct_enhanced_graph(train_distance_matrix, 3)

    mixmax_normalization(mst_graph)

    results = cluster(mst_graph, out_folder, data, "antigen.epitope", train_indices)
    best_cluster = results.loc[results['accuracy'].idxmax(), 'clusters']
    return best_cluster
    # results.to_csv("test.csv")
    # print()

def test(data, test_indices, chephy_matrix, train_indices, clusters):
    closest_train_indices = np.argmin(chephy_matrix[np.ix_(test_indices, train_indices)], axis=1)
    test_predictions = {test_idx: clusters[train_indices[closest_idx]] for test_idx, closest_idx in zip(test_indices, closest_train_indices)}
    accuracy, recall, f1 = clustering.evaluate_clustering(test_predictions, data, "antigen.epitope")
    print(f"Test Accuracy: {accuracy}, Recall: {recall}, F1-score: {f1}")
    return test_predictions

def construct_enhanced_graph(distance_matrix, min_edges):
    """
    Constructs a graph from a distance matrix, ensuring each node has at least 'min_edges' connections.

    Parameters:
        distance_matrix (numpy.ndarray): The original n*n distance matrix.
        min_edges (int): Minimum number of edges each node should have.

    Returns:
        nx.Graph: The resulting graph with the MST and additional edges.
    """
    # Compute the Minimum Spanning Tree (MST)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst_graph = nx.from_scipy_sparse_array(mst_sparse)
    if min_edges == 1:
        return mst_graph
    
    num_nodes = distance_matrix.shape[0]

    # Ensure each node has at least 'min_edges' connections
    for node in range(num_nodes):
        current_edges = list(mst_graph.neighbors(node))
        num_current_edges = len(current_edges)

        if num_current_edges < min_edges:
            # Extract distances from the node to all others
            node_distances = [(j, distance_matrix[node, j]) for j in range(num_nodes) 
                              if j != node and not mst_graph.has_edge(node, j)]
            
            # Sort potential edges by weight
            node_distances.sort(key=lambda x: x[1])  # Sort by distance (ascending)
            
            # Add edges with lowest weights until the node has 'min_edges' edges
            for neighbor, weight in node_distances:
                if len(list(mst_graph.neighbors(node))) >= min_edges:
                    break
                mst_graph.add_edge(node, neighbor, weight=weight)

    return mst_graph


def create_graph(hamming_matrix, chephy_matrix, threshold=3):
    """
    Create a graph based on the given matrices.
    
    1. Add edges with Hamming distance < threshold.
    2. Check if the graph is connected.
    3. If not, iteratively add the lowest-weight Chephy edge until connected.
    
    :param hamming_matrix: N x N Hamming distance matrix
    :param chephy_matrix: N x N Chephy distance matrix
    :param threshold: Hamming distance threshold for initial edges
    :return: Connected graph (NetworkX object)
    """


    mst_sparse = minimum_spanning_tree(chephy_matrix)
    mst_graph = nx.from_scipy_sparse_array(mst_sparse)

    # Step 1: Add initial edges based on Hamming distance
    for i in range(len(chephy_matrix)):
        for j in range(i + 1, len(chephy_matrix)):
            if hamming_matrix[i, j] <= threshold:
                mst_graph.add_edge(i, j, weight=chephy_matrix[i, j])

    return mst_graph

# def main():
#     start_time = time.time()

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", default="files/vdjdb_score3.csv", help="Input csv for the model to train on, must be a csv file")
#     parser.add_argument("-of", "--out_folder", default=get_time(), help="Name of output sub folder, default is current time")
#     parser.add_argument("-r", "--right", default=4, help="Trim from the right side, default is 4")
#     parser.add_argument("-l", "--left", default=4, help="Trim from the left side, default is 4")
#     parser.add_argument("-ch", "--cdr_header",default="cdr3", help="cdr3 header in the input file. default id CDR3b")
#     parser.add_argument("-eh", "--epitope_header",default="antigen.epitope", help="Epitope header in the input file. default id Epitope")


#     args = parser.parse_args()


#     # max_mutations = args.mutations
#     out_folder = f"output/{args.out_folder}"
#     out_name = "neighbors"
#     params_file = "stats.csv"
#     right = args.right
#     left = args.left
#     input_file = args.input
#     max_dist=1
#     label_header = args.epitope_header
#     cdr3_header = args.cdr_header
    

#     path = Path(f"{out_folder}")
#     path.mkdir(parents=True, exist_ok=True)

#     data = load_dataframe(input_file)

#     # data = read_csv_files_from_folder()

#     data = prepare_data(data, cdr3_header, label_header)    
#     data = cpm.truncate_sequences(data, cdr3_header, right, left)

#     sequences = set(data["cdr3_truncated"])
#     chephy_matrix, hamming_matrix, sequences_list = cpm.find_che_phy_dist(sequences)
#     data['index'] = data['cdr3_truncated'].apply(lambda x: sequences_list.index(x) if x in sequences_list else -1)
#     sequence_indices = np.arange(len(sequences_list))


    

#     # Split into train and test (80% train, 20% test)
#     train_indices, test_indices = train_test_split(sequence_indices, test_size=0.2)

#     # Extract submatrix for training set
#     train_distance_matrix = chephy_matrix[np.ix_(train_indices, train_indices)]
#     hamming_training_matrix = hamming_matrix[np.ix_(train_indices, train_indices)]

#     clusters = train(train_distance_matrix, train_indices, data, out_folder, hamming_training_matrix)

#     test(data, test_indices, chephy_matrix, train_indices, clusters)
#     data.to_csv(f"{out_folder}/data.csv")
#     execution_time = time.time() - start_time
#     save_stats(f"{out_folder}/{params_file}", args, execution_time)


# if __name__ == "__main__":
#     main()