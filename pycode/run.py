import chephy_model as cpm
import pandas as pd
import argparse, datetime, csv
from pathlib import Path
import clustering as clustering
from itertools import product
from matplotlib.colors import to_hex
import ast
import matplotlib.pyplot as plt
import os
import time
from scipy.sparse.csgraph import minimum_spanning_tree


def prepare_data(data, cdr3_header, epitope_header, score_header='vdjdb.score'):

    # understand tcrscapes data preparation


    # remove cdr3 sequences that contain non-aa letters (this also removes nan)
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]

    data_filtered = data_filtered[data_filtered[cdr3_header].str.len() >= 8]

    # drop duplicates
    data_filtered = data_filtered.drop_duplicates(subset=cdr3_header)

    # move truncate to here

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


def cluster(mst, output_folder, data, labels_header):

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
            mst, data, labels_header, method, param_combinations, param_names
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
    return results_df


def save_best_cluster(df_cluster_stats, out_folder):

    # Find the row with the highest accuracy
    highest_accuracy_row = df_cluster_stats.loc[df_cluster_stats['accuracy'].idxmax()]

    # Parse the JSON in the 'clusters' column
    clusters = ast.literal_eval(highest_accuracy_row["clusters"])
    method = highest_accuracy_row["method"]

    # Create a DataFrame from the JSON
    clusters_df = pd.DataFrame(list(clusters.items()), columns=['id', method])

    # Generate unique colors for each cluster
    num_clusters = len(clusters_df[method].unique())
    colors = plt.cm.tab20.colors  # Use a colormap with enough distinct colors
    hex_colors = [to_hex(c) for c in colors]
    color_mapping = {cluster: hex_colors[i % len(hex_colors)] for i, cluster in enumerate(clusters_df[method].unique())}

    # Add the color column
    clusters_df['color'] = clusters_df[method].map(color_mapping)

    # Save the resulting DataFrame to a new CSV
    clusters_df.to_csv(f'{out_folder}/clusters_output.csv', index=False)

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
    

def build_mst(chephy_matrix, ham_matrix, sequences_list, full_to_trunc_map, output_file):

    # todo: add truncated metadata to the nodes, or maybe the full sequence?

    # Compute MST
    mst = minimum_spanning_tree(chephy_matrix).toarray()
    
    # Extract MST edges and distances
    sources, targets, chephy_distances, hamming_distances = [], [], [], []
    
    for i in range(mst.shape[0]):
        for j in range(mst.shape[1]):
            if mst[i, j] > 0:  # Edge exists in MST
                sources.append(sequences_list[i])
                targets.append(sequences_list[j])
                chephy_distances.append(mst[i, j])
                hamming_distances.append(ham_matrix[i, j])
    
    # Convert to full sequences
    def map_to_full(truncated_seq):
        return "|".join(full_to_trunc_map[truncated_seq])
    
    source_full = [map_to_full(seq) for seq in sources]
    target_full = [map_to_full(seq) for seq in targets]
    
    # Create DataFrame
    mst_df = pd.DataFrame({
        "source": source_full,
        "target": target_full,
        "chephy_distance": chephy_distances,
        "hamming_distance": hamming_distances
    })
    
    # Save to CSV
    mst_df.to_csv(output_file, index=False)
    
    return mst_df




def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="files/vdjdb_score3.csv", help="Input csv for the model to train on, must be a csv file")
    parser.add_argument("-of", "--out_folder", default=get_time(), help="Name of output sub folder, default is current time")
    parser.add_argument("-r", "--right", default=4, help="Trim from the right side, default is 4")
    parser.add_argument("-l", "--left", default=4, help="Trim from the left side, default is 4")
    parser.add_argument("-ch", "--cdr_header",default="cdr3", help="cdr3 header in the input file. default id CDR3b")
    parser.add_argument("-eh", "--epitope_header",default="antigen.epitope", help="Epitope header in the input file. default id Epitope")


    args = parser.parse_args()


    # max_mutations = args.mutations
    out_folder = f"output/{args.out_folder}"
    out_name = "neighbors"
    params_file = "stats.csv"
    right = args.right
    left = args.left
    input_file = args.input
    max_dist=1
    label_header = args.epitope_header
    cdr3_header = args.cdr_header
    

    path = Path(f"{out_folder}")
    path.mkdir(parents=True, exist_ok=True)

    data = load_dataframe(input_file)

    # data = read_csv_files_from_folder()

    data = prepare_data(data, cdr3_header, label_header)    
    cdr3 = list(data[cdr3_header])
    epitopes = list(data[label_header])

    sequences_set, full_to_trunc_map, trunc_to_epitope = cpm.truncate_sequences(cdr3, epitopes, right, left)
    chephy_matrix, hamming_matrix, sequences_list = cpm.find_che_phy_dist(sequences_set)
    
    mst = build_mst(chephy_matrix, hamming_matrix, sequences_list, full_to_trunc_map, f"{out_folder}/mst.csv")


    # df_results = cluster(mst, out_folder, data, label_header)
    # save_best_cluster(df_results, out_folder)
    # # think of a general method to write files

    execution_time = time.time() - start_time
    save_stats(f"{out_folder}/{params_file}", args, execution_time)


if __name__ == "__main__":
    main()