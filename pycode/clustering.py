import pandas as pd
import networkx as nx
# from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score
from node2vec import Node2Vec
import numpy as np
from collections import Counter
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import json

def load_graph(file_path):
    """Load a graph from a CSV file."""
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['weight', 'distance'])
    return G


# def louvain_clustering(G):
#     """Apply Louvain clustering to the graph."""
#     partition = community_louvain.best_partition(G, weight='weight')
#     return partition

def spectral_clustering(G, n_clusters):
    """Apply Spectral Clustering to the graph."""
    # Convert graph to adjacency matrix
    nodes = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, weight='weight')

    # Optional: Add progress if the adjacency matrix is large
    print("Running Spectral Clustering...")
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = sc.fit_predict(adj_matrix)

    # Map cluster labels back to nodes
    return {node: labels[i] for i, node in enumerate(tqdm(nodes, desc="Mapping Spectral Clustering Labels"))}


def kmeans_clustering(G, n_clusters):
    """Apply KMeans clustering to the graph using Node2Vec embeddings."""
    node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, workers=8)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[node] for node in G.nodes()}
    X = np.array([embeddings[node] for node in G.nodes()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}

def general_clustering(G, method, n_clusters=None):
    """General clustering function to call specific methods."""

    if method == 'louvain':
        return louvain_clustering(G)
    
    elif method == 'spectral':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Spectral Clustering.")
        return spectral_clustering(G, n_clusters)
    
    elif method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans Clustering.")
        return kmeans_clustering(G, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def evaluate_clustering(clustering_result, labels_df):
    """Evaluate clustering results using majority vote for cluster labels."""
    labels_df = labels_df[labels_df['cdr3'].isin(clustering_result.keys())]
    labels_df = labels_df.copy()

    # Add a cluster column
    labels_df.loc[:, 'cluster'] = labels_df.loc[:, 'cdr3'].map(clustering_result)

    # Determine majority label for each cluster
    cluster_labels = {}
    for cluster, group in labels_df.groupby('cluster'):
        majority_label = Counter(group['antigen.epitope']).most_common(1)[0][0]
        cluster_labels[cluster] = majority_label

    # Map predicted labels
    labels_df.loc[:, 'predicted_label'] = labels_df.loc[:, 'cluster'].map(cluster_labels)

    # Calculate metrics
    y_true = labels_df['antigen.epitope']
    y_pred = labels_df['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, recall, f1

def grid_search_clustering_parallel(G, labels_df, method, param_combinations, param_names):
    def evaluate_params(params):
        param_dict = dict(zip(param_names, params))  # Create a parameter dictionary
        try:
            clustering_result = general_clustering(G, method, **param_dict)
            accuracy, recall, f1 = evaluate_clustering(clustering_result, labels_df)
            return accuracy, recall, f1, param_dict, clustering_result  # Return accuracy, recall, f1, and params
        except Exception:
            return 0, 0, 0, params, {}  # Return zero values in case of an error

    # Parallelize the grid search
    results = Parallel(n_jobs=-1)(delayed(evaluate_params)(params) for params in param_combinations)
    # results = [evaluate_params(params) for params in param_combinations] # for debugging

    return results

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, str):  # Ensure strings are returned as-is
            return obj
        else:
            return super().default(obj)

def main():
    graph_file_path = "pycode/graph create and cluster/cosmo_graph1.csv"
    labels_file_path = "files/vdjdb_cdr3.csv"

    G = load_graph(graph_file_path)
    labels_df = pd.read_csv(labels_file_path)

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

        results = grid_search_clustering_parallel(
            G, labels_df, method, param_combinations, param_names
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
    results_df.to_csv('clustering_stats.csv', index=False)
    print("Results saved to clustering_results.csv")

if __name__ == "__main__":
    main()