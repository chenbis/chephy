import pandas as pd
import networkx as nx
# from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import accuracy_score, recall_score, f1_score
from node2vec import Node2Vec
import numpy as np
from collections import Counter
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import json
from networkx.algorithms.community import louvain_communities

def load_graph(file_path):
    """Load a graph from a CSV file."""
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['weight', 'distance'])
    return G


def louvain_clustering(G):
    """Apply Louvain clustering to the graph."""
    for u, v, d in G.edges(data=True):
        d['weight'] = 1 - d['weight']
    louvain_clusters  = louvain_communities(G, weight='weight')

    # Convert list of sets to a dictionary mapping nodes to cluster IDs
    node_to_cluster = {node: cluster_id for cluster_id, cluster in enumerate(louvain_clusters) for node in cluster}
    
    return node_to_cluster

def dbscan_clustering(G, eps=0.5, min_samples=2):
    """Apply DBSCAN clustering to the graph and return a dictionary of node assignments."""
    
    # Convert graph to adjacency matrix and compute distance matrix
    adj_matrix = nx.to_numpy_array(G)
    distance_matrix = 1 - adj_matrix  # Convert similarity to distance
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    dbscan_labels = dbscan.fit_predict(distance_matrix)
    
    # Convert cluster labels into a dictionary
    node_to_cluster = {node: label for node, label in zip(G.nodes, dbscan_labels)}
    
    return node_to_cluster

def spectral_clustering(G, n_clusters):
    """Apply Spectral Clustering to the graph."""
    # Convert graph to adjacency matrix
    nodes = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, weight='weight')

    # Optional: Add progress if the adjacency matrix is large
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


def general_clustering(G, method, params):
    """General clustering function to call specific methods."""

    if method == 'louvain':
        return louvain_clustering(G)
    
    elif method == 'spectral':
        if params is None:
            raise ValueError("n_clusters must be specified for Spectral Clustering.")
        return spectral_clustering(G, **params)
    
    elif method == 'kmeans':
        if params is None:
            raise ValueError("n_clusters must be specified for KMeans Clustering.")
        return kmeans_clustering(G, **params)
    
    elif method == "dbscan":
        return dbscan_clustering(G, **params)

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def evaluate_clustering(clustering_result, labels_df, labels_header):
    """Evaluate clustering results using majority vote for cluster labels."""
    
    labels_df = labels_df[labels_df['index'].isin(clustering_result.keys())]
    labels_df = labels_df.copy()

    # Add a cluster column
    labels_df.loc[:, 'cluster'] = labels_df.loc[:, 'index'].map(clustering_result)

    # Determine majority label for each cluster
    cluster_labels = {}
    for cluster, group in labels_df.groupby('cluster'):
        majority_label = Counter(group[labels_header]).most_common(1)[0][0]
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

def grid_search_clustering_parallel(G, labels_df, labels_header, method, param_combinations, param_names, train_indices):
    def evaluate_params(params):
        param_dict = dict(zip(param_names, params))  # Create a parameter dictionary
        try:
            clustering_result = general_clustering(G, method, param_dict)
            clustering_result = {train_indices[key]: value for key, value in clustering_result.items()}
            accuracy, recall, f1 = evaluate_clustering(clustering_result, labels_df, labels_header)
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
