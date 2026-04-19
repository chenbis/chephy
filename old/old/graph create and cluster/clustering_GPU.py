import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import ray
import cudf
import cugraph
from itertools import product


# Node2Vec with GPU
def node2vec_embeddings_gpu(G, embedding_dim=64, walk_length=30, context_size=10, walks_per_node=20, epochs=5, lr=0.01):
    """Generate Node2Vec embeddings using PyTorch Geometric on GPU."""
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node2vec = Node2Vec(
        edge_index, 
        embedding_dim=embedding_dim, 
        walk_length=walk_length, 
        context_size=context_size, 
        walks_per_node=walks_per_node
    ).to(device)
    node2vec.train(batch_size=256, lr=lr, epochs=epochs)
    return node2vec.embedding.weight.detach().cpu().numpy()

# KMeans Clustering with GPU-based embeddings
def kmeans_clustering_gpu(G, n_clusters):
    """Apply KMeans clustering to the graph using GPU-accelerated Node2Vec embeddings."""
    embeddings = node2vec_embeddings_gpu(G)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}

# Spectral Clustering with cuGraph
def spectral_clustering_gpu(G, n_clusters):
    """Apply GPU-accelerated Spectral Clustering using cuGraph."""
    cu_G = cugraph.Graph()
    edge_list = nx.to_pandas_edgelist(G)
    edge_df = cudf.DataFrame.from_pandas(edge_list)
    cu_G.from_cudf_edgelist(edge_df, source='source', destination='target', edge_attr='weight')

    parts = cugraph.spectralBalancedCutClustering(cu_G, n_clusters)
    cluster_map = parts.to_pandas().set_index('vertex')['cluster'].to_dict()
    return cluster_map

# General Clustering Wrapper
def general_clustering(G, method, **kwargs):
    """Wrapper for various clustering methods."""
    if method == 'kmeans':
        return kmeans_clustering_gpu(G, **kwargs)
    elif method == 'spectral':
        return spectral_clustering_gpu(G, **kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

# Clustering Evaluation
def evaluate_clustering(clustering_result, labels_df):
    """Evaluate clustering accuracy."""
    true_labels = labels_df['label'].values
    pred_labels = [clustering_result.get(node, -1) for node in labels_df['node']]
    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy

# Ray-based Grid Search for Hyperparameter Optimization
@ray.remote
def evaluate_params_ray(G, labels_df, method, param_names, params):
    param_dict = dict(zip(param_names, params))
    try:
        clustering_result = general_clustering(G, method, **param_dict)
        accuracy = evaluate_clustering(clustering_result, labels_df)
        return accuracy, param_dict
    except Exception as e:
        return 0, param_dict

def grid_search_clustering_ray(G, labels_df, methods, param_grid):
    """Perform grid search for clustering methods using Ray."""
    ray.init(ignore_reinit_error=True)
    best_params = {}
    best_scores = {}

    for method in methods:
        method_params = param_grid[method]
        param_combinations = list(product(*method_params.values()))
        param_names = list(method_params.keys())

        tasks = [
            evaluate_params_ray.remote(G, labels_df, method, param_names, params)
            for params in param_combinations
        ]
        results = ray.get(tasks)
        best_result = max(results, key=lambda x: x[0])
        best_scores[method], best_params[method] = best_result

        print(f"Best params for {method}: {best_params[method]}, Best Accuracy: {best_scores[method]:.4f}")

    ray.shutdown()
    return best_params, best_scores

# Example Usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.karate_club_graph()
    labels_df = pd.DataFrame({
        'node': list(G.nodes()),
        'label': [G.nodes[node]['club'] == 'Mr. Hi' for node in G.nodes()]  # Example labels
    })

    # Define methods and parameter grid
    methods = ['kmeans', 'spectral']
    param_grid = {
        'kmeans': {'n_clusters': [2, 3, 4]},
        'spectral': {'n_clusters': [2, 3, 4]}
    }

    # Perform grid search
    best_params, best_scores = grid_search_clustering_ray(G, labels_df, methods, param_grid)
    print(f"Best Parameters: {best_params}")
    print(f"Best Scores: {best_scores}")
