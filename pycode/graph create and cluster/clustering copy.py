import pandas as pd
import networkx as nx
# from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score
from node2vec import Node2Vec
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import scipy.sparse as sp



def load_graph(file_path):
    """Load a graph from a CSV file."""
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['weight', 'distance'])
    return G


def louvain_clustering(G):
    """Apply Louvain clustering to the graph."""
    partition = community_louvain.best_partition(G, weight='weight')
    return partition


def spectral_clustering(G, n_clusters):
    """Apply Spectral Clustering to the graph."""
    # Convert the graph to a sparse adjacency matrix with weight
    adj_matrix = nx.to_scipy_sparse_array(G, weight='weight', dtype=np.float64)

    # Ensure the sparse matrix has 32-bit integer indices
    adj_matrix = sp.csr_matrix(adj_matrix, dtype=np.float64)  # CSR format
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

    # Perform Spectral Clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = sc.fit_predict(adj_matrix)
    return {node: labels[i] for i, node in enumerate(G.nodes())}


def kmeans_clustering(G, n_clusters):
    """Apply KMeans clustering to the graph using Node2Vec embeddings."""
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[node] for node in G.nodes()])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}

def general_clustering(G, method, n_clusters=None):
    """General clustering function to call specific methods."""
    if method == 'louvain':
        return louvain_clustering(G)
    if n_clusters is None:
        raise ValueError("n_clusters must be specified for Spectral and KMeans Clustering.")
    if method == 'spectral':
        return spectral_clustering(G, n_clusters)
    if method == 'kmeans':
        return kmeans_clustering(G, n_clusters)
    raise ValueError(f"Unknown clustering method: {method}")


def evaluate_clustering(clustering_result, labels_df):
    """Evaluate clustering results using majority vote for cluster labels."""
    labels_df['cluster'] = labels_df['cdr3'].map(clustering_result)
    cluster_labels = labels_df.groupby('cluster')['antigen.epitope'].agg(lambda x: x.mode()[0])
    labels_df['predicted_label'] = labels_df['cluster'].map(cluster_labels)

    y_true = labels_df['antigen.epitope']
    y_pred = labels_df['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, recall, f1

def run_clustering(G, method, n_clusters):
    clustering_result = general_clustering(G, method, n_clusters)
    return method, clustering_result

def main():
    graph_file_path = "/home/chen/repos/sol_lab/cosmo_10k.csv"
    labels_file_path = "/home/chen/repos/sol_lab/files/vdjdb_cdr3.csv"

    G = load_graph(graph_file_path)
    labels_df = pd.read_csv(labels_file_path)

    methods = ['spectral', 'kmeans']
    n_clusters = 5

    results = Parallel(n_jobs=-1)(
        delayed(run_clustering)(G, method, n_clusters) for method in methods
    )

    for method, clustering_result in results:
        accuracy, recall, f1 = evaluate_clustering(clustering_result, labels_df)
        print(f"Results for {method} clustering: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        pd.DataFrame({'id': list(G.nodes()), 'cluster': list(clustering_result.values())}).to_csv(f"clustering_results_{method}.csv", index=False)


if __name__ == "__main__":
    main()
