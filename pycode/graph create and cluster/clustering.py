import pandas as pd
import networkx as nx
# from community import community_louvain
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score
from node2vec import Node2Vec
import numpy as np
from collections import Counter

def load_graph(file_path):
    """Load a graph from a CSV file."""
    df = pd.read_csv(file_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'], distance=row['distance'])
    return G

def louvain_clustering(G):
    """Apply Louvain clustering to the graph."""
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def spectral_clustering(G, n_clusters):
    """Apply Spectral Clustering to the graph."""
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = sc.fit_predict(adj_matrix)
    return {node: labels[i] for i, node in enumerate(G.nodes())}

def kmeans_clustering(G, n_clusters):
    """Apply KMeans clustering to the graph using Node2Vec embeddings."""
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
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
    labels_df['cluster'] = labels_df['cdr3'].map(clustering_result)

    # Determine majority label for each cluster
    cluster_labels = {}
    for cluster, group in labels_df.groupby('cluster'):
        majority_label = Counter(group['antigen.epitope']).most_common(1)[0][0]
        cluster_labels[cluster] = majority_label

    # Map predicted labels
    labels_df['predicted_label'] = labels_df['cluster'].map(cluster_labels)

    # Calculate metrics
    y_true = labels_df['antigen.epitope']
    y_pred = labels_df['predicted_label']
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, recall, f1

def main():
    graph_file_path = "/Users/camir/Documents/repos/sol_lab/cosmo_10k.csv"
    labels_file_path = "/Users/camir/Documents/repos/sol_lab/files/vdjdb_cdr3.csv"

    G = load_graph(graph_file_path)

    # Run all clustering methods
    methods = ['spectral', 'kmeans']
    n_clusters = 5  # Default number of clusters for spectral and kmeans

    for method in methods:
        print(f"Running {method} clustering...")
        if method in ['spectral', 'kmeans']:
            clustering_result = general_clustering(G, method, n_clusters)
        else:
            clustering_result = general_clustering(G, method)

        # Load node labels
        labels_df = pd.read_csv(labels_file_path)

        # Evaluate clustering
        accuracy, recall, f1 = evaluate_clustering(clustering_result, labels_df)

        print(f"Results for {method} clustering:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}\n")

        # Save clustering results to a file
        df_nodes = pd.DataFrame({'id': list(G.nodes())})
        df_nodes['cluster'] = df_nodes['id'].map(clustering_result)
        output_file = f"clustering_results_{method}.csv"
        df_nodes.to_csv(output_file, index=False)
        print(f"Clustering results for {method} saved to {output_file}\n")

if __name__ == "__main__":
    main()
