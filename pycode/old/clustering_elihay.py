###########
# cluster 10k
###########
import os
import csv
from itertools import combinations
from collections import defaultdict
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, label_propagation_communities
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# Define file paths
input_file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/regression/train_tcrs_across_samples_10K_cloness.csv'
output_dir = os.path.dirname(input_file_path)  # Use the same folder as the input file

# Output file paths for clusters
hashed_substring_clusters_file = os.path.join(output_dir, "hashed_substring_clusters.txt")
louvain_clusters_file = os.path.join(output_dir, "louvain_clusters.txt")
label_propagation_clusters_file = os.path.join(output_dir, "label_propagation_clusters.txt")
dbscan_clusters_file = os.path.join(output_dir, "dbscan_clusters.txt")
connected_components_clusters_file = os.path.join(output_dir, "connected_components_clusters.txt")

# Step 1: Load Data and Select TCR Columns
data = pd.read_csv(input_file_path)
columns_to_exclude = ['sample name', 'Biological Sex', 'Age']
tcr_columns = [col for col in data.columns if col not in columns_to_exclude]  # Select only TCR columns
sequences = tcr_columns  # TCR sequences are the column names

# Step 2: Generate Hashed Substrings and Build Graph
def generate_hashed_substrings(sequence):
    for substring in combinations(sequence, len(sequence) - 1):
        yield hash(substring), sequence

# Group sequences by length
sequences_by_length = defaultdict(list)
for seq in sequences:
    if isinstance(seq, str) and len(seq) > 1:
        sequences_by_length[len(seq)].append(seq)

graph = nx.Graph()
edge_list = []

for length, seq_group in sorted(sequences_by_length.items(), reverse=True):
    print(f"Processing sequences of length {length}...")
    substring_map = defaultdict(list)

    # Map sequences by their hashed substrings
    for seq in seq_group:
        for hashed_substring, sequence in generate_hashed_substrings(seq):
            substring_map[hashed_substring].append(sequence)

    # Generate edges from hashed substring groups
    for group in substring_map.values():
        if len(group) > 1:
            edge_list.extend(combinations(group, 2))

# Add edges to the graph
graph.add_edges_from(edge_list)
print(f"Graph constructed with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

# Step 3: Hashed Substring Clustering (Cliques and Components)
# Find cliques
print("Finding cliques...")
cliques = list(nx.find_cliques(graph))
with open(hashed_substring_clusters_file, "w") as f:
    for i, clique in enumerate(cliques):
        f.write(f"Clique {i + 1}: {', '.join(clique)}\n")


# Step 4: Apply Clustering Methods
# 1. Louvain Method
print("Applying Louvain Method...")
louvain_clusters = louvain_communities(graph)
with open(louvain_clusters_file, "w") as f:
    for i, community in enumerate(louvain_clusters):
        f.write(f"Community {i + 1}: {', '.join(community)}\n")
        
# 2. Label Propagation
print("Applying Label Propagation...")
label_propagation_clusters = list(label_propagation_communities(graph))
with open(label_propagation_clusters_file, "w") as f:
    for i, community in enumerate(label_propagation_clusters):
        f.write(f"Community {i + 1}: {', '.join(community)}\n")

# 3. DBSCAN
print("Applying DBSCAN...")
adj_matrix = nx.to_numpy_array(graph)
distance_matrix = 1 - adj_matrix
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="precomputed")
dbscan_labels = dbscan.fit_predict(distance_matrix)
dbscan_clusters = defaultdict(list)
for node, label in zip(graph.nodes, dbscan_labels):
    dbscan_clusters[label].append(node)
with open(dbscan_clusters_file, "w") as f:
    for label, cluster in dbscan_clusters.items():
        cluster_type = "Noise" if label == -1 else f"Cluster {label + 1}"
        f.write(f"{cluster_type}: {', '.join(cluster)}\n")

# 4. Connected Components
print("Finding Connected Components...")
connected_components = list(nx.connected_components(graph))
with open(connected_components_clusters_file, "w") as f:
    for i, component in enumerate(connected_components):
        f.write(f"Component {i + 1}: {', '.join(component)}\n")
print("Clustering completed and results saved.")