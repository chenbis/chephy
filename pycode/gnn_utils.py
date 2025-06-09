import numpy as np
import torch
from torch_geometric.data import HeteroData
from scipy.sparse.csgraph import minimum_spanning_tree



def get_aa_interaction_features(sequence, atchley_dict):
    """
    Compute interaction features between adjacent amino acids
    using their Atchley factors
    """
    feature_dim = len(next(iter(atchley_dict.values())))
    vecs = [list(atchley_dict[aa].values()) for aa in sequence if aa in atchley_dict]
    if len(vecs) < 2:
        return np.zeros(feature_dim)
        
    mat = np.array(vecs)
    
    # Compute interactions between adjacent AAs
    interactions = []
    for i in range(len(mat)-1):
        # Element-wise product captures chemical property interactions
        prod = mat[i] * mat[i+1]
        # Sum captures overall interaction strength
        sum_vec = mat[i] + mat[i+1]
        # Difference captures property changes
        diff = np.abs(mat[i] - mat[i+1])
        interactions.extend([prod, sum_vec, diff])
    
    interactions = np.stack(interactions)
    
    # Aggregate interaction features
    mean_interactions = interactions.mean(axis=0)
    max_interactions = interactions.max(axis=0)
    std_interactions = interactions.std(axis=0)
    
    return np.concatenate([mean_interactions, max_interactions, std_interactions])

def get_hybrid_sequence_features(sequence, atchley_dict):
    """
    Hybrid encoder combining:
    1. Best features from both previous versions
    2. AA interaction features
    3. Overlapping position windows
    """
    feature_dim = len(next(iter(atchley_dict.values())))
    vecs = [list(atchley_dict[aa].values()) for aa in sequence if aa in atchley_dict]
    if not vecs:
        vecs = [np.zeros(feature_dim)]
    mat = np.array(vecs)
    seq_len = len(mat)
    
    # 1. Global statistics
    global_mean = mat.mean(axis=0)
    global_std = mat.std(axis=0)
    global_max = mat.max(axis=0)
    
    # 2. Enhanced k-mer features (k=3)
    k = 3
    kmer_features = []
    for i in range(len(mat) - k + 1):
        kmer = mat[i:i+k]
        kmer_mean = kmer.mean(axis=0)
        kmer_max = kmer.max(axis=0)
        kmer_features.extend([kmer_mean, kmer_max])
    
    if kmer_features:
        kmer_features = np.stack(kmer_features)
        kmer_stats = np.concatenate([
            kmer_features.mean(axis=0),
            kmer_features.max(axis=0)
        ])
    else:
        kmer_stats = np.zeros(feature_dim * 2)
    
    # 3. Position-aware features with overlap
    n_segments = 3
    position_features = []
    for i in range(n_segments):
        start = max(0, (i * seq_len) // n_segments - 1)  # Overlap by 1
        end = min(seq_len, ((i + 1) * seq_len) // n_segments + 1)
        segment = mat[start:end]
        position_features.append(segment.mean(axis=0) if len(segment) > 0 else np.zeros(feature_dim))
    position_features = np.concatenate(position_features)
    
    # 4. AA interaction features
    interaction_features = get_aa_interaction_features(sequence, atchley_dict)
    
    return torch.tensor(np.concatenate([
        global_mean, global_std, global_max,  # 15 features
        kmer_stats,  # 10 features
        position_features,  # 15 features
        interaction_features  # 15 features
    ]), dtype=torch.float32)

def build_edges_with_mst_and_enrichment(
    chephy_matrix,
    sequence_indices,
    hamming_matrix=None,
    chephy_threshold=None,
    hamming_threshold=None,
    use_mst=True,
    enrich=True
):
    """
    Build edge_index and edge_attr arrays from dense ChePhy distances, optionally using MST and enrichment.

    Parameters:
        chephy_matrix (np.ndarray): Square ChePhy distance matrix.
        sequence_indices (np.ndarray): Mapping of node index to global sequence index.
        hamming_matrix (np.ndarray, optional): Square Hamming distance matrix.
        chephy_threshold (float, optional): Max allowed ChePhy distance for enrichment.
        hamming_threshold (float, optional): Max allowed Hamming distance for enrichment.
        use_mst (bool): Whether to build MST as backbone.
        enrich (bool): Whether to add extra edges passing threshold filters.

    Returns:
        edge_index (np.ndarray): Shape (2, num_edges).
        edge_attr (np.ndarray): Shape (num_edges,).
    """
    n = len(sequence_indices)
    edge_set = set()
    edge_list = []

    if use_mst:
        mst = minimum_spanning_tree(chephy_matrix)
        mst = mst.toarray()
        for i in range(n):
            for j in range(n):
                if mst[i, j] > 0:
                    edge_set.add((i, j))
                    edge_list.append((sequence_indices[i], sequence_indices[j], chephy_matrix[i, j]))
        print("mst done")

    if enrich and (chephy_threshold is not None or hamming_threshold is not None):
        chephy_mask = (chephy_matrix <= chephy_threshold) if chephy_threshold is not None else np.ones_like(chephy_matrix, dtype=bool)
        hamming_mask = (hamming_matrix <= hamming_threshold) if (hamming_matrix is not None and hamming_threshold is not None) else np.ones_like(chephy_matrix, dtype=bool)
        combined_mask = chephy_mask & hamming_mask

        for i in range(n):
            for j in range(n):
                if i != j and combined_mask[i, j] and (i, j) not in edge_set:
                    edge_list.append((sequence_indices[i], sequence_indices[j], chephy_matrix[i, j]))
        print("enrichment done")

    # Build edge_index and edge_attr
    src = [i for i, j, _ in edge_list]
    dst = [j for i, j, _ in edge_list]
    edge_index = np.array([src, dst])
    edge_attr = np.array([dist for _, _, dist in edge_list], dtype=np.float32)

    return edge_index, edge_attr

def build_hetero_graph(
    data,
    chephy_matrix,
    sequences_list,
    epitopes,
    atchley_dict,
    distance_threshold=0.5,
    use_mst=True,
    enrich=True
):
    N = len(sequences_list)
    tcr_sim_edge_index = None  # Initialize as None instead of an empty string

    # --- Step 3: TCR similarity edges via MST/thresholds using only ChePhy distance ---
    if use_mst:
        print("computing MST")
        # Use scipy's minimum_spanning_tree for MST computation
        mst_sparse = minimum_spanning_tree(chephy_matrix)
        print("scipy done")
        mst_dense = mst_sparse.toarray()
        print("toarray done")
        mst_edges = []
        for i in range(N):
            for j in range(N):
                if mst_dense[i, j] > 0:
                    mst_edges.append((i, j, mst_dense[i, j]))

        selected_edges = mst_edges

        if enrich:
            print("enrichment")
            # Add extra edges within threshold without duplicating MST edges
            mst_edge_set = set((u, v) for u, v, _ in mst_edges)
            for i in range(N):
                for j in range(i + 1, N):
                    weight = chephy_matrix[i, j]
                    if weight <= distance_threshold and (i, j) not in mst_edge_set:
                        selected_edges.append((i, j, weight))

        edge_array = np.array([(u, v) for u, v, _ in selected_edges], dtype=np.int64).T
        tcr_sim_edge_index = (
            torch.from_numpy(edge_array) if edge_array.size else torch.empty((2, 0), dtype=torch.long)
        )

    else:
        print("skip mst")
        rows, cols = np.where((chephy_matrix < distance_threshold) & (chephy_matrix > 0))  # Exclude self-loops
        edge_pairs = list(zip(rows, cols))    
        
        tcr_sim_edge_index = torch.tensor(edge_pairs, dtype=torch.long).T  # shape [2, num_edges]
    print("Step 1, Done!")

    # --- Step 1: Node features ---
    feature_dim = len(next(iter(atchley_dict.values())))
    epitope_features = torch.stack([
        get_hybrid_sequence_features(epi, atchley_dict) for epi in epitopes
    ])

    tcr_features = torch.stack([
        get_hybrid_sequence_features(seq, atchley_dict) for seq in sequences_list
    ])
    print("Step 2, Done!")

    # --- Step 2: Bipartite binding edges from raw DataFrame ---
    edge_index = torch.tensor(
        data[['tcr_index', 'epitope_index']].dropna().astype(int).values.T,
        dtype=torch.long
    )
    rev_edge_index = torch.tensor(
        data[['epitope_index', 'tcr_index']].dropna().astype(int).values.T,
        dtype=torch.long
    )
    print("Step 3, Done!")

    # --- Step 4: Construct HeteroData object ---
    data_hetero = HeteroData()
    data_hetero['tcr'].x = tcr_features
    data_hetero['epitope'].x = epitope_features
    data_hetero['tcr', 'binds', 'epitope'].edge_index = edge_index
    data_hetero['epitope', 'rev_binds', 'tcr'].edge_index = rev_edge_index
    data_hetero['tcr', 'similar', 'tcr'].edge_index = tcr_sim_edge_index
    print("Step 4, Done!")

    # --- Step 5: Optional multi-label targets ---
    num_tcr = len(sequences_list)
    num_epitopes = len(epitopes)
    tcr_labels = torch.zeros((num_tcr, num_epitopes), dtype=torch.float)
    for row in data.itertuples():
        if row.tcr_index != -1 and row.epitope_index != -1:
            tcr_labels[row.tcr_index, row.epitope_index] = 1.0
    print("Step 5, Done!")

    return data_hetero, tcr_labels

