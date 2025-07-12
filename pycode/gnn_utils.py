import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from scipy.sparse.csgraph import minimum_spanning_tree
import esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ESM model globally to avoid reloading
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(device)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()


def get_esm_sequence_features(sequence):
    """
    Extract ESM embeddings for a protein sequence
    Returns per-sequence representation (mean pooled over sequence length)
    """
    # Prepare data for ESM model
    data = [("seq", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Extract representations
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
    
    # Get the representation from the last layer (layer 6 for ESM2-8M)
    token_representations = results["representations"][6]
    
    # Remove special tokens and get sequence representation
    # ESM adds <cls> at start and <eos> at end, so we take tokens 1:-1
    sequence_representations = token_representations[0, 1:len(sequence)+1]
    
    # Mean pool over sequence length to get fixed-size representation
    sequence_features = sequence_representations.mean(dim=0)
    
    return sequence_features


def build_graph(
    chephy_matrix,
    sequences_list,
    atchley_dict=None,  # Made optional since ESM doesn't need it
    distance_threshold=0.5,
    use_mst=True,
    enrich=True
):
    # edge_index = None 

    if use_mst:
        mst_sparse = minimum_spanning_tree(chephy_matrix)
        coo = mst_sparse.tocoo()
        mst_edges = list(zip(coo.row, coo.col, coo.data))

        selected_edges = mst_edges

        if enrich:
            mst_edge_set = set((u, v) for u, v, _ in mst_edges)
            mask = (chephy_matrix <= distance_threshold)
            triu_mask = np.triu(mask, k=1)
            new_rows, new_cols = np.where(triu_mask)
            enriched_edges = [
                (i, j, chephy_matrix[i, j])
                for i, j in zip(new_rows, new_cols)
                if (i, j) not in mst_edge_set
            ]
            selected_edges.extend(enriched_edges)

        edge_array = np.array([(u, v) for u, v, _ in selected_edges], dtype=np.int64).T
        edge_index = (
            torch.from_numpy(edge_array) if edge_array.size else torch.empty((2, 0), dtype=torch.long)
        )

    else:
        rows, cols = np.where((chephy_matrix < distance_threshold) & (chephy_matrix > 0))  
        edge_pairs = list(zip(rows, cols))    
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).T  


    # Build node features using ESM embeddings
    x = torch.stack([
        get_esm_sequence_features(seq) for seq in sequences_list
    ])

    data = Data(x=x, edge_index=edge_index).to(device)
    # data.y = torch.tensor([label])

    return data
