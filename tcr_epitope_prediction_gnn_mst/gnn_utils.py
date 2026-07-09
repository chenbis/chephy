import numpy as np
import torch
from torch_geometric.data import HeteroData
from scipy.sparse.csgraph import minimum_spanning_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_esm2(model_name="esm2_t6_8M_UR50D"):
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

@torch.no_grad()
def esm_embed_sequences(seqs, model, batch_converter, pooling="mean"):
    """
    seqs: list[str]
    returns: torch.Tensor shape [N, D]
    """
    data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
    _, _, toks = batch_converter(data)
    toks = toks.to(device)

    out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    reps = out["representations"][model.num_layers]  # [B, L, D]


    mask = toks != 1
    mask[:, 0] = False  # CLS

    # pooling
    if pooling == "mean":
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        emb = (reps * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
    elif pooling == "cls":
        emb = reps[:, 0, :]  # CLS
    else:
        raise ValueError("pooling must be 'mean' or 'cls'")

    return emb  # [N, D]

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
    esm_model, batch_converter = load_esm2("esm2_t6_8M_UR50D")

    # --- Step 3: TCR similarity edges via MST/thresholds using only ChePhy distance ---
    if use_mst:
        # print("computing MST")
        mst_sparse = minimum_spanning_tree(chephy_matrix)
        coo = mst_sparse.tocoo()
        mst_edges = list(zip(coo.row, coo.col, coo.data))

        selected_edges = mst_edges

        if enrich:
            # print("enrichment")
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
        tcr_sim_edge_index = (
            torch.from_numpy(edge_array) if edge_array.size else torch.empty((2, 0), dtype=torch.long)
        )

    else:
        rows, cols = np.where((chephy_matrix < distance_threshold) & (chephy_matrix > 0))  # Exclude self-loops
        edge_pairs = list(zip(rows, cols))    
        
        tcr_sim_edge_index = torch.tensor(edge_pairs, dtype=torch.long).T  # shape [2, num_edges]


    epitope_features = esm_embed_sequences(epitopes, esm_model, batch_converter, pooling="mean").to(device)

    tcr_features = esm_embed_sequences(sequences_list, esm_model, batch_converter, pooling="mean").to(device)


    # --- Step 2: Bipartite binding edges from raw DataFrame ---
    edge_index = torch.tensor(
        data[['tcr_index', 'epitope_index']].dropna().astype(int).values.T,
        dtype=torch.long
    )
    rev_edge_index = torch.tensor(
        data[['epitope_index', 'tcr_index']].dropna().astype(int).values.T,
        dtype=torch.long
    )

    # --- Step 4: Construct HeteroData object ---
    data_hetero = HeteroData()
    data_hetero['tcr'].x = tcr_features
    data_hetero['peptide'].x = epitope_features
    data_hetero['tcr', 'binds', 'peptide'].edge_index = edge_index
    # data_hetero['epitope', 'rev_binds', 'tcr'].edge_index = rev_edge_index
    data_hetero['tcr', 'similar', 'tcr'].edge_index = tcr_sim_edge_index
    data_hetero = data_hetero.to(device)

    # --- Step 5: Optional multi-label targets ---
    num_tcr = len(sequences_list)
    num_epitopes = len(epitopes)
    tcr_labels = torch.zeros((num_tcr, num_epitopes), dtype=torch.float)
    for row in data.itertuples():
        if row.tcr_index != -1 and row.epitope_index != -1:
            tcr_labels[row.tcr_index, row.epitope_index] = 1.0

    return data_hetero, tcr_labels

