import numpy as np
import torch
from torch_geometric.data import Data
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

esm_model, batch_converter = load_esm2("esm2_t6_8M_UR50D")

def build_graph(
    chephy_matrix,
    sequences_list,
    label=None,
    node_weights=None,
    distance_threshold=0.5,
    use_mst=True,
    enrich=True,
    use_edge_weights=True,
):
    # esm_model, batch_converter = load_esm2("esm2_t6_8M_UR50D")

    # ---------- Build directed edge list, then symmetrize ----------
    if use_mst:
        mst_sparse = minimum_spanning_tree(chephy_matrix)
        coo = mst_sparse.tocoo()
        mst_edges = list(zip(coo.row, coo.col, coo.data))

        selected_edges = list(mst_edges)

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

        base_edges = np.array([(u, v) for u, v, _ in selected_edges], dtype=np.int64)

    else:
        rows, cols = np.where((chephy_matrix < distance_threshold) & (chephy_matrix > 0))
        base_edges = np.array(list(zip(rows, cols)), dtype=np.int64)

    # Symmetrize to represent an undirected graph in PyG
    if base_edges.size:
        edge_array = np.concatenate([base_edges, base_edges[:, [1, 0]]], axis=0)
        edge_index = torch.from_numpy(edge_array.T).contiguous()
    else:
        edge_array = np.empty((0, 2), dtype=np.int64)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # ---------- Edge weights (computed once, aligned with edge_index) ----------
    edge_weight = None
    if use_edge_weights and edge_array.size:
        # Distance from matrix
        dist = np.array([chephy_matrix[i, j] for i, j in edge_array], dtype=np.float32)
        dist_t = torch.from_numpy(dist)

        # Convert distance -> similarity (your original logic)
        edge_weight = torch.exp(-dist_t)

    # ---------- Node features ----------
    x = esm_embed_sequences(
        sequences_list, esm_model, batch_converter, pooling="mean"
    ).to(device)

    # ---------- Build Data ----------
    data_dict = {"x": x, "edge_index": edge_index}

    if edge_weight is not None:
        data_dict["edge_weight"] = edge_weight.to(device)

    if node_weights is not None:
        data_dict["node_weights"] = torch.tensor(node_weights, dtype=torch.float32).to(device)

    if label is not None:
        data_dict["y"] = torch.tensor([label]).to(device)

    # for feature extraction
    data_dict["node_seqs"] = sequences_list

    data = Data(**data_dict).to(device)
    return data
