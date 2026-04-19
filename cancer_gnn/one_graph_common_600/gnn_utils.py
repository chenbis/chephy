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

esm_model, batch_converter = load_esm2("esm2_t6_8M_UR50D")

def mst_enriched_edges_from_distance_matrix(
    chephy_matrix: np.ndarray,
    distance_threshold: float = 0.5,
    use_mst: bool = True,
    enrich: bool = True,
):
    """
    Returns:
      edge_array: np.ndarray shape [E, 2] with directed edges (i->j) for an undirected PyG graph
      dist_array: np.ndarray shape [E] distances aligned with edge_array
    This preserves your exact MST + enrichment logic, then symmetrizes.
    """

    if use_mst:
        mst_sparse = minimum_spanning_tree(chephy_matrix)
        coo = mst_sparse.tocoo()
        mst_edges = list(zip(coo.row, coo.col, coo.data))  # (u,v,dist)

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

    # Symmetrize for undirected representation
    if base_edges.size:
        edge_array = np.concatenate([base_edges, base_edges[:, [1, 0]]], axis=0)
        dist_array = np.array([chephy_matrix[i, j] for i, j in edge_array], dtype=np.float32)
    else:
        edge_array = np.empty((0, 2), dtype=np.int64)
        dist_array = np.empty((0,), dtype=np.float32)

    return edge_array, dist_array


def build_global_hetero_graph(
    chephy_matrix: np.ndarray,
    sequences_list: list,                 # global unique CDR3 list in SAME ORDER as chephy_matrix
    patient_repertoires: dict,            # {patient_id: list of (cdr3_seq, count)} OR (cdr3_seq, weight)
    patient_labels: dict = None,          # {patient_id: 0/1} optional
    patient_features: dict = None,        # {patient_id: np.ndarray or torch.Tensor} optional
    distance_threshold: float = 0.5,
    use_mst: bool = True,
    enrich: bool = True,
    use_edge_weights: bool = True,        # on cdr3-sim edges: weight = exp(-dist)
    membership_log1p: bool = True,        # on patient->cdr3 edges: weight = log1p(count)
    pooling: str = "mean",
):
    """
    Builds one global heterograph:
      - cdr3 nodes: ESM embeddings from sequences_list
      - patient nodes: optional features OR zeros
      - edges:
          ('cdr3','sim','cdr3') from MST+enrichment
          ('patient','has','cdr3') membership (with edge_attr weights)
          ('cdr3','rev_has','patient') reverse membership
    """

    # ---------- Map sequences/patients to indices ----------
    seq_to_idx = {s: i for i, s in enumerate(sequences_list)}
    patient_ids = list(patient_repertoires.keys())
    pid_to_idx = {p: i for i, p in enumerate(patient_ids)}

    data = HeteroData()

    # ---------- cdr3 node features ----------
    cdr3_x = esm_embed_sequences(sequences_list, esm_model, batch_converter, pooling=pooling).to(device)
    data["cdr3"].x = cdr3_x
    data["cdr3"].node_seqs = sequences_list  # handy for debugging

    # ---------- patient node features ----------
    if patient_features is None:
        # default: learnable signal can still come via messages; but we need a tensor
        # Here: a single zero feature per patient
        patient_x = torch.zeros((len(patient_ids), 1), dtype=torch.float32, device=device)
    else:
        # stack in patient_ids order
        feats = []
        for p in patient_ids:
            v = patient_features[p]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            feats.append(v.float().view(-1))
        patient_x = torch.stack(feats, dim=0).to(device)
    data["patient"].x = patient_x
    data["patient"].patient_ids = patient_ids

    # ---------- patient labels ----------
    if patient_labels is not None:
        y = torch.tensor([patient_labels[p] for p in patient_ids], dtype=torch.long, device=device)
        data["patient"].y = y

    # ---------- Build cdr3<->cdr3 similarity edges (KEEP MST+ENRICH) ----------
    edge_array, dist_array = mst_enriched_edges_from_distance_matrix(
        chephy_matrix=chephy_matrix,
        distance_threshold=distance_threshold,
        use_mst=use_mst,
        enrich=enrich,
    )

    # edge_index for ('cdr3','sim','cdr3')
    sim_edge_index = torch.from_numpy(edge_array.T).long()
    data[("cdr3", "sim", "cdr3")].edge_index = sim_edge_index.to(device)

    if use_edge_weights and dist_array.size:
        # your original logic: similarity = exp(-dist)
        edge_weight = torch.exp(-torch.from_numpy(dist_array).float())
        data[("cdr3", "sim", "cdr3")].edge_weight = edge_weight.to(device)

    # ---------- Build membership edges patient -> cdr3 (and reverse) ----------
    src_p, dst_s, w = [], [], []
    for p in patient_ids:
        pi = pid_to_idx[p]
        for s, count in patient_repertoires[p]:
            if s not in seq_to_idx:
                # If you want strict behavior: skip unknown sequences
                # If you want to add new cdr3 nodes dynamically, do that elsewhere.
                continue
            si = seq_to_idx[s]
            src_p.append(pi)
            dst_s.append(si)
            if membership_log1p:
                w.append(np.log1p(float(count)))
            else:
                w.append(float(count))

    if len(src_p) == 0:
        # allow empty (but model will be dead)
        p2s_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        p2s_w = torch.empty((0, 1), dtype=torch.float32, device=device)
    else:
        p2s_edge_index = torch.tensor([src_p, dst_s], dtype=torch.long, device=device)
        p2s_w = torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1)

    data[("patient", "has", "cdr3")].edge_index = p2s_edge_index
    data[("patient", "has", "cdr3")].edge_attr = p2s_w

    # reverse edges
    data[("cdr3", "rev_has", "patient")].edge_index = p2s_edge_index.flip(0)
    data[("cdr3", "rev_has", "patient")].edge_attr = p2s_w

    return data
