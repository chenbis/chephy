#!/usr/bin/env python3
"""Train a simple TCR-epitope binding predictor (MLP or hetero GNN).

Usage:
    python tcr_epitope/train_model.py --train train_small.tsv --test test_small.tsv --model gnn
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import chephy_model as cpm

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Optional PyG imports (used only for hetero GNN)
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except Exception:
    HeteroData = None
    HeteroConv = None
    SAGEConv = None

from sklearn.preprocessing import StandardScaler


def scaler_from_state(state: dict) -> StandardScaler:
    sc = StandardScaler(with_mean=True, with_std=True)
    sc.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    sc.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc

def load_and_prepare(path: str, cdr3_header: str = "cdr3", label_header: str = "peptide") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=[cdr3_header, label_header])
    df = df[df[cdr3_header].str.match(r'^[ARNDCEQGHILKMFPSTWYV]+$', na=False)]
    # Keep only sufficiently long TCRs
    df = df[df[cdr3_header].str.len() >= 8]
    # truncate TCRs (center truncation to length 8)
    df = cpm.truncate_sequences(df, cdr3_header, right=4, left=4)
    return df


class PairDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tcr_features: np.ndarray, epi_features: np.ndarray):
        self.rows = rows.reset_index(drop=True)
        self.tcr_features = tcr_features
        self.epi_features = epi_features
        # mode: 'mlp' returns concatenated feature vector; 'gnn' returns indices
        self.mode = 'mlp'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        t_idx = int(r['tcr_index'])
        e_idx = int(r['epitope_index'])
        y = float(r.get('Binding', r.get('binding', 0)))
        if self.mode == 'mlp':
            x_t = self.tcr_features[t_idx]
            x_e = self.epi_features[e_idx]
            x = np.concatenate([x_t, x_e]).astype(np.float32)
            return x, y
        else:
            # for GNN training we return indices to look up node embeddings
            return int(t_idx), int(e_idx), y


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeteroGNN(nn.Module):
    """Heterogeneous GNN using PyG HeteroConv + SAGEConv per relation.

    Expects node feature dict {'tcr': Tensor, 'epitope': Tensor} and edge_index_dict.
    """
    def __init__(self, tcr_in: int, epi_in: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        # initial linear projections per node type
        self.lin_t = nn.Linear(tcr_in, hidden)
        self.lin_e = nn.Linear(epi_in, hidden)

        # build hetero conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('tcr', 'similar', 'tcr'): SAGEConv((-1, -1), hidden),
                    ('tcr', 'binds', 'epitope'): SAGEConv((-1, -1), hidden),
                    ('epitope', 'rev_binds', 'tcr'): SAGEConv((-1, -1), hidden),
                },
                aggr='sum',
            )
            self.convs.append(conv)

        self.act = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # project
        x_dict = {
            'tcr': self.lin_t(x_dict['tcr']),
            'epitope': self.lin_e(x_dict['epitope']),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.act(v) for k, v in x_dict.items()}

        return x_dict


def train(
    train_path: str,
    test_path: str,
    out_model: str = 'tcr_epitope_model.pt',
    epochs: int = 10,
    model_type: str = 'mlp',
):
    train_df = load_and_prepare(train_path)
    test_df = load_and_prepare(test_path)

    # === define vocab from TRAIN ONLY (inductive-friendly) ===
    train_sequences = list(dict.fromkeys(train_df['cdr3_truncated'].tolist()))
    train_epitopes = list(dict.fromkeys(train_df['peptide'].tolist()))
    all_tcrs = train_sequences
    all_epitopes = train_epitopes

    atchley_map = cpm.load_atchley_map()

    # === build features + fit scalers on TRAIN vocab ===
    tcr_feats_all, tcr_scaler = cpm.sequences_to_features(
        all_tcrs,
        atchley=atchley_map,
        standardize=True,
        scaler=None,
    )
    epi_feats_all, epi_scaler = cpm.sequences_to_features_any_length(
        all_epitopes,
        atchley=atchley_map,
        standardize=True,
        scaler=None,
        agg='pad',
    )

    # === map sequences to indices ===
    tcr_to_idx = {s: i for i, s in enumerate(all_tcrs)}
    epi_to_idx = {s: i for i, s in enumerate(all_epitopes)}

    # map train/test; drop any rows with unseen sequences
    for df in (train_df, test_df):
        df['tcr_index'] = df['cdr3_truncated'].map(tcr_to_idx)
        df['epitope_index'] = df['peptide'].map(epi_to_idx)
        df.dropna(subset=['tcr_index', 'epitope_index'], inplace=True)
        df[['tcr_index', 'epitope_index']] = df[['tcr_index', 'epitope_index']].astype(int)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'mlp':
        # Datasets
        train_ds = PairDataset(train_df, tcr_feats_all, epi_feats_all)
        test_ds = PairDataset(test_df, tcr_feats_all, epi_feats_all)
        batch_size = 128
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        sample_x, _ = train_ds[0]
        model = MLP(input_dim=sample_x.shape[0], hidden=256).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # eval
            model.eval()
            ys = []
            preds = []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)
                    logits = model(X)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    preds.extend(prob.tolist())
                    ys.extend(y.tolist())

            try:
                auc = roc_auc_score(ys, preds)
            except Exception:
                auc = float('nan')
            acc = accuracy_score(ys, [1 if p >= 0.5 else 0 for p in preds])
            print(f"Epoch {epoch}/{epochs}  train_loss={np.mean(losses):.4f}  test_auc={auc:.4f}  test_acc={acc:.4f}")

        # save model and metadata (MLP)
        os.makedirs('models', exist_ok=True)
        torch.save(
            {
                'model_state': model.state_dict(),
                'input_dim': sample_x.shape[0],
                'model_type': 'mlp',
                # optional: also save scalers for inductive MLP
                'tcr_scaler_state': {
                    'mean_': tcr_scaler.mean_,
                    'scale_': tcr_scaler.scale_,
                },
                'epi_scaler_state': {
                    'mean_': epi_scaler.mean_,
                    'scale_': epi_scaler.scale_,
                },
            },
            out_model,
        )
        print('Saved MLP model to', out_model)

    elif model_type == 'gnn':
        # GNN: build node list and adjacency from TRAIN vocab
        if HeteroData is None or HeteroConv is None:
            raise ImportError('PyTorch Geometric (torch_geometric) is required for hetero GNN training; please install it to use --model gnn')

        # Build TCR-TCR kNN similarity graph on TRAIN TCRs
        all_tcrs_full = list(tcr_to_idx.keys())
        knn_k = 10
        tcr_edge_index, tcr_edge_weight, _, _ = cpm.build_patient_knn_graph(
            all_tcrs_full,
            k=knn_k,
            atchley_map=atchley_map,
            standardize_features=True,  # only affects distance space, not tcr_feats_all
            weighting='self_tuning',
            mutual=True,
            knn_cfg=cpm.KNNConfig(k=knn_k, use_ann_if_large=True, ann_threshold=5000),
        )

        # build HeteroData
        data_hetero = HeteroData()
        data_hetero['tcr'].x = torch.tensor(tcr_feats_all, dtype=torch.float32)
        data_hetero['epitope'].x = torch.tensor(epi_feats_all, dtype=torch.float32)

        # tcr-tcr edges (relation: 'similar')
        if tcr_edge_index is not None and tcr_edge_index.size > 0:
            data_hetero['tcr', 'similar', 'tcr'].edge_index = torch.tensor(tcr_edge_index, dtype=torch.long)
            if tcr_edge_weight is not None:
                data_hetero['tcr', 'similar', 'tcr'].edge_weight = torch.tensor(tcr_edge_weight, dtype=torch.float32)
        else:
            data_hetero['tcr', 'similar', 'tcr'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # binding edges: use train+test pairs for message passing + evaluation
        # combined_rows = pd.concat([train_df, test_df], ignore_index=True)
        combined_rows = train_df.copy()
        if not combined_rows.empty:
            tcr_epi = torch.tensor(
                combined_rows[['tcr_index', 'epitope_index']].astype(int).values.T,
                dtype=torch.long,
            )
            tcr_epi_rev = torch.tensor(
                combined_rows[['epitope_index', 'tcr_index']].astype(int).values.T,
                dtype=torch.long,
            )
        else:
            tcr_epi = torch.empty((2, 0), dtype=torch.long)
            tcr_epi_rev = torch.empty((2, 0), dtype=torch.long)

        data_hetero['tcr', 'binds', 'epitope'].edge_index = tcr_epi
        data_hetero['epitope', 'rev_binds', 'tcr'].edge_index = tcr_epi_rev

        # move to device
        data_hetero = data_hetero.to(device)

        # prepare x_dict and edge_index_dict for HeteroConv
        x_dict = {
            'tcr': data_hetero['tcr'].x,
            'epitope': data_hetero['epitope'].x,
        }
        edge_index_dict = {
            ('tcr', 'similar', 'tcr'): data_hetero['tcr', 'similar', 'tcr'].edge_index,
            ('tcr', 'binds', 'epitope'): data_hetero['tcr', 'binds', 'epitope'].edge_index,
            ('epitope', 'rev_binds', 'tcr'): data_hetero['epitope', 'rev_binds', 'tcr'].edge_index,
        }

        # dataset returns indices
        train_ds = PairDataset(train_df, tcr_feats_all, epi_feats_all)
        test_ds = PairDataset(test_df, tcr_feats_all, epi_feats_all)
        train_ds.mode = 'gnn'
        test_ds.mode = 'gnn'

        batch_size = 128
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # build hetero GNN
        model = HeteroGNN(
            tcr_in=tcr_feats_all.shape[1],
            epi_in=epi_feats_all.shape[1],
            hidden=256,
            num_layers=2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc = -float("inf")
        best_state = None
        best_epoch = -1

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                # batch is (t_idx, e_idx, y) tuples
                if isinstance(batch[0], torch.Tensor):
                    t_idx = batch[0].to(device)
                    e_idx = batch[1].to(device)
                    y = batch[2].to(device).float()
                else:
                    t_idx, e_idx, y = batch
                    if not isinstance(t_idx, torch.Tensor):
                        t_idx = torch.tensor(t_idx, device=device)
                    else:
                        t_idx = t_idx.to(device)
                    if not isinstance(e_idx, torch.Tensor):
                        e_idx = torch.tensor(e_idx, device=device)
                    else:
                        e_idx = e_idx.to(device)
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y, device=device).float()
                    else:
                        y = y.float().to(device)

                out_dict = model(x_dict, edge_index_dict)
                t_emb = out_dict['tcr'][t_idx]
                e_emb = out_dict['epitope'][e_idx]
                pair_emb = torch.cat([t_emb, e_emb], dim=1)
                logits = model.head(pair_emb).squeeze(-1)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # eval
            model.eval()
            ys = []
            preds = []
            with torch.no_grad():
                out_dict = model(x_dict, edge_index_dict)
                for batch in test_loader:
                    t_idx, e_idx, y = batch
                    if not isinstance(t_idx, torch.Tensor):
                        t_idx = torch.tensor(t_idx, device=device)
                    else:
                        t_idx = t_idx.to(device)
                    if not isinstance(e_idx, torch.Tensor):
                        e_idx = torch.tensor(e_idx, device=device)
                    else:
                        e_idx = e_idx.to(device)
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y, device=device).float()
                    else:
                        y = y.float().to(device)

                    t_emb = out_dict['tcr'][t_idx]
                    e_emb = out_dict['epitope'][e_idx]
                    pair_emb = torch.cat([t_emb, e_emb], dim=1)
                    logits = model.head(pair_emb).squeeze(-1)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    preds.extend(prob.tolist())
                    ys.extend(y.cpu().numpy().tolist())

            try:
                auc = roc_auc_score(ys, preds)
            except Exception:
                auc = float('nan')
            acc = accuracy_score(ys, [1 if p >= 0.5 else 0 for p in preds])
            print(f"Epoch {epoch}/{epochs}  train_loss={np.mean(losses):.4f}  test_auc={auc:.4f}  test_acc={acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_state = {
                    'model_state': model.state_dict(),
                    'model_type': 'gnn_hetero',
                    'tcr_in': tcr_feats_all.shape[1],
                    'epi_in': epi_feats_all.shape[1],
                    'tcr_scaler_state': {
                        'mean_': tcr_scaler.mean_,
                        'scale_': tcr_scaler.scale_,
                    },
                    'epi_scaler_state': {
                        'mean_': epi_scaler.mean_,
                        'scale_': epi_scaler.scale_,
                    },
                    'knn_k': knn_k,
                    'best_epoch': epoch,
                    'best_auc': auc,
                }


        # save model and metadata (GNN + scalers for inductive inference)
        os.makedirs('models', exist_ok=True)
        if best_state is not None:
            torch.save(best_state, out_model)
            print(f"Saved best model (epoch={best_epoch}, auc={best_auc:.4f}) to {out_model}")
        else:
            print("No valid model was saved (AUC was never computed).")

    else:
        raise ValueError(f'Unknown model_type: {model_type}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='train_small.tsv')
    p.add_argument('--test', default='test_small.tsv')
    p.add_argument('--out', default='models/tcr_epitope_model.pt')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--model', choices=['mlp', 'gnn'], default='mlp')
    args = p.parse_args()
    train(args.train, args.test, out_model=args.out, epochs=args.epochs, model_type=args.model)


if __name__ == '__main__':
    main()
