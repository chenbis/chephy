#!/usr/bin/env python3
"""Evaluate model on seen vs unseen TCR-epitope pairs and optionally retrain with negative sampling.

Usage examples:
  python tcr_epitope/eval_and_neg_train.py --model models/tcr_epitope_model.pt
  python tcr_epitope/eval_and_neg_train.py --retrain --neg-factor 1.0

This script:
 - loads a saved model
 - builds preprocessing from `train_small.tsv` and `test_small.tsv` (same as training)
 - loads `files/combined_data.csv`, truncates TCRs, and splits pairs into seen/unseen relative to training positives
 - computes predictions on both groups and prints summary statistics
 - if `--retrain` is passed, creates negative samples (random non-binding pairs) and writes an augmented train file, then calls the training routine to train a new model
"""
from __future__ import annotations
import argparse
import os
import random
from typing import Tuple, Set

import numpy as np
import pandas as pd
import torch

import chephy_model as cpm
from train_model import MLP, train as train_model_fn


def read_and_truncate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=['cdr3', 'peptide'])
    df = df[df['cdr3'].str.match(r'^[ARNDCEQGHILKMFPSTWYV]+$', na=False)]
    df = df[df['cdr3'].str.len() >= 8]
    df = cpm.truncate_sequences(df, 'cdr3', 4, 4)
    return df


def load_model(path: str) -> Tuple[torch.nn.Module, int]:
    ckpt = torch.load(path, map_location='cpu')
    input_dim = ckpt.get('input_dim')
    model = MLP(input_dim=input_dim, hidden=256)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, input_dim


def make_pair_key(t: str, e: str) -> Tuple[str, str]:
    return (str(t), str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/tcr_epitope_model.pt')
    p.add_argument('--combined', default='files/combined_data.csv')
    p.add_argument('--train-small', default='tcr_epitope/train_small.tsv')
    p.add_argument('--test-small', default='tcr_epitope/test_small.tsv')
    p.add_argument('--max-samples', type=int, default=2000)
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--neg-factor', type=float, default=1.0, help='number of negatives per positive when retraining')
    p.add_argument('--make-neg-test', action='store_true', help='create negative test pairs for evaluation')
    p.add_argument('--neg-test-factor', type=float, default=1.0, help='negatives per positive for negative test set')
    p.add_argument('--out-test-neg', default='tcr_epitope/test_negatives.tsv', help='where to save generated negative test pairs')
    p.add_argument('--out-aug', default='tcr_epitope/train_small_augmented.tsv')
    p.add_argument('--out-model', default='models/tcr_epitope_model_neg.pt')
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print('Loading model...')
    model, input_dim = load_model(args.model)

    print('Preparing preprocessing using train/test small files...')
    train_df = read_and_truncate(args.train_small)
    test_df = read_and_truncate(args.test_small)

    all_tcrs = list(dict.fromkeys(list(map(str, train_df['cdr3_truncated'])) + list(map(str, test_df['cdr3_truncated']))))
    all_epitopes = list(dict.fromkeys(list(map(str, train_df['peptide'])) + list(map(str, test_df['peptide']))))

    tcr_feats_all, tcr_scaler = cpm.sequences_to_features(all_tcrs, atchley=cpm.load_atchley_map(), standardize=True)
    epi_feats_all, epi_scaler = cpm.sequences_to_features_any_length(all_epitopes, atchley=cpm.load_atchley_map(), standardize=True, agg='pad')

    # build set of training positive pairs
    pos_pairs = set()
    for _, r in train_df.iterrows():
        if int(r.get('Binding', 0)) == 1:
            pos_pairs.add(make_pair_key(r['cdr3_truncated'], r['peptide']))

    print(f'Num positive pairs in train_small: {len(pos_pairs)}')

    print('Loading combined data (may be large)...')
    combined = pd.read_csv(args.combined)
    # ensure columns exist: assume columns named cdr3 and peptide
    if 'cdr3' not in combined.columns or 'peptide' not in combined.columns:
        raise ValueError('combined_data.csv must contain columns named cdr3 and peptide')

    # truncate combined TCRs similarly
    combined = combined.dropna(subset=['cdr3', 'peptide'])
    combined = combined[combined['cdr3'].str.match(r'^[ARNDCEQGHILKMFPSTWYV]+$', na=False)]
    combined = combined[combined['cdr3'].str.len() >= 8]
    combined = cpm.truncate_sequences(combined, 'cdr3', 4, 4)

    # build seen/unseen masks
    combined['pair_key'] = combined.apply(lambda r: make_pair_key(r['cdr3_truncated'], r['peptide']), axis=1)
    combined['seen_in_train'] = combined['pair_key'].apply(lambda k: k in pos_pairs)

    seen_df = combined[combined['seen_in_train']].drop_duplicates(subset=['cdr3_truncated','peptide'])
    unseen_df = combined[~combined['seen_in_train']].drop_duplicates(subset=['cdr3_truncated','peptide'])

    print(f'Seen pairs: {len(seen_df)}, Unseen pairs: {len(unseen_df)}')

    # sample for speed
    ns = min(len(seen_df), args.max_samples)
    nu = min(len(unseen_df), args.max_samples)
    seen_sample = seen_df.sample(n=ns, random_state=42) if ns>0 else seen_df
    unseen_sample = unseen_df.sample(n=nu, random_state=42) if nu>0 else unseen_df

    # Optionally create a negative test set (random non-positive pairs)
    neg_test_probs = None
    if args.make_neg_test:
        print('Generating negative test pairs...')
        # build set of all candidate pairs from combined
        all_tcr_list = list(dict.fromkeys(combined['cdr3_truncated'].tolist()))
        all_epi_list = list(dict.fromkeys(combined['peptide'].tolist()))
        # avoid positives (in training pos_pairs)
        negs = set()
        # target number = neg-test-factor * number of sampled positives (use seen_sample size)
        target_neg = int(len(seen_sample) * args.neg_test_factor) if len(seen_sample) > 0 else int(args.neg_test_factor * 1000)
        attempts = 0
        while len(negs) < target_neg and attempts < target_neg * 20:
            t = random.choice(all_tcr_list)
            e = random.choice(all_epi_list)
            k = make_pair_key(t, e)
            if k in pos_pairs:
                attempts += 1
                continue
            negs.add(k)
            attempts += 1

        print(f'Generated {len(negs)} negative test pairs (requested {target_neg})')
        neg_rows = [{'cdr3_truncated': t, 'peptide': e} for (t, e) in negs]
        neg_test_df = pd.DataFrame(neg_rows)
        # save negatives (as tsv with minimal columns)
        os.makedirs(os.path.dirname(args.out_test_neg) or '.', exist_ok=True)
        neg_test_df.to_csv(args.out_test_neg, sep='\t', index=False)
        print('Wrote negative test pairs to', args.out_test_neg)
        # predict on negatives
        neg_test_probs = predict_df(neg_test_df)

    def predict_df(df: pd.DataFrame):
        X = []
        for _, r in df.iterrows():
            t = r['cdr3_truncated']
            e = r['peptide']
            # find indices in all lists (if missing, skip)
            try:
                it = all_tcrs.index(t)
            except ValueError:
                continue
            try:
                ie = all_epitopes.index(e)
            except ValueError:
                continue
            xt = tcr_feats_all[it]
            xe = epi_feats_all[ie]
            X.append(np.concatenate([xt, xe]))
        if len(X) == 0:
            return np.array([])
        X = torch.from_numpy(np.vstack(X).astype(np.float32))
        with torch.no_grad():
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    print('Predicting seen sample...')
    seen_probs = predict_df(seen_sample)
    print('Predicting unseen sample...')
    unseen_probs = predict_df(unseen_sample)

    def summarize(arr):
        if arr.size == 0:
            return 'no samples'
        return f'mean={arr.mean():.4f}  median={np.median(arr):.4f}  std={arr.std():.4f}  n={len(arr)}'

    print('Seen sample prediction summary:', summarize(seen_probs))
    print('Unseen sample prediction summary:', summarize(unseen_probs))

    if args.retrain:
        print('Creating negative samples and retraining...')
        # build positive set (all positive pairs in train_small)
        pos_set = set(make_pair_key(r['cdr3_truncated'], r['peptide']) for _, r in train_df.iterrows() if int(r.get('Binding',0))==1)

        all_tcr_list = list(dict.fromkeys(combined['cdr3_truncated'].tolist()))
        all_epi_list = list(dict.fromkeys(combined['peptide'].tolist()))

        num_pos = len(train_df)
        num_neg = int(num_pos * args.neg_factor)
        negs = set()
        attempts = 0
        while len(negs) < num_neg and attempts < num_neg * 10:
            t = random.choice(all_tcr_list)
            e = random.choice(all_epi_list)
            k = make_pair_key(t, e)
            if k in pos_set:
                attempts += 1
                continue
            negs.add(k)
            attempts += 1

        print(f'Generated {len(negs)} negative samples (requested {num_neg})')
        neg_rows = [{'cdr3': t, 'peptide': e, 'Binding': 0} for (t,e) in negs]

        # write augmented train file
        orig = pd.read_csv(args.train_small, sep='\t')
        aug = orig.append(pd.DataFrame(neg_rows), ignore_index=True)
        os.makedirs(os.path.dirname(args.out_aug) or '.', exist_ok=True)
        aug.to_csv(args.out_aug, sep='\t', index=False)
        print('Wrote augmented train to', args.out_aug)

        # call training
        train_model_fn(args.out_aug, args.test_small, out_model=args.out_model, epochs=5)


if __name__ == '__main__':
    main()
