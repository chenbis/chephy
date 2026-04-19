
import os
import glob
import json
import numpy as np
import pandas as pd

from knn_mst_graph import compute_global_ref_from_patients, build_patient_graph

# ------------ USER SETTINGS ------------
# Directory containing *.tsv files (one per patient). The patient ID is the filename (without extension).
DATA_DIR = "/dsi/sbm/chen2/downsampled_data/500"    # <-- change this
OUT_DIR = "/dsi/sbm/chen2/graph_data/500"              # where to save outputs
ATCHLEY_CSV = "/home/dsi/chenbis/repos/sol_lab/files/atchley.csv" # <-- change if your Atchley CSV is elsewhere

# k for k-NN and normalization percentile
K = 30
PAD_TO_LEN = 8
PERCENTILE = 95.0
TRAIN_FRACTION_FOR_GLOBAL_REF = 0.8   # fraction of patients used to compute global ref
# --------------------------------------

def load_sequences_from_tsv(tsv_path: str) -> list:
    """Reads 'cdr3_truncated' column from a TSV and returns a list of sequences (strings)."""
    df = pd.read_csv(tsv_path, sep="\t")
    if "cdr3_truncated" not in df.columns:
        raise ValueError(f"'cdr3_truncated' column not found in {tsv_path}")
    seqs = (
        df["cdr3_truncated"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Filter out empty strings
    seqs = [s for s in seqs if len(s) > 0]
    return seqs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover patient TSVs
    tsv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {DATA_DIR}")

    # Map patient_id -> list of sequences
    patients = {}
    for tsv in tsv_files:
        pid = os.path.splitext(os.path.basename(tsv))[0]
        seqs = load_sequences_from_tsv(tsv)
        if len(seqs) < 2:
            print(f"[WARN] {pid} has <2 sequences; skipping.")
            continue
        patients[pid] = seqs

    if not patients:
        raise RuntimeError("No patients with at least 2 sequences.")

    patient_ids = list(patients.keys())
    print(f"Found {len(patient_ids)} patients.")

    # Split patients into train (for global ref) and the rest
    rng = np.random.default_rng(0)
    n_train = max(1, int(len(patient_ids) * TRAIN_FRACTION_FOR_GLOBAL_REF))
    train_ids = set(rng.choice(patient_ids, size=n_train, replace=False).tolist())

    # Compute global reference percentile on TRAIN patients only
    ref_p95 = compute_global_ref_from_patients(
        [patients[pid] for pid in patient_ids if pid in train_ids],
        atchley_csv_path=ATCHLEY_CSV,
        k=K,
        pad_to_len=PAD_TO_LEN,
        percentile=PERCENTILE,
    )
    print(f"Computed global REF_P{int(PERCENTILE)} = {ref_p95:.6f} from {len(train_ids)} patients.")

    # Build graphs for all patients (train/val/test splits inside each graph are node-level splits)
    manifest = {}
    for pid in patient_ids:
        print(f"[{pid}] Building graph...")
        saved = build_patient_graph(
            sequences=patients[pid],
            atchley_csv_path=ATCHLEY_CSV,
            k=K,
            pad_to_len=PAD_TO_LEN,
            normalize_percentile=PERCENTILE,
            global_ref_value=ref_p95,         # use global normalization
            train_test_split=0.8,
            out_dir=OUT_DIR,
            patient_id=pid,
            ensure_connected=True,
        )
        manifest[pid] = saved
        print(f"[{pid}] Done. Saved: {saved}")

    # Save a manifest of all output files
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest to:", manifest_path)


if __name__ == "__main__":
    main()
