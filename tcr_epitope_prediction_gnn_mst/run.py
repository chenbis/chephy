import chephy_model as cpm
import pandas as pd
import argparse, datetime, csv
from pathlib import Path
# import clustering
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from itertools import product


def prepare_data(data, cdr3_header, epitope_header, score_header='vdjdb.score'):

    # remove cdr3 sequences that contain non-aa letters (this also removes nan)
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]

    data_filtered = data_filtered[data_filtered[cdr3_header].str.len() >= 8]

    return data_filtered


def get_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV or TSV file and loads it into a Pandas DataFrame.
    Raises an error if the file is not a CSV or TSV.
    
    :param file_path: Path to the file.
    :return: Pandas DataFrame containing the file data.
    """
    # Check file extension
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_extension.lower() == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")    

