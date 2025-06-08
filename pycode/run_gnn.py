import chephy_model as cpm
import pandas as pd
import argparse, datetime, csv
from pathlib import Path
import os
import time
import numpy as np


def prepare_data(data, cdr3_header, epitope_header, score_header='vdjdb.score'):

    # remove cdr3 sequences that contain non-aa letters (this also removes nan)
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]

    data_filtered = data_filtered[data_filtered[cdr3_header].str.len() >= 8]

    return data_filtered


def get_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_stats(stats_file, args, execution_time):
    
    # todo add stats to every method

    """Save parameters and execution time to stats file."""
    args_dict = vars(args)
    args_dict["execution_time_seconds"] = execution_time  # Add execution time

    with open(stats_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(args_dict.keys())  # Write headers
        writer.writerow(args_dict.values())  # Write values


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

def read_csv_files_from_folder(folder_path="/dsi/scratch/home/dsi/solefroni/orforchen/downsamples_clonotype_21355"):
    """
    Reads all CSV files in a given folder and concatenates them into a single DataFrame.
    
    :param folder_path: Path to the folder containing CSV files.
    :return: Concatenated Pandas DataFrame.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        raise ValueError("No CSV files found in the specified folder.")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="files/vdjdb_score3.csv", help="Input csv for the model to train on, must be a csv file")
    parser.add_argument("-of", "--out_folder", default=get_time(), help="Name of output sub folder, default is current time")
    parser.add_argument("-r", "--right", default=4, help="Trim from the right side, default is 4")
    parser.add_argument("-l", "--left", default=4, help="Trim from the left side, default is 4")
    parser.add_argument("-ch", "--cdr_header",default="cdr3", help="cdr3 header in the input file. default id CDR3b")
    parser.add_argument("-eh", "--epitope_header",default="antigen.epitope", help="Epitope header in the input file. default id Epitope")


    args = parser.parse_args()


    # max_mutations = args.mutations
    out_folder = f"output/{args.out_folder}"
    out_name = "neighbors"
    params_file = "stats.csv"
    right = args.right
    left = args.left
    input_file = args.input
    max_dist=1
    label_header = args.epitope_header
    cdr3_header = args.cdr_header
    

    path = Path(f"{out_folder}")
    path.mkdir(parents=True, exist_ok=True)

    data = load_dataframe(input_file)

    # data = read_csv_files_from_folder()

    data = prepare_data(data, cdr3_header, label_header)    
    data = cpm.truncate_sequences(data, cdr3_header, right, left)

    sequences = set(data["cdr3_truncated"])
    chephy_matrix, hamming_matrix, sequences_list = cpm.find_che_phy_dist(sequences)
    
    data['index'] = data['cdr3_truncated'].apply(lambda x: sequences_list.index(x) if x in sequences_list else -1)
    sequence_indices = np.arange(len(sequences_list))




if __name__ == "__main__":
    main()