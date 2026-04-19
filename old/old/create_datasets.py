import pandas as pd
import os
from chephy_model import truncate_sequences

def prepare_data(data, cdr3_header, epitope_header, score_header='vdjdb.score'):
    """Filter amino acid sequences and validate format/length."""
    amino_acid_pattern = r'^[ARNDCEQGHILKMFPSTWYV]+$'

    data_filtered = data[data[cdr3_header].str.match(amino_acid_pattern, case=False, na=False)]
    data_filtered = data_filtered[data_filtered[cdr3_header].str.len() >= 8]
    data_filtered = data_filtered[data_filtered[epitope_header].str.match(amino_acid_pattern, case=False, na=False)]
    data_filtered[epitope_header] = data_filtered[epitope_header].str.upper()
    data_filtered[cdr3_header] = data_filtered[cdr3_header].str.upper()

    return data_filtered


def standardize(df, cdr3_col, epitope_col, score_col=None, source_name=""):
    """Rename columns and add source + default score if needed."""
    df = df[[cdr3_col, epitope_col]] if score_col is None else df[[cdr3_col, epitope_col, score_col]]
    df.columns = ['cdr3', 'epitope'] + (['score'] if score_col else [])
    
    if 'score' not in df.columns:
        df['score'] = -1
    df['source'] = source_name
    return df


def main():
    # Load files
    vdjdb = pd.read_csv('/home/dsi/chenbis/repos/sol_lab/files/vdjdb/vdjdb_full.tsv', sep='\t')
    mcpas = pd.read_csv('/home/dsi/chenbis/repos/sol_lab/files/McPAS-TCR.csv')
    iedb = pd.read_csv('/home/dsi/chenbis/repos/sol_lab/files/IEDB.csv')
    trait = pd.read_csv('/home/dsi/chenbis/repos/sol_lab/files/vdjdb_trait_onlyB.csv')

    # VDJdb
    vdjdb_clean = prepare_data(vdjdb, 'cdr3.beta', 'antigen.epitope', 'vdjdb.score')
    vdjdb_std = standardize(vdjdb_clean, 'cdr3.beta', 'antigen.epitope', 'vdjdb.score', 'VDJDB')

    # McPAS
    mcpas_clean = prepare_data(mcpas, 'CDR3.beta.aa', 'Epitope.peptide')
    mcpas_std = standardize(mcpas_clean, 'CDR3.beta.aa', 'Epitope.peptide', None, 'McPAS')

    # IEDB
    iedb_clean = prepare_data(iedb, 'Chain 2 CDR3 Curated', 'Description')
    iedb_std = standardize(iedb_clean, 'Chain 2 CDR3 Curated', 'Description', None, 'IEDB')

    # TRAIT
    trait_filtered = trait[trait['Source'] == 'TRAIT']
    trait_clean = prepare_data(trait_filtered, 'CDR3b', 'Epitope')
    trait_std = standardize(trait_clean, 'CDR3b', 'Epitope', None, 'TRAIT')

    # Combine
    all_data = pd.concat([vdjdb_std, mcpas_std, iedb_std, trait_std], ignore_index=True)

    # all_data = truncate_sequences(all_data, 'cdr3')


    # Save to CSV
    all_data.to_csv('combined_data.csv', index=False)
    print(f"Combined dataset saved to 'combined_data.csv' with {len(all_data)} rows.")


if __name__ == "__main__":
    main()
