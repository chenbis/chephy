def truncate_single_sequence(sequence, n_term_size, c_term_size):
    """Truncate a sequence to include only n_term_size from N-terminus and c_term_size from C-terminus"""
    if len(sequence) <= n_term_size + c_term_size:
        return sequence
    return sequence[:n_term_size] + sequence[-c_term_size:]

def calculate_chemical_physical_distance(seq1, seq2):
    """Calculate chemical-physical distance between two sequences"""
    import numpy as np
    import pandas as pd
    
    # This is a simplified implementation
    if len(seq1) != len(seq2):
        return float('inf')  # Cannot compare sequences of different lengths
        
    # Get Atchley factors for each amino acid
    atchley_df = pd.read_csv("/home/dsi/chenbis/repos/sol_lab/files/atchley.csv")
    atchley_dict = atchley_df.set_index('amino.acid').to_dict('index')
    
    distance = 0
    for aa1, aa2 in zip(seq1, seq2):
        if aa1 in atchley_dict and aa2 in atchley_dict:
            # Calculate Euclidean distance between Atchley vectors
            vec1 = np.array(list(atchley_dict[aa1].values()))
            vec2 = np.array(list(atchley_dict[aa2].values()))
            distance += np.linalg.norm(vec1 - vec2)
    
    return distance / len(seq1)  # Normalize by sequence length