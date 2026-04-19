import pandas as pd
import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Read CSV file with 'amino.acid' as the index column
df = pd.read_csv('./files/atchley.csv', index_col='amino.acid')

amino_acids = df.index.tolist()
features = df.values

# Calculate Euclidean distance matrix
distance_matrix = np.zeros((len(amino_acids), len(amino_acids)))

for i in range(len(amino_acids)):
    for j in range(len(amino_acids)):
        amino_acid1 = features[i, :]
        amino_acid2 = features[j, :]
        distance_matrix[i, j] = euclidean_distance(amino_acid1, amino_acid2)

# Use amino_acids as both index and columns in the DataFrame
distance_df = pd.DataFrame(distance_matrix, index=amino_acids, columns=amino_acids)

# Rename the first column to 'amino_acid'
# distance_df.rename_axis('amino.acid', axis=1, inplace=True)
distance_df.columns.names = ['amino.acids']

distance_df.to_csv('distance_matrix.csv', index=True)
