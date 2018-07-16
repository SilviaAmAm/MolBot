import numpy as np
import joblib

# Importing the data
in_d = open("bioactivity_PPARg_filtered.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)

# Preparing for one hot encoding

all_possible_char = ['G', 'E', 'A']
max_size = 2

new_molecules = []

for molecule in molecules:
    all_char = list(molecule)

    if len(all_char)+2 > max_size:
        max_size=len(all_char)+2

    unique_char = list(set(all_char))
    for item in unique_char:
        if not item in all_possible_char:
            all_possible_char.append(item)

    molecule = 'G' + molecule + 'E'
    new_molecules.append(molecule)

idx_to_char = {idx:char for idx, char in enumerate(all_possible_char)}
char_to_idx = {char:idx for idx, char in enumerate(all_possible_char)}

window_length = 10

X = []
y = []

for new_mol in new_molecules:
    for i in range(len(new_mol)-window_length):
        X.append(new_mol[i:i+window_length])
        y.append(new_mol[i+window_length])

# One hot encoding
n_samples = len(X)
n_features = len(all_possible_char)

X_hot = np.zeros((n_samples, window_length, n_features))
y_hot = np.zeros((n_samples, n_features))

for n in range(n_samples):
    sample_x = X[n]
    sample_x_idx = [char_to_idx[char] for char in sample_x]
    input_sequence = np.zeros((window_length, n_features))
    for j in range(window_length):
        input_sequence[j][sample_x_idx[j]] = 1.0
    X_hot[n]=input_sequence

    output_sequence = np.zeros((n_features,))
    sample_y = y[n]
    sample_y_idx = char_to_idx[sample_y]
    output_sequence[sample_y_idx] = 1.0
    y_hot[n] = output_sequence


data = {"X":X_hot, "y":y_hot, "idx_to_char":idx_to_char, "char_to_idx":char_to_idx}
joblib.dump(data, "dataset_1.bz")