import numpy as np
import joblib

# Importing the data
in_d = open("bioactivity_PPARg_filtered.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw.strip('"')
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)

# Preparing for one hot encoding

all_possible_char = ['G', 'E', 'A']
max_size = 2

for molecule in molecules:
    all_char = list(molecule)

    if len(all_char)+2 > max_size:
        max_size=len(all_char)+2

    unique_char = list(set(all_char))
    for item in unique_char:
        if not item in all_possible_char:
            all_possible_char.append(item)

# Padding
padded_molecules = []
for molecule in molecules:
    molecule = 'G' + molecule
    molecule = molecule + 'E'
    if len(molecule) < max_size:
        padding = int(max_size - len(molecule))
        for i in range(padding):
            molecule += 'A'
    padded_molecules.append(molecule)

idx_to_char = {idx:char for idx, char in enumerate(all_possible_char)}
char_to_idx = {char:idx for idx, char in enumerate(all_possible_char)}

n_feat = len(all_possible_char)
n_samples = int(len(molecules))

X = np.zeros((n_samples, max_size, n_feat))
y = np.zeros((n_samples, max_size, n_feat))

for n in range(n_samples):
    sample = padded_molecules[n]
    sample_idx = [char_to_idx[char] for char in sample]
    input_sequence = np.zeros((max_size, n_feat))
    for j in range(max_size):
        input_sequence[j][sample_idx[j]] = 1.0
    X[n]=input_sequence

    output_sequence = np.zeros((max_size, n_feat))
    for j in range(max_size-1):
        output_sequence[j][sample_idx[j+1]] = 1.0
    y[n] = output_sequence

# np.savez("dataset_1.npz", X, y, idx_to_char, char_to_idx)
data = {"X":X, "y":y, "idx_to_char":idx_to_char, "char_to_idx":char_to_idx}
joblib.dump(data, "dataset_2.bz")

