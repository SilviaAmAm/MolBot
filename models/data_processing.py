import numpy as np

def onehot_encode(molecules, debug=0):

    # Padding the molecules and finding all the unique characters
    max_size = len(max(molecules, key=len)) + 2                                 # Length of longest molecule
    padded_molecules = np.tile(np.array('A'), (len(molecules), max_size))       # Molecules padded with G, E and A

    for i in range(len(molecules)):
        all_char = ['G'] + list(molecules[i]) + ['E']
        padded_molecules[i][:len(all_char)] = all_char

    new_molecules = []
    for old_molecule in molecules:
        new_molecule = 'G' + old_molecule + 'E'
        if len(new_molecule) <= max_size:
            padding = int(max_size - len(new_molecule))
            for i in range(padding):
                new_molecule += 'A'
        new_molecules.append(new_molecule)

    # Indices of characters so that they can be translated into numbers
    all_possible_char = np.unique(padded_molecules)
    idx_to_char = {idx: char for idx, char in enumerate(all_possible_char)}
    char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}

    # Hot encode
    n_samples = int(len(new_molecules))
    n_feat = int(len(all_possible_char))

    hot_molecules = np.zeros((n_samples, max_size, n_feat), dtype=np.int16)

    for n in range(n_samples):
        sample = new_molecules[n]
        try:
            sample_idx = [char_to_idx[char] for char in sample]
            input_sequence = np.zeros((max_size, n_feat))
            for j in range(max_size):
                input_sequence[j][sample_idx[j]] = 1
            hot_molecules[n] = input_sequence
        except KeyError:
            print("One of the molecules contains a character that was not present in the first round of training.")
            exit()

    if debug:
        assert check_onehot_encoding(hot_molecules, molecules, idx_to_char)

    return hot_molecules

def check_onehot_encoding(hot_molecules, molecules, idx_to_char):

    for i in range(len(hot_molecules)):
        # One hot decodeing each molecule one at a time
        mol = ''
        for j in range(len(hot_molecules[i])):
            mol += idx_to_char[np.argmax(hot_molecules[i][j])]
        # Unpadding
        e_idx = mol.index("E")
        mol = mol[1:e_idx]
        if mol != molecules[i]:
            return False

    return True



