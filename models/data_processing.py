import numpy as np

def onehot_encode(molecules, debug=0):
    """
    This function takes the unpadded SMILES strings and returns the padded SMILES in one-hot encoding form.

    :param molecules: unpadded SMILES strings
    :type molecules: list of strings
    :param debug: if 1, checks whether the decoded smiles strings are equal to the original SMILES
    :type debug: bool
    :return: the one hot encoded molecules
    :rtype: np array of shape (n_samples, max_str_len, n_characters)
    """

    # Padding the molecules and finding all the unique characters
    max_size = len(max(molecules, key=len)) + 2                                 # Length of longest molecule
    padded_molecules = np.tile(np.array('A'), (len(molecules), max_size))       # Molecules padded with G, E and A

    for i in range(len(molecules)):
        all_char = ['G'] + list(molecules[i]) + ['E']
        padded_molecules[i][:len(all_char)] = all_char

    # Indices of characters so that they can be translated into numbers
    all_possible_char = np.unique(padded_molecules)
    idx_to_char = {idx: char for idx, char in enumerate(all_possible_char)}
    char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}

    # Hot encode
    n_samples = int(len(padded_molecules))
    n_feat = int(len(all_possible_char))

    hot_molecules = np.zeros((n_samples, max_size, n_feat), dtype=np.int16)

    for n in range(n_samples):
        sample = padded_molecules[n]
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
    """
    This function takes in some one hot encoded padded SMILES and the original SMILES strings. It decodes the one-hot
    encoded SMILES and compares them to the original molecules. If they are the same it returns TRUE, otherwise FALSE.

    :param hot_molecules: one-hot encoded SMILES
    :type hot_molecules: np array of shape (n_samples, max_str_len, n_characters)
    :param molecules: unpadded SMILES strings
    :type molecules: list of strings
    :param idx_to_char: dictionary mapping indices to characters
    :type idx_to_char: dictionary
    :return: bool
    """

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

def string_to_int(molecules):
    """
    This function takes in some unpadded SMILES strings and returns the padded SMILES where each character is replaced
    by an integer.

    :param molecules: unpadded SMILES strings
    :type molecules: list of strings
    :return: numeric molecules
    :rtype: np array of shape (n_samples, max_str_len)
    """

    # Padding the molecules and finding all the unique characters
    max_size = len(max(molecules, key=len)) + 2  # Length of longest molecule
    padded_molecules = np.tile(np.array('A'), (len(molecules), max_size))  # Molecules padded with G, E and A

    for i in range(len(molecules)):
        all_char = ['G'] + list(molecules[i]) + ['E']
        padded_molecules[i][:len(all_char)] = all_char

    # Indices of characters so that they can be translated into numbers
    all_possible_char = np.unique(padded_molecules)
    char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}

    # Turn characters to int
    n_samples = padded_molecules.shape[0]

    int_molecules = np.zeros((n_samples, max_size), dtype=np.int16)

    for n in range(n_samples):
        sample = padded_molecules[n]
        int_molecules[n] = [char_to_idx[char] for char in sample]

    return int_molecules