# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This class can be used to turn SMILES representations into one-hot encoded arrays and viceversa.
The initial character is "G" and the end character is "E", while the padding character is "A".
"""

import numpy as np
import pickle

class Molecules_processing():

    def __init__(self):

        # Dictionaries that map characters and their index
        self.idx_to_char = {}
        self.char_to_idx = {}

        # Size of largest molecule
        self.max_size = 0

    def onehot_encode(self, molecules):
        """
        This function takes the unpadded SMILES strings and returns the padded SMILES in one-hot encoding form.

        :param molecules: unpadded SMILES strings
        :type molecules: list of strings
        :return: the one hot encoded molecules
        :rtype: np array of shape (n_samples, max_str_len, n_characters)
        """

        # Turn every character into a number
        int_molecules = self.string_to_int(molecules)

        # One-hot encode
        n_samples = int(int_molecules.shape[0])
        n_feat = int(len(self.idx_to_char))

        hot_molecules = np.zeros((n_samples, self.max_size, n_feat), dtype=np.int16)

        for n in range(n_samples):
            try:
                input_sequence = np.zeros((self.max_size, n_feat))
                for j in range(self.max_size):
                    input_sequence[j][int_molecules[n][j]] = 1
                hot_molecules[n] = input_sequence
            except KeyError:
                print("One of the molecules contains a character that was not present in the first round of training.")
                exit()

        return hot_molecules

    def onehot_decode(self, hot_molecules):
        """
        This function takes in some one-hot encoded padded SMILES and returns a list of unpadded SMILES strings.

        :param hot_molecules: one-hot encoded SMILES
        :type hot_molecules: np array of shape (n_samples, max_str_len, n_characters)
        :return: unpadded SMILES strings
        :rtype: list of strings
        """

        molecules = []

        for i in range(len(hot_molecules)):
            # One hot decodeing each molecule one at a time
            mol = ''
            for j in range(len(hot_molecules[i])):
                mol += self.idx_to_char[np.argmax(hot_molecules[i][j])]
            # Unpadding
            try:
                e_idx = mol.index("E")
                molecules.append(mol[1:e_idx])
            except ValueError:
                molecules.append(mol)

        return molecules

    def string_to_int(self, molecules):
        """
        This function takes in some unpadded SMILES strings and returns the padded SMILES where each character is replaced
        by an integer.

        :param molecules: unpadded SMILES strings
        :type molecules: list of strings
        :return: numeric molecules
        :rtype: np array of shape (n_samples, max_str_len)
        """

        # Padding the molecules and finding all the unique characters
        if self.max_size == 0:
            self.max_size = len(max(molecules, key=len)) + 2  # Length of longest molecule

        padded_molecules = np.tile(np.array('A'), (len(molecules), self.max_size))  # Molecules padded with G, E and A

        for i in range(len(molecules)):
            all_char = ['G'] + list(molecules[i]) + ['E']
            padded_molecules[i][:len(all_char)] = all_char

        # Indices of characters so that they can be translated into numbers
        all_possible_char = np.unique(padded_molecules)
        if len(self.char_to_idx) == 0:
            self.char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}
            self.idx_to_char = {idx: char for idx, char in enumerate(all_possible_char)}

        # Turn characters to int
        n_samples = padded_molecules.shape[0]

        int_molecules = np.zeros((n_samples, self.max_size), dtype=np.int16)

        for n in range(n_samples):
            sample = padded_molecules[n]
            int_molecules[n] = [self.char_to_idx[char] for char in sample]

        return int_molecules

    def get_empty(self, n):
        """
        This function outputs a one hot encoded G character, to be used by the SMILES generator as the initial character
        to generate a new  SMILES string.

        :param n: number of empty one-hot encoded smiles to output
        :type n: int
        :return: One hot-encoded G character
        :rtype: numpy array of shape (n, 1, n_char)
        """

        empty_smiles = [""]*n

        hot_empty = self.onehot_encode(empty_smiles)
        trim_hot_empty = np.reshape(hot_empty[:, :1, :], (hot_empty[:, :1, :].shape[0], 1, hot_empty[:, :1, :].shape[-1]))

        return trim_hot_empty

    def save(self, filename='data_proc.pickle'):
        """
        This function saves the data processing object so it can be used at a later stage.

        :param filename: name of the file in which to save the object.
        :type filename: string
        :return: None
        """

        pickle.dump([self.char_to_idx, self.idx_to_char, self.max_size], open(filename, "wb"))

    def load(self, filename='data_proc.pickle'):
        """
        This function reloads previously created dictionaries with the indices of the characters in the SMILES.

        :param filename: name of the file in which the object is saved.
        :type filename: string
        :return: None
        """

        loaded_dictionaries = pickle.load(open(filename, "rb"))
        self.char_to_idx = loaded_dictionaries[0]
        self.idx_to_char = loaded_dictionaries[1]
        self.max_size = loaded_dictionaries[2]