# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

from molbot import data_processing
import os

def _get_data():
    # Reading the data set
    current_dir = os.path.dirname(os.path.realpath(__file__))
    in_d = open(current_dir + "/../data/example_data_2.csv", 'r')

    molecules = []

    for line in in_d:
        line = line.rstrip()
        molecules.append(line)
    return molecules

def test_onehot_encode():

    molecules = _get_data()

    data_handler = data_processing.Molecules_processing()
    hot_mols = data_handler.onehot_encode(molecules)
    mols = data_handler.onehot_decode(hot_mols)

    for i in range(len(molecules)):
        assert molecules[i] == mols[i]

if __name__ == "__main__":
    test_onehot_encode()
