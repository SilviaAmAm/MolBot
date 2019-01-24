# Copyright (c) NovaData Solutions LTD. All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This example shows how to reload a previously saved model and carrying on fitting. It requires having run the script
example_training.py first.
"""

from models import smiles_generator, data_processing
import os
import numpy as np
import random

# Reading the data
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../data/bioactivity_PPARg_filtered.csv", 'r')

# Parsing the data
molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)
random.shuffle(molecules)
print("The total number of molecules is: %i \n" % (len(molecules)))

# One-hot encode the molecules
dp = data_processing.Molecules_processing()
X = dp.onehot_encode(molecules)
# y is just the same as X just shifted by one
y = np.zeros(X.shape)
idx_A = dp.char_to_idx['A']
y[:, :-1, :] = X[:, 1:, :]
y[:, -1, idx_A] = 1

# Creating the model
estimator = smiles_generator.Smiles_generator(epochs=20, batch_size=100, tensorboard=False, hidden_neurons_1=100,
                                              hidden_neurons_2=100, dropout_1=0.3, dropout_2=0.5, learning_rate=0.001,
                                              validation=0.01)

# Reloading the model
estimator.load("example-save")

# Carrying on fitting the model that was previously saved
estimator.fit(X, y)