# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This example shows how to reload a previously saved model and carrying on fitting. It requires having run the script
example_training.py first.
"""

from molbot import smiles_generator, data_processing
import os
import numpy as np
import random

# Reading the data
data_dir = os.path.join("..", "data")
data_path = os.path.join(data_dir, "example_data_2.csv")
in_d = open(data_path, 'r')

# Parsing the data
molecules = []

for line in in_d:
    line = line.rstrip()
    molecules.append(line)

random.shuffle(molecules)
print("The total number of molecules is: %i \n" % (len(molecules)))

# One-hot encode the molecules by loading the already created data processing object
dp = data_processing.Molecules_processing()
dp.load("example-dp.pickle")
X = dp.onehot_encode(molecules)
# y is the same as X, but shifted by one character to the left and with the last character equal to the padding 'A' character
idx_A = dp.char_to_idx['A']
y = np.zeros(X.shape)
idx_A = dp.char_to_idx['A']
y[:, :-1, :] = X[:, 1:, :]
y[:, -1, idx_A] = 1

# Creating the model
estimator = smiles_generator.Smiles_generator(epochs=20, batch_size=100, tensorboard=False, validation=0.01)

# Reloading the parameters from the previously trained model
estimator.load("example-model.h5")

# Carrying on fitting the model that was previously saved
estimator.fit(X, y)