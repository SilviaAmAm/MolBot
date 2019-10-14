# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This example shows how to train an RNN on a set of smiles and how to predict new smiles with the trained model.
This is just an example to show how to construct the model, because the example data set only contains 458 samples, which
is not enough to train the RNN.
"""

from molbot import smiles_generator, data_processing
import os
import random
import numpy as np

# Reading the data
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../data/example_data_2.csv", 'r')

# Parsing the data
molecules = []

for line in in_d:
    line = line.rstrip()
    molecules.append(line)

print("The total number of molecules is: %i \n" % (len(molecules)))

# One-hot encode the molecules
dp = data_processing.Molecules_processing()
X = dp.onehot_encode(molecules)
# y is just the same as X just shifted by one with the A character at the end
idx_A = dp.char_to_idx['A']
y = np.zeros(X.shape)
y[:, :-1, :] = X[:, 1:, :]
y[:, -1, idx_A] = 1

# Creating the model
estimator = smiles_generator.Smiles_generator(epochs=20, batch_size=100, tensorboard=False, hidden_neurons_1=100,
                                              hidden_neurons_2=100, dropout_1=0.3, dropout_2=0.5, learning_rate=0.001,
                                              validation=0.01)

# Training the model on the one-hot encoded molecules
estimator.fit(X, y)

# Predicting 10 new molecules from the fitted model at a temperature of 0.75
X_pred_hot = dp.get_empty(10)
pred_hot = estimator.predict(X_pred_hot, temperature=0.75)
pred = dp.onehot_decode(pred_hot)

print(pred)

# Saving the estimator for later re-use
estimator.save("example-save.h5")
dp.save()

