# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This script is an example of how to set up a model to learn molecular properties. The data set is a small example data
set containing 50 molecules and their activity.
"""

import numpy as np
import os

import sklearn.model_selection as modsel
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from molbot import data_processing
from molbot import properties_pred

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def convert_ic50_pic50(ic50):
    """
    Converts IC50 to pIC50 values.

    :param ic50: list of float
    :return: list of float
    """

    ic50 = np.asarray(ic50)

    return -1 * np.log(ic50 * 1e-9)

# Getting the data
data_dir = os.path.join("..", "data")
data_path = os.path.join(data_dir, "example_data_1.csv")
in_d = open(data_path, 'r')

# Read molecules and activities from CSV file
molecules = []
activities = []

for line in in_d:
    line = line.rstrip()
    line_split = line.split(",")
    molecule = line_split[0]
    activity = line_split[1]

    molecules.append(molecule)
    activities.append(float(activity))
activities = np.asarray(activities)

in_d.close()

# Processing the data
dp = data_processing.Molecules_processing()
X, y = dp.string_to_int(molecules), convert_ic50_pic50(activities)
X_train, X_test, y_train, y_test = modsel.train_test_split(X, y, test_size=0.1, shuffle=True)

# Hyperparameters
hidden_neurons_1 = 243
hidden_neurons_2 = 23
n_feat = X.shape[-1]
l1 = 0.00009
l2 = 0.000001
learning_rate = 0.0005
batch_size = 50
epochs = 119

# Creating the pipeline model
scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
estimator = properties_pred.Properties_predictor(hidden_neurons_1, hidden_neurons_2, l1, l2, learning_rate, batch_size, epochs)
pl = Pipeline(steps=[('scaling', scaler), ('nn', estimator)])

# Fitting and predicting
pl.fit(X_train, y_train)
y_pred = pl.predict(X_test)

# Plot correlation
plt.scatter(y_test, y_pred)
plt.show()
