"""
This example shows how to reload a previously saved model and carrying on fitting. It requires having run the script
ex_model_1.py first.
"""

from models import sklearn_models
import os

# Reading the data
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../data/bioactivity_PPARg_filtered.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)

# Creating the model
estimator = sklearn_models.Model_1(epochs=4)

# Reloading the model
estimator.load("example-save")

# Carrying on fitting the model that was previously saved
estimator.fit(molecules[:100])