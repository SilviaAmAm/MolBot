"""
This example shows how to train Model 2 on a set of smiles, how to predict new smiles with the trained model and how to
score the predictions and assess their Tanimoto similarity to the original molecules in the training set.
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
estimator = sklearn_models.Model_2(epochs=3, batch_size=20)

# Training the model on 100 molecules from the data set
estimator.fit(molecules[:100])

# Predicting 10 new molecules from the fitted model at a temperature of 0.75
predictions = []
for a in range(10):
    p = estimator.predict(temperature=0.75)
    predictions.append(p[0])

# Saving the estimator for later re-use
estimator.save("example-save")

