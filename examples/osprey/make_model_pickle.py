"""
This script shows how to prepare a pickled model to be used with Osprey.
"""

from models import sklearn_models
import pickle
import os

# Reading the data
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../../data/bioactivity_PPARg_filtered.csv", 'r')
molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)

# Creating the model and saving the smiles in the model
estimator = sklearn_models.Model_1(epochs=1, batch_size=1000 , smiles=molecules)

# Creating the pickle
pickle.dump(estimator, open('model.pickle', 'wb'))

# Creating a list of indices that Osprey will use to refer to the samples
with open('idx.csv', 'w') as f:
    for i in range(len(molecules)):
        f.write('%s\n' % i)
