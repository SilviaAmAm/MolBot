"""
This example shows how to train Model 1 on a set of smiles, how to predict new smiles with the trained model and how to
score the predictions and assess their Tanimoto similarity to the original molecules in the training set.
"""

from models import sklearn_models
import time
import os

# Reading the data set
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
estimator = sklearn_models.Model_1(epochs=3)

# Training the model on 100 molecules from the data set (the molecules are split into windows, so the number of samples
# looks much larger (5901 for training, 311 for validations in this case)
estimator.fit(molecules[:100])

# Predicting 40 new molecules
start = time.time()
predictions = estimator.predict(molecules[:40])
end = time.time()
print("The time taken to predict is %f" % (end-start))

# Saving the estimator for later re-use
estimator.save("example-save")

# Scoring some predictions based on the percenatge of valid smiles. This requires RDKit
score = estimator.score(molecules[:100])
print("The score obtained on the predictions is %s." % str(score))

# Scoring the Tanimoto similarity of the predictions to the training set
tanimoto = estimator.score_similarity(predictions, molecules[:100])
print("The Tanimoto similarity between the predictions and the training set is %s." % str(tanimoto))

