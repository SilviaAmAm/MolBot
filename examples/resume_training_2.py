import sys
sys.path.append('/Volumes/Transcend/repositories/NovaData/models/')
import sklearn_models
import numpy as np
import time

in_d = open("/Volumes/Transcend/repositories/NovaData/data/bioactivity_PPARg_filtered.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append(molecule)

estimator = sklearn_models.Model_2(nb_epochs=1, smiles=molecules)


estimator.fit(list(range(100)))

estimator.save()

estimator.load("./saved_models/model_2.h5")
estimator.continue_fit(list(range(100)))