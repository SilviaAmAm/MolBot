import sys
sys.path.append('/Volumes/Transcend/repositories/NovaData/models/')
import sklearn_models
import pickle
import numpy as np

estimator = sklearn_models.Model_1(nb_epochs=1)

pickle.dump(estimator, open('model.pickle', 'wb'))

# making dataset
in_d = open("/Volumes/Transcend/repositories/NovaData/data/bioactivity_PPARg_filtered.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-3]
    molecule = molecule_raw[1:-1]
    if molecule == "CANONICAL_SMILES":
        pass
    else:
        molecules.append([molecule])

mol_array = np.asarray(molecules)
print(mol_array.shape)

test = np.random.randn(10, 2)

np.save("molecules.npy", test)