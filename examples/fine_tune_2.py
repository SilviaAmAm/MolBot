from models import sklearn_models
from random import shuffle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

in_d = open("/Volumes/Transcend/PhD/NovaData_solutions/dataset/Cleaned_ChemBL_24.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.split(",")
    molecule_raw = line_split[-1]
    if "canonical_smiles" in molecule_raw:
        pass
    else:
        molecule_raw = molecule_raw.rstrip()
        molecule = molecule_raw[1:-1]
        molecules.append(molecule)

# shuffle(molecules)
molecules = molecules[:100]

estimator = sklearn_models.Model_2()

estimator.load("example-save")

estimator.fit_with_rl(temperature=0.75, n_train_episodes=40)

for b in range(10):
    p = estimator.predict(temperature=0.75)
    print(p[0])
