import sys
sys.path.append('/Volumes/Transcend/repositories/NovaData/models/')
import sklearn_models

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

estimator = sklearn_models.Model_1(nb_epochs=3)

estimator.fit(molecules[:4])

predictions = estimator.predict(molecules[:4])

f = open("/Volumes/Transcend/repositories/NovaData/pred_smiles/pred_smiles_1.txt", 'w')

for i in range(4):
    f.write('%100s  %100s' % (molecules[i], predictions[i]))
    f.write("\n")