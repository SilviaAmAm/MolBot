import sys
sys.path.append('/Volumes/Transcend/repositories/NovaData/models/')
import sklearn_models
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

estimator = sklearn_models.Model_1(nb_epochs=1)

estimator.fit(molecules)

start = time.time()
predictions = estimator.predict(molecules[:100])
end = time.time()
print("The time taken to predict is %f" % (end-start))

score = estimator.score(molecules[:100])
print(score)

tanimoto = estimator.score_similarity(predictions, molecules)



f = open("/Volumes/Transcend/repositories/NovaData/pred_smiles/pred_smiles_1.txt", 'w')

for i in range(4):
    f.write('%100s  %100s' % (molecules[i], predictions[i]))
    f.write("\n")