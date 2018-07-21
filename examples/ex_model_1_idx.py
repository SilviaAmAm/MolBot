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

estimator = sklearn_models.Model_1(nb_epochs=1, smiles=molecules, batch_size=5000)

idx_train = range(int(len(molecules)))

estimator.fit(idx_train)



start = time.time()
predictions = estimator.predict(range(5))
end = time.time()
print("The time taken to predict is %f" % (end-start))

score = estimator.score(range(5))
print(score)

tanimoto = estimator.score_similarity(predictions, molecules[:5])
