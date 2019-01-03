import pickle
import numpy as np
from models import properties_pred, data_processing
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


# Getting the data
data_file = "/home/sa16246/data_sets/Novadata/TyrosineproteinkinaseJAK2.csv"
# data_file = "/Volumes/Transcend/PhD/NovaData_solutions/dataset/TyrosineproteinkinaseJAK2.csv"
in_d = open(data_file, "r")

# Read molecules and activities from CSV file
molecules = []
activities = []

for line in in_d:
    line = line.rstrip()
    line_split = line.split(",")
    molecule_raw = line_split[-1]
    activity = line_split[53]
    molecule = molecule_raw[1:-1]
    if molecule == "SMI (Canonical)":
        pass
    else:
        molecules.append(molecule)
        activities.append(float(activity))
activities = np.asarray(activities)

# Processing the data
X, y = data_processing.string_to_int(molecules), activities

scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
estimator = properties_pred.Properties_predictor()

pl = Pipeline(steps=[('scaling', scaler), ('nn', estimator)])

pickle.dump(estimator, open('model.pickle', 'wb'))

pickle.dump({"X":X, "y":y}, open('tpsa.pickle', 'wb'))