"""
This shows how to create the pickled model of the properties predictor so that Osprey can be used to optimise its
hyper-parameters.
"""

import pickle
import numpy as np
from models import properties_pred, data_processing
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from random import shuffle
import os

def convert_ic50_pic50(ic50):
    ic50 = np.asarray(ic50)

    return -1 * np.log(ic50 * 1e-9)


# Getting the data
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../data/TyrosineproteinkinaseJAK2.csv", 'r')

# Read molecules and activities from CSV file
molecules = []
activities = []

for line in in_d:
    line = line.rstrip()
    line_split = line.split(",")
    molecule_raw = line_split[-1]
    activity = line_split[3]
    molecule = molecule_raw[1:-1]
    if molecule == "SMI (Canonical)":
        pass
    else:
        molecules.append(molecule)
        activities.append(float(activity))

shuffle(molecules)

# Processing the data
dp = data_processing.Molecules_processing()
X, y = dp.string_to_int(molecules), convert_ic50_pic50(activities)

# Creating the model
scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
estimator = properties_pred.Properties_predictor()
pl = Pipeline(steps=[('scaling', scaler), ('nn', estimator)])

# Dump the model
pickle.dump(estimator, open('model.pickle', 'wb'))

# Dump the data set
pickle.dump({"X":X, "y":y}, open('activity.pickle', 'wb'))