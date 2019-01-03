"""
This script is an example of how to train the properties predictor model.
"""

import numpy as np

import sklearn.model_selection as modsel

from models import data_processing
from models import properties_pred

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Getting the data
data_file = "/Volumes/Transcend/PhD/NovaData_solutions/dataset/TyrosineproteinkinaseJAK2.csv"
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
X_train, X_test, y_train, y_test = modsel.train_test_split(X, y, test_size=0.1, shuffle=True)
idx = list(range(100))

mean = np.mean(X)
X_train = X_train-mean

# Hyperparameters
hidden_neurons_1 = 100
hidden_neurons_2 = 100
n_feat = X.shape[-1]
dropout_1 = 1.0
dropout_2 = 1.0
learning_rate = 0.0075
batch_size = 10
epochs = 2000


estimator = properties_pred.Properties_predictor(hidden_neurons_1, hidden_neurons_2, dropout_1, dropout_2, learning_rate, batch_size, epochs)
estimator.fit(X_train[idx], y_train[idx])
y_pred = estimator.predict(X_train[idx])

print(y_pred)


# pickle.dump(estimator, open('model.pickle', 'wb'))
#
# del estimator
# estimator_new = pickle.load(open("model.pickle",'rb'))

# Plot correlation
plt.scatter(y_train[idx], y_pred)
plt.show()