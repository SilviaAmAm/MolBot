
"""
This module contains the model that is used to predict the activity from SMILES strings
"""

from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras import optimizers
from keras.callbacks import TensorBoard
from keras import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as modsel
import numpy as np
import data_processing as dp

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
    activity = line_split[51]
    molecule = molecule_raw[1:-1]
    if molecule == "SMI (Canonical)":
        pass
    else:
        molecules.append(molecule)
        activities.append(activity)
activities = np.asarray(activities)

# Getting the data and splitting it
X, y = dp.onehot_encode(molecules), activities
X_train, X_test, y_train, y_test = modsel.train_test_split(X, y, test_size=0.1, shuffle=True)
X_train, X_val, y_train, y_val = modsel.train_test_split(X_train, y_train, test_size=0.1, shuffle=True)

# Hyperparameters
hidden_neurons_1 = 10
hidden_neurons_2 = 10
n_feat = X.shape[-1]
dropout_1 = 0.0
dropout_2 = 0.0
learning_rate = 0.001
batch_size = 100
epochs = 5

# Model
model = Sequential()
# This will output (max_size, n_hidden_1)
model.add(LSTM(units=hidden_neurons_1, input_shape=(None, n_feat), return_sequences=True, dropout=dropout_1))
# This will output (n_hidden_2,)
model.add(LSTM(units=hidden_neurons_2, input_shape=(None, hidden_neurons_1), return_sequences=False, dropout=dropout_2))
# This will output (1)
model.add(Dense(1))

optimiser = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="mean_squared_error", optimizer=optimiser)


# Fitting
tensorboard = TensorBoard(log_dir='./tb', write_graph=True, write_images=False)
callbacks_list = [tensorboard]
model.fit(X_train, y_train, batch_size=batch_size, verbose=1, epochs=epochs,
               # callbacks=callbacks_list,
                validation_data=(X_val, y_val))

# Predictions
y_pred = model.predict(X_train)

# Plot correlation
sns.set()
plt.scatter(y_pred, y_train)
plt.show()