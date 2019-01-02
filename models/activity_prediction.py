
"""
This module contains the model that is used to predict the activity from SMILES strings
"""
import keras
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import TensorBoard
import sklearn.model_selection as modsel
import numpy as np
import data_processing as dp
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
import pickle
import tempfile

class Properties_predictor(BaseEstimator):

    def __init__(self, hidden_neurons_1=100, hidden_neurons_2=100, dropout_1=0.0, dropout_2=0.0, learning_rate=0.1, batch_size=20, epochs=4):

        self.hidden_neurons_1 = hidden_neurons_1
        self.hidden_neurons_2 = hidden_neurons_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.n_feat = 0

    def fit(self, X, y):

        # Split the data
        X_train, X_val, y_train, y_val = modsel.train_test_split(X, y, test_size=0.05)

        self.n_feat = X.shape[-1]
        self.model = self._build_model()

        tensorboard = TensorBoard(log_dir='./tb', write_graph=True, write_images=False)
        callbacks_list = [tensorboard]
        self.model.fit(X_train, y_train, batch_size=self.batch_size, verbose=1, epochs=self.epochs,
                    # callbacks=callbacks_list,
                  validation_data=(X_val, y_val))

        self.is_fitted_ = True
        return self

    def predict(self, X):

        check_is_fitted(self, 'is_fitted_')

        y_pred = self.model.predict(X)

        return y_pred

    def _build_model(self):

        model = Sequential()
        # This will output (max_size, n_hidden_1)
        model.add(LSTM(units=self.hidden_neurons_1, input_shape=(None, self.n_feat), return_sequences=True, dropout=self.dropout_1))
        # This will output (n_hidden_2,)
        model.add(LSTM(units=self.hidden_neurons_2, input_shape=(None, self.hidden_neurons_1), return_sequences=False,
                       dropout=self.dropout_2))
        # This will output (1)
        model.add(Dense(1))

        optimiser = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=optimiser)

        return model

    # Needed because the models in keras dont have a __getstate__ function, which is needed for pickling
    def __getstate__(self):

        try:
            state = super(BaseEstimator, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as tmp:
                self.model.save(tmp.name, overwrite=True)
                saved_model = tmp.read()
            state["model"] = saved_model

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    # Needed because the models in keras dont have a __setstate__ function, which is needed for pickling
    def __setstate__(self, state):

        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super(BaseEstimator, self).__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as tmp:
            tmp.write(state['model'])
            tmp.flush()
            self.__dict__["model"] = keras.models.load_model(tmp.name)

if __name__ == "__main__":

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

    # Hyperparameters
    hidden_neurons_1 = 10
    hidden_neurons_2 = 10
    n_feat = X.shape[-1]
    dropout_1 = 0.0
    dropout_2 = 0.0
    learning_rate = 0.001
    batch_size = 100
    epochs = 1

    X = np.ones((X_train.shape[0], X_train.shape[-1]))

    estimator = Properties_predictor(hidden_neurons_1, hidden_neurons_2, dropout_1, dropout_2, learning_rate, batch_size, epochs)
    estimator.fit(X, y_train)
    y_pred = estimator.predict(X)

    pickle.dump(estimator, open('model.pickle', 'wb'))

    del estimator
    estimator_new = pickle.load(open("model.pickle",'rb'))

    y_pred_new = estimator_new.predict(X_test)

    # Plot correlation
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    plt.scatter(y_pred, y_pred_new)
    plt.show()