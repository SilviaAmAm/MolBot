"""
This module contains the model that is used to predict the activity from SMILES strings
"""

import keras
from keras import Sequential, optimizers, regularizers
from keras.layers import Dense
from keras.callbacks import TensorBoard

import sklearn.model_selection as modsel
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import __version__
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tempfile
import warnings

class Properties_predictor(BaseEstimator):

    def __init__(self, hidden_neurons_1=100, hidden_neurons_2=100, l1=0.0, l2=0.0, learning_rate=0.001, batch_size=20,
                 epochs=4):

        self.hidden_neurons_1 = hidden_neurons_1
        self.hidden_neurons_2 = hidden_neurons_2
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self._n_feat = 0

    def fit(self, X, y):

        # Check the inputs
        X, y = check_X_y(X, y, accept_sparse=False)

        self._n_feat = X.shape[-1]
        model = self._build_model()

        # Set up tensorboard
        tensorboard = TensorBoard(log_dir='./tb', write_graph=True, write_images=False)
        callbacks_list = [tensorboard]

        # If there are enough samples, use some as validation data
        if X.shape[0] >= 20:
            X_train, X_val, y_train, y_val = modsel.train_test_split(X, y, test_size=0.05)


            model.fit(X_train, y_train, batch_size=self.batch_size, verbose=1, epochs=self.epochs,
                        callbacks=callbacks_list, validation_data=(X_val, y_val))
        else:
            model.fit(X, y, batch_size=self.batch_size, verbose=1, epochs=self.epochs, callbacks=callbacks_list)

        self._model = model
        self.is_fitted_ = True
        return self

    def predict(self, X):

        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        y_pred = self._model.predict(X)

        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        else:
            return y_pred

    def score(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=False)
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        y_pred = (self._model.predict(X)).ravel()
        mae = (-1.0) * mean_absolute_error(y, y_pred, sample_weight=None)

        return mae

    def _build_model(self):

        model = Sequential()
        model.add(Dense(self.hidden_neurons_1, input_dim=self._n_feat, activation='tanh',
                        kernel_regularizer=regularizers.l1_l2(self.l1, self.l2)))
        model.add(Dense(self.hidden_neurons_2, activation='tanh',
                        kernel_regularizer=regularizers.l1_l2(self.l1, self.l2)))
        model.add(Dense(1, activation='linear'))
        optimiser = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss="mean_squared_error", optimizer=optimiser)

        return model

    # Needed because the models in keras dont have a __getstate__ function, which is needed for pickling
    def __getstate__(self):

        try:
            state = super(BaseEstimator, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()
            try:
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as tmp:
                    self._model.save(tmp.name, overwrite=True)
                    saved_model = tmp.read()
                state["_model"] = saved_model
            except AttributeError:
                pass

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
            try:
                with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as tmp:
                    tmp.write(state['_model'])
                    tmp.flush()
                    self.__dict__["_model"] = keras.models.load_model(tmp.name)
            except KeyError:
                pass
