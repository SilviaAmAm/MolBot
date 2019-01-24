# Copyright (c) NovaData Solutions LTD. All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This module contains the RNN that learns from SMILES strings and then generates new SMILES.
"""
import numpy as np

from keras import optimizers
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from keras.layers import Lambda
from keras.models import load_model

from models import utils

class Smiles_generator():

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size="auto", epochs=4, learning_rate=0.001, validation=0.05):
        """
        This function initialises the parent class common to both Model 1 and 2.

        :param tensorboard: whether to log progress to tensorboard or not
        :type tensorboard: bool
        :param hidden_neurons_1: number of hidden units in the first LSTM
        :type hidden_neurons_1: int
        :param hidden_neurons_2: number of hidden units in the second LSTM
        :type hidden_neurons_2: int
        :param dropout_1: dropout rate in the first LSTM
        :type dropout_1: float
        :param dropout_2:  dropout rate in the 2nd LSTM
        :type dropout_2: float
        :param batch_size: Size of the data set batches to use during training
        :type batch_size: int
        :param epochs: number of iterations of training
        :type epochs: int
        :param smiles: list of smiles strings from which to learn
        :type smiles: list of strings
        :param learning_rate: size of the step taken by the optimiser
        :type learning_rate: float > 0
        :param validation: percentage of samples to use for validation during training.
        :type validation: float >= 0 and < 1
        """

        self.tensorboard = utils.set_tensorboard(tensorboard)
        self.hidden_neurons_1 = utils.set_hidden_neurons(hidden_neurons_1)
        self.hidden_neurons_2 = utils.set_hidden_neurons(hidden_neurons_2)
        self.dropout_1 = utils.set_dropout(dropout_1)
        self.dropout_2 = utils.set_dropout(dropout_2)
        self.batch_size = utils.set_provisional_batch_size(batch_size)
        self.epochs = utils.set_epochs(epochs)
        self.learning_rate = utils.set_learning_rate(learning_rate)
        self.validation = utils.set_validation(validation)

        self.model = None
        self.loaded_model = None

    def fit(self, X, y):
        """
        This function fits the parameters of the RNN to the data provided.

        :param X: Input one-hot-encoded padded smiles strings
        :type X: np.array of shape (n_samples, max_len, n_char)
        :param y: Output one-hot-encoded padded smiles strings
        :type y: np.array of shape (n_samples, max_len, n_char)

        :return: the estimator object
        """

        # Check the inputs
        X, y = utils.check_X_y(X, y)

        # Construct the model
        model = self._build_model(X.shape[-1])

        # Adjust batch_size based on number of samples provided
        batch_size = utils.set_batch_size(self.batch_size, X.shape[0])

        # Set up tensorboard
        if self.tensorboard:
            tensorboard = TensorBoard(log_dir='./tb', write_graph=True, write_images=False)
            callbacks_list = [tensorboard]
        else:
            callbacks_list = []

        # If the model has never been trained, create a new one, otherwise restart training from a previously trained model
        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            model = self._build_model(X.shape[-1])
        elif not isinstance(self.model, type(None)):
            model = self.model
        elif not isinstance(self.loaded_model, type(None)):
            model = self.loaded_model

        # If there are enough samples, use some as validation data
        if int(self.validation*X.shape[0]) > 0:
            train_idx = int((1-self.validation)*X.shape[0])

            model.fit(X[:train_idx], y[:train_idx], batch_size=batch_size, verbose=1, epochs=self.epochs,
                        callbacks=callbacks_list, validation_data=(X[train_idx:], y[train_idx:]))
        else:
            model.fit(X, y, batch_size=batch_size, verbose=1, epochs=self.epochs, callbacks=callbacks_list)

        # Update the right variable with the newly trained model
        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            self.model = model
        elif not isinstance(self.model, type(None)):
            self.model = model
        elif not isinstance(self.loaded_model, type(None)):
            self.loaded_model = model

        self._model = model
        self.is_fitted_ = True

        return self

    def predict(self, X, temperature=1.0, max_length=200):
        """
        This function starts from a hot encoded SMILES and predicts the remaining part of the molecule. X needs to
        at least contain a 'G' character, it cannot be empty.

        :param X: hot-encoded SMILES
        :type X: np.array with shape (n_samples, length_smiles, n_char)
        :param temperature: Temperature factor for the modified softmax
        :type temperature: float
        :param max_length: maximum length of a smile string to generate
        :type max_length: int
        :return: hot-encoded SMILES
        :rtype: np.array of shape (n_samples, length_smiles, n_char)
        """

        utils.check_temperature(temperature)
        utils.check_maxlength(max_length)

        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            raise Exception("The model has not been fit and no saved model has been loaded.\n")
        elif not isinstance(self.model, type(None)):
            model = self._modify_model_for_predictions(self.model, temperature)
        else:
            model = self._modify_model_for_predictions(self.loaded_model, temperature)

        X_pred = self._pred(X, model, max_length)

        return X_pred

    def save(self, filename='model.h5'):
        """
        This function enables to save the trained model so that then training or predictions can be done at a later stage.

        :param filename: Name of the file in which to save the model.
        :return: None
        """

        if not isinstance(self.model, type(None)):
            self.model.save(filename, overwrite=False)
        elif not isinstance(self.loaded_model, type(None)):
            self.loaded_model.save(filename, overwrite=False)
        else:
            raise utils.InputError("No model to be saved.")

    def load(self, filename='model.h5'):
        """
        This function loads a model that has been previously saved.

        :param filename: Name of the file in which the model has been previously saved.
        :return: None
        """

        self.loaded_model = load_model(filename)

    def _build_model(self, n_feat):
        """
        This function generates the RNN.

        :return: Keras Sequential object
        """
        model = Sequential()
        # This will output (max_size, n_hidden_1)
        model.add(LSTM(units=self.hidden_neurons_1, input_shape=(None, n_feat), return_sequences=True,
                       dropout=self.dropout_1))
        # This will output (max_size, n_hidden_2)
        model.add(
            LSTM(units=self.hidden_neurons_2, input_shape=(None, self.hidden_neurons_1), return_sequences=True,
                 dropout=self.dropout_2))
        # This will output (max_size, n_feat)
        model.add(TimeDistributed(Dense(n_feat), input_shape=(None, self.hidden_neurons_2)))
        # Modifying softmax with temperature
        model.add(Lambda(lambda x: x / 1))
        model.add(Activation('softmax'))
        optimiser = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                    amsgrad=False)
        model.compile(loss="categorical_crossentropy", optimizer=optimiser)

        return model

    def _modify_model_for_predictions(self, model, temperature):
        """
        This function modifies the model for predict time by adding a temperature factor to the softmax activation
        function.

        :param model: the model to modify
        :type model: keras model
        :param temperature: temperature that modifies the softmax
        :type temperature: float > 0 and <= 1
        :return: The modified model
        """

        model.pop()
        model.pop()
        model.add(Lambda(lambda x: x / temperature))
        model.add(Activation('softmax'))

        return model

    def _pred(self, X, model, max_length):
        """
        This function predicts one-hot encoded smiles strings starting from a fragment.

        :param X: One-hot encoded fragment of smile string
        :param model: the keras model to use for prediction
        :param max_length: maximum length of predicted molecules
        :return: predicted one-hot encoded smiles strings
        """

        n_feat = X.shape[-1]
        n_samples = X.shape[0]

        # Predictions
        X_pred = np.zeros((n_samples, max_length, n_feat))
        X_pred[:, :X.shape[1], :] = X

        for i in range(1, max_length):
            prob_distribution = model.predict(X_pred[:, :i, :])

            # Slow step
            for j in range(n_samples):
                idx_out = np.random.choice(np.arange(n_feat), p=prob_distribution[j, -1])
                X_pred[j, i, idx_out] = 1

        return X_pred

