from sklearn.base import BaseEstimator
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from keras.layers import Lambda
from keras.models import load_model
import os
import numpy as np

class Model_1(BaseEstimator):
    """Estimator Model 1"""

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size=500, nb_epochs=4, window_length=10):

        self.tensorboard = tensorboard
        self.hidden_neurons_1 = hidden_neurons_1
        self.hidden_neurons_2 = hidden_neurons_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.window_length = window_length

        self.model = None
        self.loaded_model = None
        self.idx_to_char = None
        self.char_to_idx = None

    def fit(self, X, y=None):
        """
        This function takes in a list of smiles strings and hot encodes them before fitting the model to them.
        :param X: list of smiles strings
        :param y: None
        :return: None
        """

        X_hot, y_hot = self._hot_encode(X)

        self.n_samples = X_hot.shape[0]
        self.max_size = X_hot.shape[1]
        self.n_feat = X_hot.shape[2]

        self._generate_model()

        if self.tensorboard == True:
            tensorboard = TensorBoard(log_dir='./tb/model_1',
                                      write_graph=True, write_images=False)
            callbacks_list = [tensorboard]
            self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs, callbacks=callbacks_list)
        else:
            self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs)

    def save(self, dir='./saved_models/'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = dir + "model_1.h5"
        self.model.save(filename)
        print("Saved model in directory: " + dir + "\n")

    def load(self, filename):

        self.loaded_model = load_model(filename)

    def predict(self, X):
        """
        This takes a complete smiles string and takes the first window and then predicts the following characters until
        the 'E' is produced. if the 'E' is not produced then it cuts it when it reaches 100 characters.

        :param X: Set of full smiles strings
        :type X: list of strings
        :return: predicted smiles strings
        :rtype: list of strings
        """

        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            raise Exception("The model has not been fit and no saved model has been loaded.\n")

        elif isinstance(self.model, type(None)):
            predictions = self._predict(X, self.loaded_model)

        else:
            predictions = self._predict(X, self.model)

        return predictions

    def _generate_model(self):
        """
        This function generates the model.
        :return: None
        """
        model = Sequential()
        # This will output (max_size, n_hidden_1)
        model.add(LSTM(units=self.hidden_neurons_1, input_shape=(None, self.n_feat), return_sequences=True, dropout=self.dropout_1))
        # This will output (n_hidden_2,)
        model.add(
            LSTM(units=self.hidden_neurons_2, input_shape=(None, self.hidden_neurons_1), return_sequences=False, dropout=self.dropout_2))
        # This will output (n_feat,)
        model.add(Dense(self.n_feat))
        # Modifying the softmax with the `Temperature' parameter
        model.add(Lambda(lambda x: x / 1))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

        self.model = model

    def _hot_encode(self, X):

        if isinstance(self.idx_to_char, type(None)) and isinstance(self.char_to_idx, type(None)):
            all_possible_char = ['G', 'E', 'A']
            max_size = 2

            new_molecules = []
            for molecule in X:
                all_char = list(molecule)

                if len(all_char) + 2 > max_size:
                    max_size = len(all_char) + 2

                unique_char = list(set(all_char))
                unique_char.sort()

                for item in unique_char:
                    if not item in all_possible_char:
                        all_possible_char.append(item)

                molecule = 'G' + molecule + 'E'
                new_molecules.append(molecule)

            self.idx_to_char = {idx: char for idx, char in enumerate(all_possible_char)}
            self.char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}
            n_possible_char = len(self.idx_to_char)
        else:
            new_molecules = []
            for molecule in X:
                molecule = 'G' + molecule + 'E'
                new_molecules.append(molecule)

            n_possible_char = len(self.idx_to_char)


        # Splitting X into window lengths and y into the characters after each window
        window_X = []
        window_y = []

        for mol in new_molecules:
            for i in range(len(mol) - self.window_length):
                window_X.append(mol[i:i+self.window_length])
                window_y.append(mol[i+self.window_length])

        # One hot encoding
        n_samples = len(window_X)
        n_features = n_possible_char

        X_hot = np.zeros((n_samples, self.window_length, n_features))
        y_hot = np.zeros((n_samples, n_features))

        for n in range(n_samples):
            sample_x = window_X[n]
            sample_x_idx = [self.char_to_idx[char] for char in sample_x]
            input_sequence = np.zeros((self.window_length, n_features))
            for j in range(self.window_length):
                input_sequence[j][sample_x_idx[j]] = 1.0
            X_hot[n] = input_sequence

            output_sequence = np.zeros((n_features,))
            sample_y = window_y[n]
            sample_y_idx = self.char_to_idx[sample_y]
            output_sequence[sample_y_idx] = 1.0
            y_hot[n] = output_sequence

        return X_hot, y_hot

    def _hot_decode(self, X):

        cold_X = []

        n_samples = X.shape[0]
        max_length = X.shape[1]

        for i in range(n_samples):
            smile = ''
            for j in range(max_length):
                out_idx = np.argmax(X[i, j, :])
                smile += self.idx_to_char[out_idx]

            cold_X.append(smile)

        return cold_X

    def _predict(self, X, model):

        n_samples = len(X)

        all_predictions = []

        for i in range(0, n_samples):
            X_hot, _ = self._hot_encode([X[i]])
            X_pred = X_hot[i, :, :]  # shape (window_size, n_feat)
            X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))  # shape (1, window_size, n_feat)

            y_pred = self._hot_decode(X_pred)[0]

            X_pred_temp = X_pred

            while (y_pred[-1] != 'E'):
                out = self.model.predict(X_pred_temp)  # shape (1, n_feat)
                y_pred += self._hot_decode(np.reshape(out, (1, out.shape[0], out.shape[1])))[0]
                X_pred_temp[:, :-1, :] = X_pred[:, 1:, :]

                y_pred_hot = np.zeros((1, X_pred.shape[-1]))
                y_pred_hot[:, np.argmax(out)] = 1

                X_pred_temp[:, -1, :] = y_pred_hot

                if len(y_pred) == 100:
                    break

            all_predictions.append(y_pred)

        return all_predictions

class Model_2(BaseEstimator):
    """Estimator Model 2"""

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size=500, nb_epochs=4):
        self.tensorboard = tensorboard
        self.hidden_neurons_1 = hidden_neurons_1
        self.hidden_neurons_2 = hidden_neurons_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.model = None
        self.loaded_model = None
        self.idx_to_char = None
        self.char_to_idx = None

    def fit(self, X, y=None):
        """
        This function takes in a list of smiles strings and hot encodes them before fitting the model to them.
        :param X: list of smiles strings
        :param y: None
        :return: None
        """

        X_hot, y_hot = self._hot_encode(X)

        self.n_samples = X_hot.shape[0]
        self.max_size = X_hot.shape[1]
        self.n_feat = X_hot.shape[2]

        self._generate_model()

        if self.tensorboard == True:
            tensorboard = TensorBoard(log_dir='./tb/model_1',
                                      write_graph=True, write_images=False)
            callbacks_list = [tensorboard]
            self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs, callbacks=callbacks_list)
        else:
            self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs)

    def predict(self, X=None):
        """
        This function predicts new strings starting from a G or from a fragment. It carries on predicting until the 
        character 'E' is output or until the string has reached the maximum length present in the training set.
        
        :param X: one or more smile strings
        :type: list of strings
        :return: predictions
        :rtype: list of strings
        """
        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            raise Exception("The model has not been fit and no saved model has been loaded.\n")

        elif isinstance(self.model, type(None)):
            predictions = self._predict(X, self.loaded_model)

        else:
            predictions = self._predict(X, self.model)

        return predictions

    def save(self, dir='./saved_models/'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = dir + "model_2.h5"
        self.model.save(filename)
        print("Saved model in directory: " + dir + "\n")

    def load(self, filename):
        """
        This function loads a model that has been previously saved.
        :param filename: string 
        :return: None
        """

        self.loaded_model = load_model(filename)

    def _hot_encode(self, X):
        """
        This function takes in a list of smiles strings and hot encodes them. If it is called from the fit function it
        hot encodes also the 'y'. If it is called from the predict function, it just hot encodes the fragments given.
        :param X: smiles strings
        :type X: list of strings
        :return: the smiles strings hot encoded.
        :rtype: two numpy arrays of shape (n_samples, max_size, n_feat) or a list of numpy array of shape (fragment_length, n_feat)
        """
        
        if isinstance(self.idx_to_char, type(None)) and isinstance(self.char_to_idx, type(None)):
            all_possible_char = ['G', 'E', 'A']
            max_size = 2

            for molecule in X:
                all_char = list(molecule)

                if len(all_char) + 2 > max_size:
                    max_size = len(all_char) + 2

                unique_char = list(set(all_char))
                for item in unique_char:
                    if not item in all_possible_char:
                        all_possible_char.append(item)

            all_possible_char.sort()

            # Padding
            new_molecules = []
            for molecule in X:
                molecule = 'G' + molecule + 'E'
                if len(molecule) < max_size:
                    padding = int(max_size - len(molecule))
                    for i in range(padding):
                        molecule += 'A'
                new_molecules.append(molecule)

            self.idx_to_char = {idx: char for idx, char in enumerate(all_possible_char)}
            self.char_to_idx = {char: idx for idx, char in enumerate(all_possible_char)}
            n_feat = len(self.idx_to_char)

            n_samples = int(len(new_molecules))

            X_hot = np.zeros((n_samples, max_size, n_feat))
            y_hot = np.zeros((n_samples, max_size, n_feat))

            for n in range(n_samples):
                sample = new_molecules[n]
                sample_idx = [self.char_to_idx[char] for char in sample]
                input_sequence = np.zeros((max_size, n_feat))
                for j in range(max_size):
                    input_sequence[j][sample_idx[j]] = 1.0
                X_hot[n] = input_sequence

                output_sequence = np.zeros((max_size, n_feat))
                for j in range(max_size - 1):
                    output_sequence[j][sample_idx[j + 1]] = 1.0
                y_hot[n] = output_sequence
            
        else:
            new_molecules = []
            max_size = 1
            for molecule in X:
                if len(molecule)+1 > max_size:
                    max_size = len(molecule)+1
                molecule = "G" + molecule
                new_molecules.append(molecule)

            n_feat = len(self.idx_to_char)
            n_samples = int(len(new_molecules))

            X_hot = []
            y_hot = []

            for n in range(n_samples):
                sample = new_molecules[n]
                sample_idx = [self.char_to_idx[char] for char in sample]
                input_sequence = np.zeros((len(sample), n_feat))
                for j in range(len(sample)):
                    input_sequence[j][sample_idx[j]] = 1.0
                X_hot.append(input_sequence)

        return X_hot, y_hot

    def _generate_model(self):
        """
        This function generates the `model 2'.
        :return: None
        """

        model = Sequential()
        # This will output (max_size, n_hidden_1)
        model.add(LSTM(units=self.hidden_neurons_1, input_shape=(None, self.n_feat), return_sequences=True, dropout=self.dropout_1))
        # This will output (max_size, n_hidden_2)
        model.add(
            LSTM(units=self.hidden_neurons_2, input_shape=(None, self.hidden_neurons_1), return_sequences=True, dropout=self.dropout_2))
        # This will output (max_size, n_feat)
        model.add(TimeDistributed(Dense(self.n_feat), input_shape=(None, self.hidden_neurons_2)))
        # Modifying softmax with temperature
        model.add(Lambda(lambda x: x / 1))
        model.add(Activation('softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

        self.model = model
        
    def _predict(self, X, model):
        
        if isinstance(X, type(None)):
            X_pred = np.zeros((1, self.max_size, self.n_feat))
            y_pred = 'G'
            X_pred[0, 0, self.char_to_idx['G']] = 1

            for i in range(self.max_size - 1):
                out = model.predict(X_pred[:, :i + 1, :])[0][-1]
                idx_out = np.argmax(out)
                X_pred[0, i + 1, idx_out] = 1
                if self.idx_to_char[idx_out] == 'E':
                    break
                else:
                    y_pred += self.idx_to_char[idx_out]

            return y_pred
                    
        else:
            n_samples = len(X)

            all_predictions = []

            X_hot, _ = self._hot_encode(X)



            for n in range(0, n_samples):
                X_pred = X_hot[n]  # shape (fragment_length, n_feat)
                y_pred = X[n]

                while y_pred[-1] != 'E':
                    X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))  # shape (1, fragment_length, n_feat)

                    out = model.predict(X_pred)[0][-1]
                    idx_out = np.argmax(out)

                    y_pred += self.idx_to_char[idx_out]
                    X_pred = self._hot_encode([y_pred[1:]])[0][0]

                    if len(y_pred) == 100:
                        break

                all_predictions.append(y_pred)

            return all_predictions

