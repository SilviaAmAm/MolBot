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
import re
from sklearn.base import BaseEstimator
import utils

class _Model(BaseEstimator):
    """
    This is the parent class to the to different models.
    """

    def __init__(self, tensorboard, hidden_neurons_1, hidden_neurons_2, dropout_1, dropout_2,
                 batch_size, nb_epochs, smiles):
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
        :param nb_epochs: number of iterations of training
        :type nb_epochs: int
        :param smiles: list of smiles strings from which to learn
        :type smiles: list of strings
        """

        self.tensorboard = self._set_tensorboard(tensorboard)
        self.hidden_neurons_1 = self._set_hidden_neurons(hidden_neurons_1)
        self.hidden_neurons_2 = self._set_hidden_neurons(hidden_neurons_2)
        self.dropout_1 = self._set_dropout(dropout_1)
        self.dropout_2 = self._set_dropout(dropout_2)
        self.batch_size = self._set_provisional_batch_size(batch_size)
        self.nb_epochs = self._set_epochs(nb_epochs)

        self.model = None
        self.loaded_model = None
        self.idx_to_char = None
        self.char_to_idx = None
        if not isinstance(smiles, type(None)):
            self.smiles = self._check_smiles(smiles)
        else:
            self.smiles = None

    def _set_tensorboard(self, tb):

        if utils.is_bool(tb):
            return tb
        else:
            raise utils.InputError("Parameter Tensorboard should be either true or false. Got %s" % (str(tb)))

    def _set_hidden_neurons(self, h):
        if utils.is_positive_integer(h):
            return h
        else:
            raise utils.InputError("The number of hidden neurons should be a positive non zero integer. Got %s." % (str(h)))

    def _set_dropout(self, drop):
        if drop > 0 and drop < 1:
            return drop
        else:
            raise utils.InputError(
                "The dropout rate should be between 0 and 1. Got %s." % (str(drop)))

    def _set_provisional_batch_size(self, batch_size):
        if batch_size != "auto":
            if not utils.is_positive_integer(batch_size):
                raise utils.InputError("Expected 'batch_size' to be a positive integer. Got %s" % str(batch_size))
            elif batch_size == 1:
                raise utils.InputError("batch_size must be larger than 1.")
            return int(batch_size)
        else:
            return batch_size

    def _set_batch_size(self):

        if self.batch_size == 'auto':
            batch_size = min(100, self.n_samples)
        else:
            if self.batch_size > self.n_samples:
                print("Warning: batch_size larger than sample size. It is going to be clipped")
                return self.n_samples
            else:
                batch_size = self.batch_size

        better_batch_size = utils.ceil(self.n_samples, utils.ceil(self.n_samples, batch_size))

        return better_batch_size

    def _set_epochs(self, epochs):
        if utils.is_positive_integer(epochs):
            return epochs
        else:
            raise utils.InputError("The number of epochs should be a positive integer. Got %s." % (str(epochs)))

    def _check_smiles(self, smiles):
        if utils.is_array_like(smiles):
            for item in smiles:
                if not isinstance(item, str):
                    raise utils.InputError("Smiles should be a list of string.")

            return smiles
        else:
            raise utils.InputError("Smiles should be a list of string.")

    def fit(self, X):
        """
        This function fits the parameters of a GRNN to the data provided.

        :param X: list of smiles or list of indices of the smiles to use
        :type X: list of strings or list of ints
        :return: None
        """

        X_hot, y_hot = self._initialise_data_fit(X)

        self.n_samples = X_hot.shape[0]
        self.max_size = X_hot.shape[1]
        self.n_feat = X_hot.shape[2]

        batch_size = self._set_batch_size()

        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            self._generate_model()

            if self.tensorboard == True:
                tensorboard = TensorBoard(log_dir='./tb',
                                          write_graph=True, write_images=False)
                callbacks_list = [tensorboard]
                self.model.fit(X_hot, y_hot, batch_size=batch_size, verbose=1, nb_epoch=self.nb_epochs,
                               callbacks=callbacks_list)
            else:
                self.model.fit(X_hot, y_hot, batch_size=batch_size, verbose=1, nb_epoch=self.nb_epochs)

        elif not isinstance(self.model, type(None)):
            if self.tensorboard == True:
                tensorboard = TensorBoard(log_dir='./tb',
                                          write_graph=True, write_images=False)
                callbacks_list = [tensorboard]
                self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs,
                               callbacks=callbacks_list)
            else:
                self.model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs)

        elif not isinstance(self.loaded_model, type(None)):
            if self.tensorboard == True:
                tensorboard = TensorBoard(log_dir='./tb',
                                          write_graph=True, write_images=False)
                callbacks_list = [tensorboard]
                self.loaded_model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs,
                                      callbacks=callbacks_list)
            else:
                self.loaded_model.fit(X_hot, y_hot, batch_size=self.batch_size, verbose=1, nb_epoch=self.nb_epochs)

        else:
            raise utils.InputError("No model has been fit already or has been loaded.")

    def predict(self, X=None, frag_length=5):
        """
        This function predicts some smiles from either nothing or from fragments of smiles strings.

        :param X: list of smiles strings or nothing
        :type X: list of str
        :param frag_length: length of smiles string fragment to use.
        :type frag_length: int
        :return: list of smiles string
        :rtype: list of str
        """

        X_strings, X_hot = self._initialise_data_predict(X, frag_length)

        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            raise Exception("The model has not been fit and no saved model has been loaded.\n")

        elif isinstance(self.model, type(None)):
            predictions = self._predict(X_strings, X_hot, self.loaded_model)

        else:
            predictions = self._predict(X_strings, X_hot, self.model)

        return predictions

    def score(self, X=None):
        """
        This function takes in smiles strings and scores the model on the predictions. The score is the percentage of
         valid smiles strings that have been predicted.

        :param X: smiles strings
        :type X: list of strings
        :return: score
        :rtype: float
        """

        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ModuleNotFoundError("RDKit is required for scoring.")

        predictions = self.predict(X)

        n_valid_smiles = 0

        for smile_string in predictions:
            try:
                mol = Chem.MolFromSmiles(smile_string)
                if not isinstance(mol, type(None)):
                    n_valid_smiles += 1
            except Exception:
                pass

        score = n_valid_smiles/len(predictions)

        return score

    def score_similarity(self, X_1, X_2):
        """
        This function calculates the average Tanimoto similarity between each molecule in X_1 and those in X_2. It
        returns all the average Tanimoto coefficients and the percentage of duplicates.

        :param X_1: list of smiles strings to compare
        :param X_2: list of smiles strings acting as reference
        :return: Tanimoto coefficients and the percentage of duplicates
        :rtype: list of floats, float
        """

        try:
            from rdkit import Chem
            from rdkit.Chem.Fingerprints import FingerprintMols
            from rdkit import DataStructs
        except ModuleNotFoundError:
            raise ModuleNotFoundError("RDKit is required for scoring the similarity.")

        # Making the smiles strings in rdkit molecules
        mol_1, invalid_1 = self._make_rdkit_mol(self._check_smiles(X_1))
        mol_2, invalid_2 = self._make_rdkit_mol(self._check_smiles(X_2))

        # Turning the molecules in Daylight fingerprints
        fps_1 = [FingerprintMols.FingerprintMol(x) for x in mol_1]
        fps_2 = [FingerprintMols.FingerprintMol(x) for x in mol_2]

        # Obtaining similarity measure
        tanimoto_coeff = []
        n_duplicates = 0

        for i in range(len(fps_1)):
            sum_tanimoto = 0
            for j in range(len(fps_2)):
                coeff = DataStructs.FingerprintSimilarity(fps_1[i], fps_2[j])
                sum_tanimoto += coeff
                if coeff == 1:
                    n_duplicates += 1

            avg_tanimoto = sum_tanimoto/len(fps_2)
            tanimoto_coeff.append(avg_tanimoto)

        if len(fps_1) != 0:
            percent_duplicates = n_duplicates/len(fps_1)
        else:
            percent_duplicates = 1

        return tanimoto_coeff, percent_duplicates

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

    def _make_rdkit_mol(self, X):
        """
        This function takes a list of smiles strings and returns a list of rdkit objects for the valid smiles strings.

        :param X: list of smiles
        :return: list of rdkit objects
        """

        X = self._check_smiles(X)

        mol = []
        invalid = 0

        for smile in X:
            try:
                molecule = Chem.MolFromSmiles(smile)
                if not isinstance(molecule, type(None)):
                    mol.append(molecule)
                else:
                    invalid += 1
            except Exception:
                pass

        return mol, invalid

class Model_1(_Model):
    """
    Estimator Model 1

    This estimator learns from segments of smiles strings all of the same length and the next character along the sequence.
    When presented with a new smiles fragment it predicts the most likely next character."""

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size="auto", nb_epochs=4, window_length=10, smiles=None):
        """
        This function uses the initialiser of the parent class and initialises the window length.

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
        :param nb_epochs: number of iterations of training
        :type nb_epochs: int
        :param window_length: size of the smiles fragments from which to learn
        :type window_length: int
        :param smiles: list of smiles strings from which to learn
        :type smiles: list of strings
        """

        super(Model_1, self).__init__(tensorboard, hidden_neurons_1, hidden_neurons_2, dropout_1, dropout_2,
                 batch_size, nb_epochs, smiles)

        # TODO make check for window length
        self.window_length = window_length

        if not isinstance(self.smiles, type(None)):
            self.X_hot, self.y_hot = self._hot_encode(smiles)

    def _initialise_data_fit(self, X):
        """
        This function checks whether the smiles strings are stored in the class. Then it checks that X is a list of
        indices specifying which data samples to use. Then, it returns the appropriate fragments of smiles strings hot
        encoded.

        :param X: either list of smiles strings or indices.
        :type X: either list of strings or list of ints
        :return: the fragments of smiles strings hot encoded and the following character
        :rtype: numpy arrays of shape (n_samples, n_window_length, n_unique characters) and (n_samples, n_unique_characters)
        """

        if not isinstance(self.smiles, type(None)):
            if not utils.is_positive_integer_or_zero_array(X):
                raise utils.InputError("The indices need to be positive or zero integers.")

            window_idx = self._idx_to_window_idx(X)      # Converting from the index of the sample to the index of the windows
            X_hot = np.asarray([self.X_hot[i] for i in window_idx])
            y_hot = np.asarray([self.y_hot[i] for i in window_idx])
        else:
            X_strings = self._check_smiles(X)
            X_hot, y_hot = self._hot_encode(X_strings)

        return X_hot, y_hot

    def _initialise_data_predict(self, X, frag_length):
        """
        X can either be a list of smiles strings or the indices to the samples to be used for prediction. In the latter
        case, the data needs to have been stored inside the class.

        This function takes the smiles strings and splits them into fragments (of length specified by the window length)
        to be used for predictions. The first fragment of each smile is used for prediction.

        :param X: list of smiles or list of indices
        :type X: list of strings or list of ints
        :param frag_length: parameter not needed for model 1
        :return: list of the first fragment of each smile string specified and its one-hot encoded version.
        :rtype: list of strings, numpy array of shape (n_samples, n_window_length, n_unique_char)
        """

        if isinstance(X, type(None)):
            raise utils.InputError("Model_1 can only predict from fragments of length %i. No smiles given." % (self.window_length))

        if not isinstance(self.smiles, type(None)):
            if not utils.is_positive_integer_or_zero_array(X):
                raise utils.InputError("Indices should be passed to the predict function as smiles strings are already stored in the class.")
            X_strings = [self.smiles[int(i)] for i in X]
            window_idx = self._idx_to_window_idx(X)
            X_hot = np.asarray([self.X_hot[i] for i in window_idx])
        else:
            self._check_smiles(X)
            X_hot, _ = self._hot_encode(X)
            # Adding G and E at the ends since the hot version has it:
            X_strings = []
            for item in X:
                X_strings.append("G" + item + "E")

        return X_strings, X_hot

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
        """
        This function takes in a list of smiles strings and returns the smiles strings hot encoded split into windows.

        :param X: smiles strings
        :type X: list of strings
        :return: hot encoded smiles string in windows
        :rtype: numpy array of shape (n_samples*n_windows, window_length, n_features)
        """

        if isinstance(self.idx_to_char, type(None)) and isinstance(self.char_to_idx, type(None)):
            all_possible_char = ['G', 'E', 'A']
            max_size = 2

            new_molecules = []
            for molecule in X:
                all_char = list(molecule)

                if len(all_char) + 2 > max_size:
                    max_size = len(all_char) + 2

                unique_char = list(set(all_char))

                for item in unique_char:
                    if not item in all_possible_char:
                        all_possible_char.append(item)

                all_possible_char.sort()

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

        if not isinstance(self.smiles, type(None)):
            self.smiles = new_molecules

        # Splitting X into window lengths and y into the characters after each window
        window_X = []
        window_y = []

        self.idx_to_window = []
        counter = 0
        for mol in new_molecules:
            self.idx_to_window.append(counter)
            for i in range(len(mol) - self.window_length):
                window_X.append(mol[i:i+self.window_length])
                window_y.append(mol[i+self.window_length])
                counter += 1

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

    def _predict(self, X_strings, X_hot, model):
        """
        This function takes in a list of smiles strings. Then, it takes the first window from each smiles and predicts
        a full smiles string starting from that window.

        :param X: smiles strings
        :type: list of smiles strings
        :param model: the keras model
        :return: predictions
        :rtype: list of strings
        """

        n_samples = len(X_strings)

        all_predictions = []

        n_windows = 0
        idx_first_window = 0

        for i in range(0, n_samples):
            idx_first_window += n_windows

            X_pred = X_hot[idx_first_window, :, :]  # predicting from the first window
            X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))  # shape (1, window_size, n_feat)

            y_pred = X_strings[i][:self.window_length]

            X_pred_temp = X_pred

            while (y_pred[-1] != 'E'):
                out = model.predict(X_pred_temp)  # shape (1, n_feat)
                y_pred += self._hot_decode(np.reshape(out, (1, out.shape[0], out.shape[1])))[0]
                X_pred_temp[:, :-1, :] = X_pred[:, 1:, :]

                y_pred_hot = np.zeros((1, X_pred.shape[-1]))
                y_pred_hot[:, np.argmax(out)] = 1

                X_pred_temp[:, -1, :] = y_pred_hot

                if len(y_pred) == 100:
                    break

            if y_pred[0] == 'G':
                y_pred = y_pred[1:]
            if y_pred[-1] == 'E':
                y_pred = y_pred[:-1]

            all_predictions.append(y_pred)

            # This is the index of the next 'first window' in X_hot
            n_windows = len(X_strings[i]) - self.window_length

        return all_predictions

    def _idx_to_window_idx(self, idx):
        """
        This function takes the indices of the smiles strings and returns the indices of the corresponding windows.

        :param idx: list of ints
        :return:  list of ints
        """

        window_idx = []

        for i, idx_start in enumerate(idx):
            if idx_start < len(self.idx_to_window)-1:
                w_idx_start = self.idx_to_window[int(idx[i])]
                w_idx_end = self.idx_to_window[int(idx[i])+1]       # idx where the next sample starts
                for j in range(w_idx_start, w_idx_end):
                    window_idx.append(j)
            else:
                w_idx_start = self.idx_to_window[int(idx[i])]
                w_idx_end = self.X_hot.shape[0]
                for j in range(w_idx_start, w_idx_end):
                    window_idx.append(j)

        return window_idx

class Model_2(_Model):
    """
    Estimator Model 2

    This estimator learns from full sequences and can predict new smiles strings starting from any length fragment.

    """

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size=500, nb_epochs=4, smiles=None):
        """
            This function uses the initialiser of the parent class and initialises the window length.

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
            :param nb_epochs: number of iterations of training
            :type nb_epochs: int
            :param smiles: list of smiles strings from which to learn
            :type smiles: list of strings
            """

        super(Model_2, self).__init__(tensorboard, hidden_neurons_1, hidden_neurons_2, dropout_1, dropout_2,
                                     batch_size, nb_epochs, smiles)

        if not isinstance(self.smiles, type(None)):
            self.X_hot, self.y_hot = self._hot_encode(smiles)

    def _initialise_data_fit(self, X):
        """
        This function checks whether the smiles strings are stored in the class. Then it checks that X is a list of
        indices specifying which data samples to use. Then, it returns the appropriate smiles strings hot
        encoded.

        :param X: either list of smiles strings or indices.
        :type X: either list of strings or list of ints
        :return: smiles strings hot encoded and the following character
        :rtype: numpy arrays of shape (n_samples, max_length, n_unique characters) and (n_samples, max_length, n_unique_characters)
        """

        if not isinstance(self.smiles, type(None)):
            if not utils.is_positive_integer_or_zero_array(X):
                raise utils.InputError("The indices need to be positive or zero integers.")

            X_hot = np.asarray([self.X_hot[i] for i in X])
            y_hot = np.asarray([self.y_hot[i] for i in X])
        else:
            X_hot, y_hot = self._hot_encode(self._check_smiles(X))

        return X_hot, y_hot

    def _initialise_data_predict(self, X, frag_length):
        """
        X can either be a list of smiles strings or the indices to the samples to be used for prediction. In the latter
        case, the data needs to have been stored inside the class.

        This function takes the smiles strings and extract the first few characters (number specified by the parameter
        frag_length). Prediction will start from these few characters.

        :param X: list of smiles or list of indices
        :type X: list of strings or list of ints
        :param frag_length: number of characters of each smiles strings to use for prediction
        :type frag_length: int
        :return: list of the first fragment of each smile string specified and its one-hot encoded version.
        :rtype: list of strings, numpy array of shape (n_samples, frag_length, n_unique_char)
        """

        # TODO add a check that the frag_length is < than the shortest smile
        # Predictions will start from a 'G'
        if isinstance(X, type(None)):
            X_hot = None
            X_strings = None
        # Predictions will start from a fragment of smiles strings stored in the class
        elif not isinstance(self.smiles, type(None)):
            if not utils.is_positive_integer_or_zero_array(X):
                raise utils.InputError("The indices need to be positive or zero integers.")

            X_hot = np.asarray([self.X_hot[i][:frag_length] for i in X])
            X_strings = np.asarray([self.smiles[i][:frag_length] for i in X])
        # Predictions will start from fragments of smiles strings passed through the argument
        else:
            X = self._check_smiles(X)
            X_strings = [item[:frag_length] for item in X]  # No 'G' is added because it is done in the Hot encode function
            X_hot, y_hot = self._hot_encode(X_strings)
            X_strings = ["G" + item[:frag_length] for item in X] # Now the G is needed since the hot encoded fragments will have it

        return X_strings, X_hot

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

        if not isinstance(self.smiles, type(None)):
            self.smiles = new_molecules

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
        
    def _predict(self, X_strings, X_hot, model):
        """
        This function either takes in  smiles strings fragments and their hot encoded version, or it takes in nothing
        and generates smiles strings from scratch.

        :param X_strings: Fragment of smiles string or None
        :param X_hot: Hot encoded version of X
        :param model: the model to be used (either the current model or a loaded model)
        :return: predictions of smiles strings
        """
        
        if isinstance(X_hot, type(None)):
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

            if y_pred[-1] == 'E':
                y_pred = y_pred[:-1]
            if y_pred[0] == 'G':
                y_pred = y_pred[1:]

            y_pred = re.sub("A", "", y_pred)

            return [y_pred]
                    
        else:
            n_samples = len(X_hot)

            all_predictions = []

            for n in range(0, n_samples):
                X_frag = X_hot[n]  # shape (fragment_length, n_feat)
                y_pred = X_strings[n]

                while y_pred[-1] != 'E':
                    X_pred = np.reshape(X_frag, (1, X_frag.shape[0], X_frag.shape[1]))  # shape (1, fragment_length, n_feat)

                    out = model.predict(X_pred)[0][-1]
                    idx_out = np.argmax(out)

                    y_pred += self.idx_to_char[idx_out]
                    # X_pred = self._hot_encode([y_pred[1:]])[0][0]

                    X_pred_temp = np.zeros((X_frag.shape[0]+1, X_frag.shape[1]))
                    X_pred_temp[:-1] = X_frag
                    X_pred_temp[-1][idx_out] = 1
                    X_frag = X_pred_temp

                    if len(y_pred) == 100:
                        break

                if y_pred[-1] == 'E':
                    y_pred = y_pred[:-1]
                if y_pred[0] == 'G':
                    y_pred = y_pred[1:]
                y_pred = re.sub("A", "", y_pred)
                all_predictions.append(y_pred)

            return all_predictions

