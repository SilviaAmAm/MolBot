# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

import numpy as np

from sklearn.utils.validation import check_array

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_positive_integer(x):
    return (not is_array_like(x) and _is_integer(x) and x > 0)

def _is_numeric(x):
    return isinstance(x, (float, int))

def _is_integer(x):
    return (_is_numeric(x) and (float(x) == int(x)))

def ceil(a, b):
    """
    Returns a/b rounded up to nearest integer.
    """
    return -(-a//b)

class InputError(Exception):
    pass

def set_tensorboard(tb):
    if (tb in (True, False)):
        return tb
    else:
        raise InputError("Parameter Tensorboard should be either true or false. Got %s" % (str(tb)))

def set_hidden_neurons(h):
    if is_positive_integer(h):
        return h
    else:
        raise InputError("The number of hidden neurons should be a positive non zero integer. Got %s." % (str(h)))

def _check_float_perc(x):
    if x >= 0.0 and x < 1.0:
        return x
    else:
        raise InputError(
            "Parameter that should be between 0 and 1 is %s." % (str(x)))

def set_dropout(drop):
    return _check_float_perc(drop)

def set_validation(validation):
    return _check_float_perc(validation)

def set_provisional_batch_size(batch_size):
    if batch_size != "auto":
        if not is_positive_integer(batch_size):
            raise InputError("Expected 'batch_size' to be a positive integer. Got %s" % str(batch_size))
        elif batch_size == 1:
            raise InputError("batch_size must be larger than 1.")
        return int(batch_size)
    else:
        return batch_size

def set_batch_size(batch_size, n_samples):
    if batch_size == 'auto':
        batch_size = min(100, n_samples)
    else:
        if batch_size > n_samples:
            print("Warning: batch_size larger than sample size. It is going to be clipped")
            return n_samples
        else:
            batch_size = batch_size

    better_batch_size = ceil(n_samples, ceil(n_samples, batch_size))

    return better_batch_size

def set_epochs(epochs):
    if is_positive_integer(epochs):
        return epochs
    else:
        raise InputError("The number of epochs should be a positive integer. Got %s." % (str(epochs)))

def set_learning_rate(lr):
    """
    This function checks that the learning rate is a float larger than zero

    :param lr: learning rate
    :type: float > 0
    :return: approved learning rate
    """
    check_lr(lr)
    return lr

def check_X_y(X, y):
    """
    This function checks that the input and output for the estimator have the correct dimensions.
    :param X: Input one-hot-encoded padded smiles strings
    :type X: np.array of shape (n_samples, max_len, n_char)
    :param y: Output one-hot-encoded padded smiles strings
    :type y: np.array of shape (n_samples, max_len, n_char)

    :return: approved arrays X and y
    :rtype: two np.arrays of shape (n_samples, max_len, n_char)
    """
    if y is None:
        raise ValueError("y cannot be None")

    if len(X.shape) != 3 or len(y.shape) != 3:
        raise ValueError("The RNN expects a 3-dimensional input. Got inputs with shape %s and %s", (str(X.shape), str(y.shape)))

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y don't have the same number of samples.")

    X = check_array(X, allow_nd=True)
    y = check_array(y, allow_nd=True)

    return X, y

def check_temperature(T):
    """
    This function checks the Temperature parameter for the modified softmax.

    :param T: Temperature
    :type T: float
    :return: None
    """
    if T <= 0:
        raise ValueError("Temperature parameter should be > 0.0. Got %s" % (str(T)))

def check_maxlength(ml):
    """
    This function checks that the maximum length for the predicted smiles is a positive integer.

    :param ml: maximum smiles length
    :type ml: int
    :return: None
    """
    if not isinstance(ml, type(int)) and ml <= 0:
        raise ValueError("The length of the predicted strings should be an integer larger than 0.")

def check_ep(ep):
    if not is_positive_integer(ep):
        raise InputError("The number of episodes should be a positive integer. Got %s." % (str(ep)))

def check_lr(lr):
    if isinstance(lr, (float, int)):
        if lr < 0.0:
            raise InputError("The learning rate should be larger than 0.")
    else:
        raise InputError("The learning rate should be number larger than 0.")

def check_sigma(sigma):
    try:
        sigma = float(sigma)
    except ValueError:
        raise InputError("Sigma should be a float. Got %s" % str(sigma))

def root_mean_squared_err(y_true, y_pred):

    return np.mean(np.square((y_true-y_pred)))

def valid_and_unique(*args):
    """
    This function takes in a list of filenames all corresponding to csv files containing smiles_fine_tuning_1 strings generated *from
    the same run* so that the percentage of valid and unique smiles_fine_tuning_1 can be calculated, as well as their error.

    :param args: list of filenames
    :return: percentage valid, percentage unique, error valid, error unique
    :rtype: float, float, float, float
    """

    try:
        from rdkit import Chem
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.error')
    except ModuleNotFoundError:
        print("You need RDKit to use this function.")
        exit()

    smiles_files = []
    for item in args[0]:
        smiles_files.append(item)

    tot_n_tot = []
    unique_n_tot = []
    invalid_tot = []

    for file in smiles_files:
        smiles_1 = []
        f = open(file, "r")
        for line in f:
            line_split = line.rstrip()
            smiles_1.append(line_split)
        f.close()
        tot_n = len(smiles_1)
        tot_n_tot.append(tot_n)
        smiles_1 = set(smiles_1)
        unique_n = len(smiles_1)
        unique_n_tot.append(unique_n)

        invalid = 0
        valid_smiles_05 = []

        for smile in smiles_1:
            mol = Chem.MolFromSmiles(smile)
            if isinstance(mol, type(None)):
                invalid += 1
            else:
                valid_smiles_05.append(mol)
        invalid_tot.append(invalid)

    final_tot_n = np.mean(tot_n_tot)
    final_unique_n = np.mean(unique_n_tot)
    unique_err = np.std(unique_n_tot)
    final_invalid = np.mean(invalid_tot)
    invalid_err = np.std(invalid_tot)

    perc_unique = final_unique_n / final_tot_n * 100
    perc_unique_err = unique_err / final_tot_n * 100

    perc_valid = (final_tot_n - final_invalid) / final_tot_n * 100
    perc_valid_err = invalid_err / final_tot_n * 100

    return perc_valid, perc_unique, perc_valid_err, perc_unique_err

def valid_and_unique_smiles(*args):
    """
    This function takes in a list of lists of smiles generated *from the same run* so that the percentage of valid and unique
    smiles can be calculated, as well as their error.

    :param args: list of lists of smiles
    :return: percentage valid, percentage unique, error valid, error unique
    :rtype: float, float, float, float
    """

    try:
        from rdkit import Chem
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.error')
    except ModuleNotFoundError:
        print("You need RDKit to use this function.")
        exit()

    smiles_runs = []
    for item in args[0]:
        smiles_runs.append(item)

    tot_n_tot = []
    unique_n_tot = []
    invalid_tot = []

    for one_run in smiles_runs:

        tot_n = len(one_run)
        tot_n_tot.append(tot_n)
        smiles_1 = set(one_run)
        unique_n = len(smiles_1)
        unique_n_tot.append(unique_n)

        invalid = 0
        valid_smiles_05 = []

        for smile in smiles_1:
            mol = Chem.MolFromSmiles(smile)
            if isinstance(mol, type(None)):
                invalid += 1
            else:
                valid_smiles_05.append(mol)
        invalid_tot.append(invalid)

    final_tot_n = np.mean(tot_n_tot)
    final_unique_n = np.mean(unique_n_tot)
    unique_err = np.std(unique_n_tot)
    final_invalid = np.mean(invalid_tot)
    invalid_err = np.std(invalid_tot)

    perc_unique = final_unique_n / final_tot_n * 100
    perc_unique_err = unique_err / final_tot_n * 100

    perc_valid = (final_tot_n - final_invalid) / final_tot_n * 100
    perc_valid_err = invalid_err / final_tot_n * 100

    return perc_valid, perc_unique, perc_valid_err, perc_unique_err