import numpy as np
from sklearn.utils.validation import check_array

def is_positive(x):
    return (not is_array_like(x) and _is_numeric(x) and x > 0)

def is_positive_or_zero(x):
    return (not is_array_like(x) and _is_numeric(x) and x >= 0)

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_positive_integer(x):
    return (not is_array_like(x) and _is_integer(x) and x > 0)

def is_positive_integer_or_zero(x):
    return (not is_array_like(x) and _is_integer(x) and x >= 0)

def is_string(x):
    return isinstance(x, str)

def is_none(x):
    return isinstance(x, type(None))

def is_dict(x):
    return isinstance(x, dict)

def _is_numeric(x):
    return isinstance(x, (float, int))

def is_numeric_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    return False

def _is_integer(x):
    return (_is_numeric(x) and (float(x) == int(x)))

# will intentionally accept 0, 1 as well
def is_bool(x):
    return (x in (True, False))

def is_non_zero_integer(x):
    return (_is_integer(x) and x != 0)

def _is_positive_array(x):
    if is_numeric_array(x) and (np.asarray(x, dtype = float) > 0).all():
        return True
    return False

def _is_positive_or_zero_array(x):
    if is_numeric_array(x) and (np.asarray(x, dtype = float) >= 0).all():
        return True
    return False

def _is_integer_array(x):
    if is_numeric_array(x):
        if (np.asarray(x, dtype = float) == np.asarray(x, dtype = int)).all():
            return True
    return False

def is_positive_integer_array(x):
    return (_is_integer_array(x) and _is_positive_array(x))


def is_positive_integer_or_zero_array(x):
    return (_is_integer_array(x) and _is_positive_or_zero_array(x))

def ceil(a, b):
    """
    Returns a/b rounded up to nearest integer.
    """
    return -(-a//b)

class InputError(Exception):
    pass


def set_tensorboard(tb):
    if is_bool(tb):
        return tb
    else:
        raise InputError("Parameter Tensorboard should be either true or false. Got %s" % (str(tb)))


def set_hidden_neurons(h):
    if is_positive_integer(h):
        return h
    else:
        raise InputError("The number of hidden neurons should be a positive non zero integer. Got %s." % (str(h)))


def set_dropout(drop):
    if drop >= 0.0 and drop < 1.0:
        return drop
    else:
        raise InputError(
            "The dropout rate should be between 0 and 1. Got %s." % (str(drop)))


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
