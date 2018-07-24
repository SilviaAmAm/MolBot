import numpy as np

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_numeric_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    return False

def _is_integer_array(x):
    if is_numeric_array(x):
        if (np.asarray(x, dtype = float) == np.asarray(x, dtype = int)).all():
            return True
    return False

def _is_positive_or_zero_array(x):
    if is_numeric_array(x) and (np.asarray(x, dtype = float) >= 0).all():
        return True
    return False

def is_positive_integer_or_zero_array(x):
    return (_is_integer_array(x) and _is_positive_or_zero_array(x))

class InputError(Exception):
    pass