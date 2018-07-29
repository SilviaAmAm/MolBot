import numpy as np

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