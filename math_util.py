from contextlib import contextmanager
import numpy as np
from scipy.linalg import norm


EPS = np.finfo('float64').eps


def asvector(x):
    return x.reshape(x.size)

def ascolvector(x):
    return x.reshape(x.size, 1)


def asrowvector(x):
    return x.reshape(1, x.size)


def column_norms(x):
    return np.sqrt(np.add.reduce((x*x), axis=0))


def l2_normalize(x):
    x = np.asarray(x, dtype='float64')
    if x.ndim == 1:
        norm_ = np.fmax(norm(x), 100*EPS)
        return x / norm_
    elif x.ndim == 2:
        norms = np.fmax(column_norms(x), 100*EPS)
        return x / asrowvector(norms)


def cosine_similarity(a, b):
    """
    Computes the cosine similarity of the columns of A with the columns of B.
    Returns a matrix X such that Xij is the cosine similarity of A_i with B_j.
    In the case that norm(A_i) = 0 or norm(B_j) = 0, this
    implementation will return X_ij = 0.  If norm(A_i) = 0 AND norm(B_i) = 0,
    then X_ii = 0 as well.
    """
    if a.ndim == 1:
        a = ascolvector(a)
    if b.ndim == 1:
        b = ascolvector(b)
    assert a.shape[0] == b.shape[0]

    return l2_normalize(a).T.dot(l2_normalize(b))


def avk(v, k):
    """
    Mean resultant length of a vMF in dimension v with concentration k.
    """
    return (np.sqrt((v/k)**2+4) - v/k)/2.0


@contextmanager
def numpy_random_seed_temporarily(seed=None):
    """
    Do something with numpy random seed set to <seed>, then revert it.  Ex:
    with np_random_seed_temporarily(42):
       select_some_random_numbers()
    """
    random_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        yield
    finally:
        np.random.set_state(random_state)


def sum_lt(x, dim=0):
    """
    Computes the cumulative sum of an array along some dimension, up to but not including element i.
    Ex.: sum_lt([1, 2, 3]) == [0, 1, 3]
    """
    x = np.asarray(x)
    return np.subtract(x.cumsum(dim), x)