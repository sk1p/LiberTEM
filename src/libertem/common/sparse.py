import numpy as np

from .array_backends import get_backend, for_backend, NUMPY, SPARSE_COO, SPARSE_BACKENDS


def to_dense(a):
    res = for_backend(a, NUMPY)
    if res.flags.c_contiguous:
        return res
    else:
        return np.array(res)


def to_sparse(a):
    return for_backend(a, SPARSE_COO)


def is_sparse(a):
    return get_backend(a) in SPARSE_BACKENDS
